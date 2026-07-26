from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from jsonschema import Draft202012Validator
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iaei.contracts import (
    ContractError,
    load_json,
    load_yaml,
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
    validate_neural_seed_governance_alignment,
)
from iaei.modeling.candidates import build_feature_preprocessor
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets
from iaei.v2.neural_models import (
    build_neural_model,
    configure_deterministic_cpu,
    model_identity,
)


class NeuralForecastingError(RuntimeError):
    """Raised when Gate 6C2 evidence violates a governed boundary."""


@dataclass(frozen=True)
class NeuralForecastingArtifacts:
    seed_results: pd.DataFrame
    outer_fold_results: pd.DataFrame
    candidate_leaderboard: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame
    trial_evidence: dict[str, Any]
    execution_manifest: dict[str, Any]
    promotion_recommendation: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_payload(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractError(f"{label} failed validation:\n{details}")


def _feature_frame(silver: pd.DataFrame, model_contract: dict[str, Any]) -> pd.DataFrame:
    policy = model_contract["feature_policy"]
    numeric = [str(value) for value in policy["numeric_features"]]
    categorical = [str(value) for value in policy["categorical_features"]]
    requested = numeric + categorical
    missing = sorted(set(requested).difference(silver.columns))
    if missing:
        raise NeuralForecastingError(f"Silver neural fields are missing: {missing}")
    features = silver.loc[:, requested].copy()
    for column in numeric:
        features[column] = pd.to_numeric(features[column], errors="raise")
    for column in categorical:
        values = features[column].astype("object")
        features[column] = values.where(pd.notna(values), np.nan)
    return features


def _load_outer_folds(path: Path) -> tuple[list[ChronologicalFold], dict[str, Any]]:
    payload = load_json(path)
    folds = [ChronologicalFold(**item) for item in payload["folds"]]
    if len(folds) != 4:
        raise NeuralForecastingError("Gate 6C2 requires exactly four outer folds")
    if [fold.fold_id for fold in folds] != [1, 2, 3, 4]:
        raise NeuralForecastingError("Gate 6C2 fold identifiers are not canonical")
    return folds, payload


def _reference_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("fold_id", kind="stable").reset_index(drop=True)
    required = {"fold_id", "mae", "peak_mae", "validation_rows"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise NeuralForecastingError(f"V1 validation evidence is missing: {missing}")
    if frame["fold_id"].astype(int).tolist() != [1, 2, 3, 4]:
        raise NeuralForecastingError("V1 evidence does not contain four canonical folds")
    return frame


def _window_tensor(
    transformed: np.ndarray,
    origins: np.ndarray,
    *,
    context_length: int,
) -> torch.Tensor:
    offsets = np.arange(context_length, dtype=np.int64)
    positions = origins[:, None] - context_length + 1 + offsets[None, :]
    if positions.min() < 0:
        raise NeuralForecastingError("A neural origin lacks the required causal context")
    windows = np.asarray(transformed[positions], dtype=np.float32)
    if not np.isfinite(windows).all():
        raise NeuralForecastingError("Neural context windows contain nonfinite values")
    return torch.from_numpy(windows)


def _target_tensor(
    target: pd.Series,
    origins: np.ndarray,
) -> tuple[torch.Tensor, float, float]:
    values = target.iloc[origins].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise NeuralForecastingError("Neural training targets contain nonfinite values")
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise NeuralForecastingError("Neural target scale is not positive and finite")
    normalized = (values - mean) / scale
    return torch.from_numpy(normalized.astype(np.float32)), mean, scale


def _fit_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    configuration: dict[str, Any],
    seed: int,
) -> None:
    batch_size = int(configuration["batch_size"])
    max_epochs = int(configuration["max_epochs"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    loss_function = nn.L1Loss()
    model.train()
    for _ in range(max_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            if not torch.isfinite(loss):
                raise NeuralForecastingError("Neural training loss became nonfinite")
            loss.backward()
            optimizer.step()


def _predict(
    model: nn.Module,
    values: torch.Tensor,
    *,
    target_mean: float,
    target_scale: float,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            normalized = model(values[start : start + batch_size])
            prediction = normalized.mul(target_scale).add(target_mean)
            outputs.append(prediction.cpu().numpy().astype(float))
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise NeuralForecastingError("Neural predictions contain nonfinite values")
    return result


def _model_package(
    model: nn.Module,
    *,
    preprocessor: Any,
    target_mean: float,
    target_scale: float,
    algorithm_id: str,
    context_length: int,
    input_dimension: int,
    configuration: dict[str, Any],
) -> bytes:
    state_buffer = io.BytesIO()
    torch.save(model.state_dict(), state_buffer)
    return pickle.dumps(
        {
            "algorithm_id": algorithm_id,
            "context_length": context_length,
            "input_dimension": input_dimension,
            "configuration": configuration,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "preprocessor": preprocessor,
            "state_dict": state_buffer.getvalue(),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _restore_prediction(package: bytes, values: torch.Tensor) -> np.ndarray:
    payload = pickle.loads(package)
    restored = build_neural_model(
        str(payload["algorithm_id"]),
        context_length=int(payload["context_length"]),
        input_dimension=int(payload["input_dimension"]),
        configuration=dict(payload["configuration"]),
    )
    state = torch.load(io.BytesIO(payload["state_dict"]), map_location="cpu", weights_only=True)
    restored.load_state_dict(state)
    return _predict(
        restored,
        values,
        target_mean=float(payload["target_mean"]),
        target_scale=float(payload["target_scale"]),
    )


def _latency_ms_per_1000_rows(model: nn.Module, values: torch.Tensor) -> float:
    sample = values[: min(1000, len(values))]
    if len(sample) == 0:
        raise NeuralForecastingError("Neural latency sample is empty")
    timings: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(7):
            started = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - started) * 1000.0)
    scale = 1000.0 / float(len(sample))
    return float(np.percentile(np.asarray(timings) * scale, 95))


def _peak_memory_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except (ImportError, AttributeError):
        return 0.0


def _environment_lock(root: Path, output_directory: Path) -> tuple[Path, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    path = output_directory / "environment_lock.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path, _sha256_path(path)


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise NeuralForecastingError("Could not resolve the Gate 6C2 execution commit")
    return value


def _candidate_is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    objectives = [
        "mean_mae",
        "mean_peak_mae",
        "across_seed_mae_standard_deviation",
        "mean_model_size_bytes",
        "max_p95_inference_latency_ms_per_1000_rows",
        "max_peak_memory_mb",
    ]
    row_values = [float(row[name]) for name in objectives]
    for other in candidates.itertuples(index=False):
        if other.algorithm_id == row["algorithm_id"]:
            continue
        other_values = [float(getattr(other, name)) for name in objectives]
        no_worse = all(left <= right for left, right in zip(other_values, row_values, strict=True))
        strictly_better = any(
            left < right for left, right in zip(other_values, row_values, strict=True)
        )
        if no_worse and strictly_better:
            return True
    return False


def _aggregate_results(
    seed_results: pd.DataFrame,
    reference: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    for (algorithm_id, fold_id), group in seed_results.groupby(
        ["algorithm_id", "fold_id"], sort=True
    ):
        reference_row = reference.loc[reference["fold_id"].eq(fold_id)].iloc[0]
        fold_rows.append(
            {
                "algorithm_id": algorithm_id,
                "fold_id": int(fold_id),
                "seed_count": int(group["seed"].nunique()),
                "validation_rows_per_seed": int(group["validation_rows"].iloc[0]),
                "peak_rows_per_seed": int(group["peak_rows"].iloc[0]),
                "mean_mae": float(group["mae"].mean()),
                "mae_standard_deviation": float(group["mae"].std(ddof=0)),
                "minimum_mae": float(group["mae"].min()),
                "maximum_mae": float(group["mae"].max()),
                "mean_peak_mae": float(group["peak_mae"].mean()),
                "peak_mae_standard_deviation": float(group["peak_mae"].std(ddof=0)),
                "reference_mae": float(reference_row["mae"]),
                "reference_peak_mae": float(reference_row["peak_mae"]),
                "relative_mae_improvement": float(
                    (float(reference_row["mae"]) - group["mae"].mean())
                    / float(reference_row["mae"])
                ),
                "relative_peak_mae_change": float(
                    (group["peak_mae"].mean() - float(reference_row["peak_mae"]))
                    / float(reference_row["peak_mae"])
                ),
                "maximum_prediction_origin": int(group["maximum_prediction_origin"].max()),
                "maximum_target_dependency": int(group["maximum_target_dependency"].max()),
            }
        )
    outer_fold_results = pd.DataFrame(fold_rows).sort_values(
        ["algorithm_id", "fold_id"], kind="stable"
    )

    reference_mean_mae = float(reference["mae"].mean())
    reference_mean_peak_mae = float(reference["peak_mae"].mean())
    promotion = contract["promotion"]
    resources = contract["resource_constraints"]
    leaderboard_rows: list[dict[str, Any]] = []
    candidate_evidence: list[dict[str, Any]] = []

    for algorithm_id, group in seed_results.groupby("algorithm_id", sort=True):
        fold_group = outer_fold_results.loc[
            outer_fold_results["algorithm_id"].eq(algorithm_id)
        ]
        seed_means = group.groupby("seed", sort=True)["mae"].mean()
        seed_peak_means = group.groupby("seed", sort=True)["peak_mae"].mean()
        mean_mae = float(group["mae"].mean())
        mean_peak_mae = float(group["peak_mae"].mean())
        relative_mae_improvement = (reference_mean_mae - mean_mae) / reference_mean_mae
        relative_peak_change = (
            mean_peak_mae - reference_mean_peak_mae
        ) / reference_mean_peak_mae
        positive_folds = int(fold_group["relative_mae_improvement"].gt(0.0).sum())
        maximum_fold_degradation = float(
            np.maximum(-fold_group["relative_mae_improvement"].to_numpy(), 0.0).max()
        )
        seed_std = float(seed_means.std(ddof=0))
        peak_seed_std = float(seed_peak_means.std(ddof=0))
        resource_limits_passed = bool(
            group["peak_memory_mb"].le(float(resources["maximum_peak_memory_mb"])).all()
            and group["wall_clock_seconds"].le(
                float(resources["maximum_candidate_wall_clock_minutes"]) * 60.0
            ).all()
        )
        cpu_portability_passed = bool(group["cpu_portability_passed"].all())
        chronology_passed = bool(
            group["maximum_prediction_origin"]
            .lt(int(contract["data_boundary"]["maximum_prediction_origin_exclusive"]))
            .all()
            and group["maximum_target_dependency"]
            .lt(int(contract["data_boundary"]["maximum_target_dependency_exclusive"]))
            .all()
        )
        leaderboard_rows.append(
            {
                "algorithm_id": algorithm_id,
                "mean_mae": mean_mae,
                "mean_peak_mae": mean_peak_mae,
                "relative_mae_improvement_vs_v1": relative_mae_improvement,
                "relative_peak_mae_change_vs_v1": relative_peak_change,
                "positive_outer_folds": positive_folds,
                "maximum_single_fold_mae_relative_degradation": maximum_fold_degradation,
                "across_seed_mae_standard_deviation": seed_std,
                "across_seed_peak_mae_standard_deviation": peak_seed_std,
                "outer_fold_mae_dispersion": float(fold_group["mean_mae"].std(ddof=0)),
                "mean_model_size_bytes": float(group["model_size_bytes"].mean()),
                "max_p95_inference_latency_ms_per_1000_rows": float(
                    group["p95_inference_latency_ms_per_1000_rows"].max()
                ),
                "max_peak_memory_mb": float(group["peak_memory_mb"].max()),
                "total_wall_clock_seconds": float(group["wall_clock_seconds"].sum()),
                "chronology_passed": chronology_passed,
                "resource_limits_passed": resource_limits_passed,
                "cpu_portability_passed": cpu_portability_passed,
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_mae", kind="stable")
    leaderboard["pareto_eligible"] = [
        not _candidate_is_dominated(row, leaderboard) for _, row in leaderboard.iterrows()
    ]
    leaderboard["promotion_eligible"] = (
        leaderboard["relative_mae_improvement_vs_v1"].ge(
            float(promotion["minimum_mean_validation_mae_relative_improvement"])
        )
        & leaderboard["relative_peak_mae_change_vs_v1"].le(
            float(promotion["maximum_peak_state_mae_relative_degradation"])
        )
        & leaderboard["positive_outer_folds"].ge(
            int(promotion["minimum_positive_outer_folds"])
        )
        & leaderboard["maximum_single_fold_mae_relative_degradation"].le(
            float(promotion["maximum_single_fold_mae_relative_degradation"])
        )
        & leaderboard["across_seed_mae_standard_deviation"].le(
            float(promotion["maximum_across_seed_mae_standard_deviation"])
        )
        & leaderboard["chronology_passed"]
        & leaderboard["resource_limits_passed"]
        & leaderboard["cpu_portability_passed"]
        & leaderboard["pareto_eligible"]
    )

    for row in leaderboard.itertuples(index=False):
        seed_count = int(
            seed_results.loc[seed_results["algorithm_id"].eq(row.algorithm_id), "seed"].nunique()
        )
        candidate_evidence.append(
            {
                "schema_version": "1.0.0",
                "gate": "6C",
                "candidate_id": row.algorithm_id,
                "seed_count": seed_count,
                "outer_fold_count": 4,
                "validation_origin_count_per_seed": int(reference["validation_rows"].sum()),
                "aggregate": {
                    "mean_mae": float(row.mean_mae),
                    "mean_peak_mae": float(row.mean_peak_mae),
                    "relative_mae_improvement_vs_v1": float(
                        row.relative_mae_improvement_vs_v1
                    ),
                    "positive_outer_folds": int(row.positive_outer_folds),
                },
                "stability": {
                    "across_seed_mae_standard_deviation": float(
                        row.across_seed_mae_standard_deviation
                    ),
                    "across_seed_peak_mae_standard_deviation": float(
                        row.across_seed_peak_mae_standard_deviation
                    ),
                    "outer_fold_mae_dispersion": float(row.outer_fold_mae_dispersion),
                },
                "constraints": {
                    "chronology_passed": bool(row.chronology_passed),
                    "locked_test_excluded": True,
                    "v1_immutable": True,
                    "resource_limits_passed": bool(row.resource_limits_passed),
                    "cpu_portability_passed": bool(row.cpu_portability_passed),
                },
                "promotion_eligible": bool(row.promotion_eligible),
            }
        )
    return outer_fold_results, leaderboard, candidate_evidence


def _recommendation(leaderboard: pd.DataFrame) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in leaderboard.itertuples(index=False):
        failed: list[str] = []
        if row.relative_mae_improvement_vs_v1 < 0.01:
            failed.append("aggregate_mae_improvement")
        if row.relative_peak_mae_change_vs_v1 > 0.01:
            failed.append("peak_state_mae")
        if row.positive_outer_folds < 3:
            failed.append("positive_outer_folds")
        if row.maximum_single_fold_mae_relative_degradation > 0.02:
            failed.append("single_fold_degradation")
        if row.across_seed_mae_standard_deviation > 0.15:
            failed.append("across_seed_stability")
        if not row.resource_limits_passed:
            failed.append("resource_limits")
        if not row.cpu_portability_passed:
            failed.append("cpu_portability")
        if not row.pareto_eligible:
            failed.append("pareto_eligibility")
        candidates.append(
            {
                "candidate_id": row.algorithm_id,
                "promotion_eligible": bool(row.promotion_eligible),
                "failed_requirements": failed,
            }
        )
    eligible = [item["candidate_id"] for item in candidates if item["promotion_eligible"]]
    return {
        "schema_version": "1.0.0",
        "gate": "6C",
        "subgate": "6C2",
        "status": "validation_complete_pending_human_decision",
        "automatic_promotion_permitted": False,
        "human_decision_required": True,
        "eligible_candidates": eligible,
        "candidates": candidates,
        "recommended_next_gate": "6C3",
    }


def execute_neural_forecasting_gate(root: Path) -> NeuralForecastingArtifacts:
    contract = validate_neural_seed_governance_alignment(
        validate_neural_forecasting_contract()
    )
    closure = validate_gate_6c1_closure_manifest()
    if closure["status"] != "closed":
        raise NeuralForecastingError("Gate 6C1 is not closed")

    boundary = contract["data_boundary"]
    model_contract = load_yaml(root / str(boundary["model_contract_path"]))
    target_contract = load_yaml(root / str(boundary["target_contract_path"]))
    silver = pd.read_parquet(root / str(boundary["silver_path"]))
    folds, fold_payload = _load_outer_folds(root / str(boundary["outer_folds_path"]))
    reference = _reference_frame(root / str(boundary["incumbent_results_path"]))

    maximum_origin_exclusive = int(boundary["maximum_prediction_origin_exclusive"])
    maximum_dependency_exclusive = int(boundary["maximum_target_dependency_exclusive"])
    if int(fold_payload["test_purge_start"]) != maximum_origin_exclusive:
        raise NeuralForecastingError("Gate 6C2 prediction boundary changed")
    if int(fold_payload["locked_test_start"]) != maximum_dependency_exclusive:
        raise NeuralForecastingError("Gate 6C2 dependency boundary changed")

    features = _feature_frame(silver, model_contract)
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    regression_target = str(model_contract["objectives"]["regression_target"])
    output_directory = root / str(contract["outputs"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    environment_path, environment_sha = _environment_lock(root, output_directory)
    execution_commit = _git_commit(root)

    seed_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    seed_evidence_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    seed_schema = load_json(root / "schemas" / "neural_seed_evidence.schema.json")

    for fold in folds:
        if fold.validation_stop > maximum_origin_exclusive:
            raise NeuralForecastingError("Neural validation enters the locked-test purge")
        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[fold.train_start : fold.train_stop] = True
        targets = build_supervised_targets(
            silver,
            training_mask,
            contract=target_contract,
        )
        target_frame = targets.frame
        peak_threshold = float(targets.peak_threshold_kwh)

        preprocessor = build_feature_preprocessor(model_contract)
        preprocessor.fit(features.iloc[fold.train_start : fold.train_stop])
        transformed = np.asarray(
            preprocessor.transform(features.iloc[: fold.validation_stop]),
            dtype=np.float32,
        )
        if not np.isfinite(transformed).all():
            raise NeuralForecastingError("Preprocessed neural features contain nonfinite values")
        input_dimension = int(transformed.shape[1])

        validation_origins = np.arange(
            fold.validation_start,
            fold.validation_stop,
            dtype=np.int64,
        )
        validation_target = target_frame.loc[
            validation_origins, regression_target
        ].astype(float)
        if validation_target.isna().any():
            raise NeuralForecastingError(
                f"Fold {fold.fold_id} contains missing validation targets"
            )
        peak_mask = validation_target.ge(peak_threshold).to_numpy()
        if not peak_mask.any():
            raise NeuralForecastingError(f"Fold {fold.fold_id} contains no peak states")

        for candidate in contract["candidate_families"]:
            algorithm_id = str(candidate["algorithm_id"])
            context_length = int(candidate["context_length"])
            configuration = dict(candidate["configuration"])
            training_origins = np.arange(
                max(fold.train_start + context_length - 1, context_length - 1),
                fold.train_stop,
                dtype=np.int64,
            )
            train_x = _window_tensor(
                transformed,
                training_origins,
                context_length=context_length,
            )
            train_y, target_mean, target_scale = _target_tensor(
                target_frame[regression_target].astype(float),
                training_origins,
            )
            validation_x = _window_tensor(
                transformed,
                validation_origins,
                context_length=context_length,
            )

            for seed in contract["search"]["seeds"]:
                seed = int(seed)
                configure_deterministic_cpu(seed)
                started = time.perf_counter()
                try:
                    model = build_neural_model(
                        algorithm_id,
                        context_length=context_length,
                        input_dimension=input_dimension,
                        configuration=configuration,
                    )
                    identity = model_identity(
                        algorithm_id,
                        model,
                        context_length=context_length,
                        input_dimension=input_dimension,
                    )
                    _fit_model(
                        model,
                        train_x,
                        train_y,
                        configuration=configuration,
                        seed=seed,
                    )
                    prediction = _predict(
                        model,
                        validation_x,
                        target_mean=target_mean,
                        target_scale=target_scale,
                    )
                    package = _model_package(
                        model,
                        preprocessor=preprocessor,
                        target_mean=target_mean,
                        target_scale=target_scale,
                        algorithm_id=algorithm_id,
                        context_length=context_length,
                        input_dimension=input_dimension,
                        configuration=configuration,
                    )
                    restored_prediction = _restore_prediction(package, validation_x)
                    cpu_portability_passed = bool(
                        np.allclose(prediction, restored_prediction, rtol=0.0, atol=1e-6)
                    )
                    latency = _latency_ms_per_1000_rows(model, validation_x)
                    wall_clock_seconds = float(time.perf_counter() - started)
                    peak_memory_mb = _peak_memory_mb()
                    mae = float(mean_absolute_error(validation_target, prediction))
                    peak_mae = float(
                        mean_absolute_error(
                            validation_target.to_numpy()[peak_mask],
                            prediction[peak_mask],
                        )
                    )
                    if not all(
                        math.isfinite(value)
                        for value in (
                            mae,
                            peak_mae,
                            latency,
                            wall_clock_seconds,
                            peak_memory_mb,
                        )
                    ):
                        raise NeuralForecastingError("Neural evidence contains nonfinite values")

                    maximum_prediction_origin = int(validation_origins.max())
                    maximum_target_dependency = maximum_prediction_origin + 1
                    seed_record = {
                        "schema_version": "1.0.0",
                        "gate": "6C",
                        "candidate_id": algorithm_id,
                        "seed": seed,
                        "fold_id": fold.fold_id,
                        "partition": "validation",
                        "prediction_origin_max": maximum_prediction_origin,
                        "target_dependency_max": maximum_target_dependency,
                        "metrics": {"mae": mae, "peak_mae": peak_mae},
                        "resources": {
                            "wall_clock_seconds": wall_clock_seconds,
                            "peak_memory_mb": peak_memory_mb,
                            "model_size_bytes": int(len(package)),
                            "p95_inference_latency_ms_per_1000_rows": latency,
                        },
                        "portability": {
                            "device": "cpu",
                            "gpu_required": False,
                            "cpu_portability_passed": cpu_portability_passed,
                        },
                        "status": "success",
                    }
                    _validate_payload(
                        seed_record,
                        seed_schema,
                        f"Gate 6C2 seed evidence {algorithm_id}/{seed}/{fold.fold_id}",
                    )
                    seed_evidence_records.append(seed_record)
                    seed_rows.append(
                        {
                            "algorithm_id": algorithm_id,
                            "seed": seed,
                            "fold_id": fold.fold_id,
                            "training_origins": int(len(training_origins)),
                            "validation_rows": int(len(validation_origins)),
                            "peak_rows": int(peak_mask.sum()),
                            "peak_threshold_kwh": peak_threshold,
                            "input_dimension": input_dimension,
                            "parameter_count": identity.parameter_count,
                            "mae": mae,
                            "peak_mae": peak_mae,
                            "model_size_bytes": int(len(package)),
                            "p95_inference_latency_ms_per_1000_rows": latency,
                            "latency_scope": "preprocessed_context_tensor",
                            "peak_memory_mb": peak_memory_mb,
                            "wall_clock_seconds": wall_clock_seconds,
                            "cpu_portability_passed": cpu_portability_passed,
                            "maximum_prediction_origin": maximum_prediction_origin,
                            "maximum_target_dependency": maximum_target_dependency,
                            "status": "success",
                        }
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "algorithm_id": algorithm_id,
                                "seed": seed,
                                "fold_id": fold.fold_id,
                                "row_position": validation_origins,
                                "prediction_origin": timestamps.iloc[
                                    validation_origins
                                ].to_numpy(),
                                "actual": validation_target.to_numpy(),
                                "prediction": prediction,
                                "is_peak_state": peak_mask,
                                "peak_threshold_kwh": peak_threshold,
                            }
                        )
                    )
                except Exception as error:
                    failure_records.append(
                        {
                            "candidate_id": algorithm_id,
                            "seed": seed,
                            "fold_id": fold.fold_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    _write_json(
                        output_directory / "failure_records.json",
                        {
                            "schema_version": "1.0.0",
                            "gate": "6C",
                            "subgate": "6C2",
                            "records": failure_records,
                        },
                    )
                    raise

    seed_results = pd.DataFrame(seed_rows).sort_values(
        ["algorithm_id", "seed", "fold_id"], kind="stable"
    )
    expected_rows = (
        len(contract["candidate_families"])
        * int(contract["search"]["seed_count"])
        * len(folds)
    )
    if len(seed_results) != expected_rows:
        raise NeuralForecastingError("Gate 6C2 seed evidence is incomplete")
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["algorithm_id", "seed", "fold_id", "row_position"], kind="stable"
    )
    outer_fold_results, leaderboard, candidate_evidence = _aggregate_results(
        seed_results,
        reference,
        contract,
    )
    candidate_schema = load_json(
        root / "schemas" / "neural_candidate_evidence.schema.json"
    )
    for record in candidate_evidence:
        _validate_payload(
            record,
            candidate_schema,
            f"Gate 6C2 candidate evidence {record['candidate_id']}",
        )

    recommendation = _recommendation(leaderboard)
    trial_evidence = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "subgate": "6C2",
        "seed_records": seed_evidence_records,
        "candidate_records": candidate_evidence,
        "failure_records": failure_records,
    }

    paths = {
        "seed_results": output_directory / str(contract["outputs"]["seed_results"]),
        "outer_fold_results": output_directory
        / str(contract["outputs"]["outer_fold_results"]),
        "candidate_leaderboard": output_directory
        / str(contract["outputs"]["candidate_leaderboard"]),
        "out_of_fold_predictions": output_directory
        / str(contract["outputs"]["out_of_fold_predictions"]),
        "trial_evidence": output_directory / str(contract["outputs"]["trial_evidence"]),
        "promotion_recommendation": output_directory
        / str(contract["outputs"]["promotion_recommendation"]),
    }
    seed_results.to_csv(paths["seed_results"], index=False, lineterminator="\n")
    outer_fold_results.to_csv(
        paths["outer_fold_results"], index=False, lineterminator="\n"
    )
    leaderboard.to_csv(paths["candidate_leaderboard"], index=False, lineterminator="\n")
    predictions.to_parquet(paths["out_of_fold_predictions"], index=False)
    _write_json(paths["trial_evidence"], trial_evidence)
    _write_json(paths["promotion_recommendation"], recommendation)
    _write_json(
        output_directory / "failure_records.json",
        {
            "schema_version": "1.0.0",
            "gate": "6C",
            "subgate": "6C2",
            "records": failure_records,
        },
    )

    output_hashes = {
        name: _sha256_path(path)
        for name, path in paths.items()
    }
    output_hashes["failure_records"] = _sha256_path(
        output_directory / "failure_records.json"
    )
    manifest = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "subgate": "6C2",
        "status": "validation_complete_pending_human_decision",
        "execution_commit": execution_commit,
        "contract_version": str(contract["contract_version"]),
        "candidate_ids": [
            str(candidate["algorithm_id"]) for candidate in contract["candidate_families"]
        ],
        "seeds": [int(seed) for seed in contract["search"]["seeds"]],
        "seed_fold_evaluation_count": int(len(seed_results)),
        "validation_origins_per_candidate_seed": int(reference["validation_rows"].sum()),
        "prediction_row_count": int(len(predictions)),
        "maximum_prediction_origin": int(
            seed_results["maximum_prediction_origin"].max()
        ),
        "maximum_target_dependency": int(
            seed_results["maximum_target_dependency"].max()
        ),
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "v1_immutable": True,
        "automatic_promotion_permitted": False,
        "human_decision_required": True,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "canonical_device": "cpu",
            "environment_lock_path": str(environment_path.relative_to(root)),
            "environment_lock_sha256": environment_sha,
        },
        "input_hashes": {
            "neural_contract": _sha256_path(
                root / "configs" / "neural_forecasting_contract.yml"
            ),
            "outer_folds": _sha256_path(root / str(boundary["outer_folds_path"])),
            "incumbent_results": _sha256_path(
                root / str(boundary["incumbent_results_path"])
            ),
            "silver": _sha256_path(root / str(boundary["silver_path"])),
        },
        "output_hashes": output_hashes,
        "next_gate": "6C3",
        "blocked_gates": ["6D"],
    }
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    _write_json(manifest_path, manifest)

    return NeuralForecastingArtifacts(
        seed_results=seed_results,
        outer_fold_results=outer_fold_results,
        candidate_leaderboard=leaderboard,
        out_of_fold_predictions=predictions,
        trial_evidence=trial_evidence,
        execution_manifest=manifest,
        promotion_recommendation=recommendation,
    )
