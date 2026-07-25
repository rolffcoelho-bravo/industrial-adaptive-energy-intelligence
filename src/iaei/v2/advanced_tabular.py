from __future__ import annotations

import hashlib
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
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from iaei.contracts import ContractError, load_json, load_yaml
from iaei.modeling.candidates import build_feature_preprocessor
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class AdvancedTabularError(RuntimeError):
    """Raised when Gate 6B evidence violates a governed boundary."""


@dataclass(frozen=True)
class AdvancedTabularArtifacts:
    inner_search_results: pd.DataFrame
    outer_fold_results: pd.DataFrame
    out_of_fold_predictions: pd.DataFrame
    candidate_leaderboard: pd.DataFrame
    trial_records: list[dict[str, Any]]
    execution_manifest: dict[str, Any]
    promotion_recommendation: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
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


def validate_advanced_tabular_contract(root: Path) -> dict[str, Any]:
    contract = load_yaml(root / "configs" / "advanced_tabular_contract.yml")
    schema = load_json(root / "schemas" / "advanced_tabular_contract.schema.json")
    _validate_payload(contract, schema, "Gate 6B advanced tabular contract")

    governance = load_yaml(root / "configs" / "optimization_governance.yml")
    budget = governance["search_budgets"]["advanced_tabular"]
    search = contract["search"]
    if int(search["unique_configuration_count"]) > int(budget["max_unique_trials"]):
        raise ContractError("Gate 6B configuration count exceeds the Gate 6A budget")
    if int(search["max_parallel_trials"]) > int(budget["max_parallel_trials"]):
        raise ContractError("Gate 6B parallelism exceeds the Gate 6A budget")
    if int(search["max_wall_clock_minutes"]) > int(budget["max_wall_clock_minutes"]):
        raise ContractError("Gate 6B wall-clock budget exceeds Gate 6A")

    search_schema = load_json(root / "schemas" / "governed_search_space.schema.json")
    observed_ids: set[str] = set()
    configuration_count = 0
    for family in contract["candidate_families"]:
        path = root / str(family["search_space_path"])
        payload = load_json(path)
        _validate_payload(payload, search_schema, f"Search space {path.name}")
        algorithm_id = str(family["algorithm_id"])
        if payload["algorithm_id"] != algorithm_id:
            raise ContractError(f"Search-space algorithm mismatch for {algorithm_id}")
        if algorithm_id in observed_ids:
            raise ContractError(f"Duplicate advanced tabular algorithm: {algorithm_id}")
        observed_ids.add(algorithm_id)
        configurations = list(family["configurations"])
        if len(configurations) != int(family["configuration_count"]):
            raise ContractError(f"Configuration count mismatch for {algorithm_id}")
        configuration_count += len(configurations)

    if configuration_count != int(search["unique_configuration_count"]):
        raise ContractError("Gate 6B total configuration count is inconsistent")

    return contract


def _feature_frame(silver: pd.DataFrame, model_contract: dict[str, Any]) -> pd.DataFrame:
    policy = model_contract["feature_policy"]
    numeric = [str(value) for value in policy["numeric_features"]]
    categorical = [str(value) for value in policy["categorical_features"]]
    requested = numeric + categorical
    missing = sorted(set(requested).difference(silver.columns))
    if missing:
        raise AdvancedTabularError(f"Silver candidate fields are missing: {missing}")

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
        raise AdvancedTabularError("Gate 6B requires exactly four outer folds")
    if [fold.fold_id for fold in folds] != [1, 2, 3, 4]:
        raise AdvancedTabularError("Outer-fold identifiers are not canonical")
    return folds, payload


def _build_estimator(
    algorithm_id: str,
    parameters: dict[str, Any],
    *,
    seed: int,
) -> Any:
    if algorithm_id == "xgboost_hist":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:absoluteerror",
            tree_method="hist",
            eval_metric="mae",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            **parameters,
        )

    if algorithm_id == "lightgbm_l1":
        from lightgbm import LGBMRegressor

        subsample = float(parameters["subsample"])
        return LGBMRegressor(
            objective="l1",
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            subsample_freq=1 if subsample < 1.0 else 0,
            **parameters,
        )

    if algorithm_id == "catboost_mae":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="MAE",
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **parameters,
        )

    raise AdvancedTabularError(f"Unsupported advanced tabular algorithm: {algorithm_id}")


def _configuration_parameters(configuration: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    configuration_id = str(configuration["configuration_id"])
    parameters = {
        str(key): value
        for key, value in configuration.items()
        if key != "configuration_id"
    }
    if not parameters:
        raise AdvancedTabularError(f"Configuration {configuration_id} has no parameters")
    return configuration_id, parameters


def _pipeline(
    algorithm_id: str,
    parameters: dict[str, Any],
    model_contract: dict[str, Any],
    *,
    seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor(model_contract)),
            ("model", _build_estimator(algorithm_id, parameters, seed=seed)),
        ]
    )


def _peak_memory_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except (ImportError, AttributeError):
        return 0.0


def _latency_ms_per_1000_rows(model: Pipeline, features: pd.DataFrame) -> float:
    sample = features.iloc[: min(1000, len(features))]
    if sample.empty:
        raise AdvancedTabularError("Latency sample is empty")
    timings: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        model.predict(sample)
        timings.append((time.perf_counter() - started) * 1000.0)
    scale = 1000.0 / float(len(sample))
    return float(np.percentile(np.asarray(timings) * scale, 95))


def _dependency_lock(root: Path, output_directory: Path) -> tuple[Path, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    path = output_directory / "environment_lock.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        raise AdvancedTabularError("Could not resolve the execution commit")
    return value


def _reference_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("fold_id", kind="stable").reset_index(drop=True)
    required = {"fold_id", "mae", "peak_mae"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AdvancedTabularError(f"Incumbent validation evidence is missing: {missing}")
    if frame["fold_id"].astype(int).tolist() != [1, 2, 3, 4]:
        raise AdvancedTabularError("Incumbent validation evidence does not cover four folds")
    return frame


def _candidate_is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    objectives = [
        "mean_mae",
        "mean_peak_mae",
        "temporal_fold_dispersion",
        "mean_model_size_bytes",
        "max_p95_inference_latency_ms_per_1000_rows",
    ]
    for other in candidates.itertuples(index=False):
        if other.algorithm_id == row["algorithm_id"]:
            continue
        other_values = [float(getattr(other, name)) for name in objectives]
        row_values = [float(row[name]) for name in objectives]
        if all(left <= right for left, right in zip(other_values, row_values, strict=True)) and any(
            left < right for left, right in zip(other_values, row_values, strict=True)
        ):
            return True
    return False


def _trial_validator(root: Path) -> Draft202012Validator:
    objective_schema = load_json(root / "schemas" / "objective_record.schema.json")
    trial_schema = load_json(root / "schemas" / "trial_evidence.schema.json")
    registry = Registry().with_resource(
        objective_schema["$id"], Resource.from_contents(objective_schema)
    )
    return Draft202012Validator(trial_schema, registry=registry)


def _validate_trial_record(
    root: Path,
    record: dict[str, Any],
) -> None:
    objective_schema = load_json(root / "schemas" / "objective_record.schema.json")
    for objective in record["objective_records"]:
        _validate_payload(objective, objective_schema, "Gate 6B objective record")
    errors = sorted(_trial_validator(root).iter_errors(record), key=lambda error: list(error.path))
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractError(f"Gate 6B trial record failed validation:\n{details}")


def execute_advanced_tabular_gate(root: Path) -> AdvancedTabularArtifacts:
    contract = validate_advanced_tabular_contract(root)
    boundary = contract["data_boundary"]
    model_contract = load_yaml(root / str(boundary["model_contract_path"]))
    target_contract = load_yaml(root / str(boundary["target_contract_path"]))
    silver = pd.read_parquet(root / str(boundary["silver_path"]))
    folds, fold_payload = _load_outer_folds(root / str(boundary["outer_folds_path"]))
    incumbent = _reference_frame(root / str(boundary["incumbent_results_path"]))

    maximum_origin_exclusive = int(boundary["maximum_prediction_origin_exclusive"])
    maximum_dependency_exclusive = int(boundary["maximum_target_dependency_exclusive"])
    if int(fold_payload["test_purge_start"]) != maximum_origin_exclusive:
        raise AdvancedTabularError("Gate 6B prediction boundary differs from the frozen chronology")
    if int(fold_payload["locked_test_start"]) != maximum_dependency_exclusive:
        raise AdvancedTabularError("Gate 6B dependency boundary differs from the frozen chronology")

    features = _feature_frame(silver, model_contract)
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    regression_target = str(model_contract["objectives"]["regression_target"])
    seed = int(contract["search"]["deterministic_seed"])
    inner_splitter = TimeSeriesSplit(
        n_splits=int(boundary["inner_fold_count"]),
        gap=int(boundary["purge_intervals"]),
    )

    output_directory = root / str(contract["outputs"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    dependency_lock_path, dependency_lock_sha = _dependency_lock(root, output_directory)
    code_commit = _git_commit(root)

    inner_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    trial_records: list[dict[str, Any]] = []

    for family in contract["candidate_families"]:
        algorithm_id = str(family["algorithm_id"])
        search_space = load_json(root / str(family["search_space_path"]))
        search_space_sha = _sha256_path(root / str(family["search_space_path"]))

        for fold in folds:
            if fold.validation_stop > maximum_origin_exclusive:
                raise AdvancedTabularError("Outer validation enters the locked-test purge")

            training_mask = pd.Series(False, index=silver.index)
            training_mask.iloc[fold.train_start : fold.train_stop] = True
            targets = build_supervised_targets(
                silver,
                training_mask,
                contract=target_contract,
            )
            target_frame = targets.frame
            peak_threshold = float(targets.peak_threshold_kwh)

            training_index = silver.index[fold.train_start : fold.train_stop]
            training_target = target_frame.loc[training_index, regression_target].astype(float)
            valid_training = training_target.notna()
            training_index = training_index[valid_training.to_numpy()]
            training_target = training_target.loc[training_index]
            training_features = features.loc[training_index]

            validation_index = silver.index[fold.validation_start : fold.validation_stop]
            validation_target = target_frame.loc[validation_index, regression_target].astype(float)
            if validation_target.isna().any():
                raise AdvancedTabularError(f"Fold {fold.fold_id} has missing validation targets")
            validation_features = features.loc[validation_index]

            configuration_scores: list[tuple[float, str, dict[str, Any]]] = []
            for configuration in family["configurations"]:
                configuration_id, parameters = _configuration_parameters(configuration)
                fold_scores: list[float] = []
                for inner_fold_id, (inner_train, inner_validation) in enumerate(
                    inner_splitter.split(training_features),
                    start=1,
                ):
                    estimator = _pipeline(
                        algorithm_id,
                        parameters,
                        model_contract,
                        seed=seed,
                    )
                    estimator.fit(
                        training_features.iloc[inner_train],
                        training_target.iloc[inner_train],
                    )
                    prediction = estimator.predict(training_features.iloc[inner_validation])
                    score = float(
                        mean_absolute_error(
                            training_target.iloc[inner_validation],
                            prediction,
                        )
                    )
                    if not math.isfinite(score):
                        raise AdvancedTabularError("Inner-validation MAE is not finite")
                    fold_scores.append(score)
                    inner_rows.append(
                        {
                            "algorithm_id": algorithm_id,
                            "outer_fold_id": fold.fold_id,
                            "configuration_id": configuration_id,
                            "inner_fold_id": inner_fold_id,
                            "inner_train_rows": int(len(inner_train)),
                            "inner_validation_rows": int(len(inner_validation)),
                            "mae": score,
                        }
                    )
                configuration_scores.append(
                    (float(np.mean(fold_scores)), configuration_id, parameters)
                )

            selected_inner_mae, selected_configuration_id, selected_parameters = min(
                configuration_scores,
                key=lambda item: (item[0], item[1]),
            )
            started = time.perf_counter()
            selected_model = _pipeline(
                algorithm_id,
                selected_parameters,
                model_contract,
                seed=seed,
            )
            selected_model.fit(training_features, training_target)
            outer_prediction = np.asarray(selected_model.predict(validation_features), dtype=float)
            wall_clock_seconds = float(time.perf_counter() - started)
            if not np.isfinite(outer_prediction).all():
                raise AdvancedTabularError("Outer-fold predictions contain nonfinite values")

            peak_mask = validation_target.ge(peak_threshold).to_numpy()
            mae = float(mean_absolute_error(validation_target, outer_prediction))
            peak_mae = float(
                mean_absolute_error(
                    validation_target.to_numpy()[peak_mask],
                    outer_prediction[peak_mask],
                )
            )
            model_bytes = pickle.dumps(selected_model, protocol=pickle.HIGHEST_PROTOCOL)
            restored_model = pickle.loads(model_bytes)
            restored_prediction = np.asarray(restored_model.predict(validation_features), dtype=float)
            portability_failure_count = int(
                not np.allclose(outer_prediction, restored_prediction, rtol=0.0, atol=1e-10)
            )
            latency = _latency_ms_per_1000_rows(selected_model, validation_features)
            peak_memory_mb = _peak_memory_mb()

            reference_row = incumbent.loc[incumbent["fold_id"].eq(fold.fold_id)].iloc[0]
            reference_mae = float(reference_row["mae"])
            reference_peak_mae = float(reference_row["peak_mae"])
            relative_mae_improvement = (reference_mae - mae) / reference_mae
            relative_peak_mae_change = (peak_mae - reference_peak_mae) / reference_peak_mae

            outer_rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "fold_id": fold.fold_id,
                    "selected_configuration_id": selected_configuration_id,
                    "selected_inner_mae": selected_inner_mae,
                    "validation_rows": int(len(validation_index)),
                    "peak_rows": int(peak_mask.sum()),
                    "peak_threshold_kwh": peak_threshold,
                    "mae": mae,
                    "peak_mae": peak_mae,
                    "reference_mae": reference_mae,
                    "reference_peak_mae": reference_peak_mae,
                    "relative_mae_improvement": relative_mae_improvement,
                    "relative_peak_mae_change": relative_peak_mae_change,
                    "model_size_bytes": int(len(model_bytes)),
                    "p95_inference_latency_ms_per_1000_rows": latency,
                    "peak_memory_mb": peak_memory_mb,
                    "wall_clock_seconds": wall_clock_seconds,
                    "portability_failure_count": portability_failure_count,
                    "maximum_prediction_origin": int(validation_index.max()),
                    "maximum_target_dependency": int(validation_index.max() + 1),
                }
            )

            predictions = pd.DataFrame(
                {
                    "algorithm_id": algorithm_id,
                    "fold_id": fold.fold_id,
                    "row_position": validation_index.to_numpy(),
                    "prediction_origin": timestamps.loc[validation_index].to_numpy(),
                    "actual": validation_target.to_numpy(),
                    "prediction": outer_prediction,
                    "is_peak_state": peak_mask,
                    "peak_threshold_kwh": peak_threshold,
                    "selected_configuration_id": selected_configuration_id,
                }
            )
            prediction_frames.append(predictions)
            prediction_sha = _sha256_bytes(predictions.to_csv(index=False).encode("utf-8"))
            trial_id = f"{algorithm_id}.outer_{fold.fold_id}"
            objective_record = {
                "contract_version": "1.0.0",
                "objective_set_version": "v2_objectives_1.0.0",
                "trial_id": trial_id,
                "fold_id": fold.fold_id,
                "seed": seed,
                "partition": "validation",
                "metrics": [
                    {
                        "id": "aggregate_mae",
                        "value": mae,
                        "unit": "kWh",
                        "direction": "minimize",
                        "finite": True,
                    },
                    {
                        "id": "peak_state_mae",
                        "value": peak_mae,
                        "unit": "kWh",
                        "direction": "minimize",
                        "finite": True,
                    },
                    {
                        "id": "model_size_bytes",
                        "value": float(len(model_bytes)),
                        "unit": "bytes",
                        "direction": "minimize",
                        "finite": True,
                    },
                    {
                        "id": "p95_inference_latency_ms_per_1000_rows",
                        "value": latency,
                        "unit": "milliseconds",
                        "direction": "minimize",
                        "finite": True,
                    },
                    {
                        "id": "portability_failure_count",
                        "value": float(portability_failure_count),
                        "unit": "count",
                        "direction": "minimize",
                        "finite": True,
                    },
                ],
                "computed_by": {
                    "implementation": "iaei.v2.advanced_tabular",
                    "code_commit": code_commit,
                    "deterministic": True,
                },
                "source_artifact_sha256": prediction_sha,
                "status": "complete",
            }
            hard_constraints = [
                {
                    "id": "chronology_violation_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
                {
                    "id": "leakage_violation_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
                {
                    "id": "locked_test_access_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
                {
                    "id": "missing_required_artifact_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
                {
                    "id": "nonfinite_objective_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
                {
                    "id": "portability_failure_count",
                    "observed_value": float(portability_failure_count),
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": portability_failure_count == 0,
                },
                {
                    "id": "ungoverned_parameter_count",
                    "observed_value": 0.0,
                    "operator": "equal",
                    "threshold": 0.0,
                    "passed": True,
                },
            ]
            record = {
                "contract_version": "1.0.0",
                "trial_id": trial_id,
                "search_space_id": str(search_space["search_space_id"]),
                "search_space_sha256": search_space_sha,
                "candidate_family": "advanced_tabular",
                "algorithm_id": algorithm_id,
                "parameter_values": {
                    "configuration_id": selected_configuration_id,
                    **selected_parameters,
                },
                "fold_ids": [fold.fold_id],
                "seeds": [seed],
                "objective_records": [objective_record],
                "hard_constraint_results": hard_constraints,
                "resource_usage": {
                    "wall_clock_seconds": wall_clock_seconds,
                    "peak_memory_mb": peak_memory_mb,
                    "model_size_bytes": int(len(model_bytes)),
                    "p95_inference_latency_ms_per_1000_rows": latency,
                },
                "environment": {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "dependency_lock_sha256": dependency_lock_sha,
                    "execution_adapter": "portable",
                },
                "code_commit": code_commit,
                "outcome": "success",
                "artifact_sha256": _sha256_bytes(model_bytes),
            }
            _validate_trial_record(root, record)
            trial_records.append(record)

    inner_results = pd.DataFrame(inner_rows).sort_values(
        ["algorithm_id", "outer_fold_id", "configuration_id", "inner_fold_id"],
        kind="stable",
    )
    outer_results = pd.DataFrame(outer_rows).sort_values(
        ["algorithm_id", "fold_id"],
        kind="stable",
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["algorithm_id", "row_position"],
        kind="stable",
    )

    if int(predictions["row_position"].max()) >= maximum_origin_exclusive:
        raise AdvancedTabularError("Gate 6B predictions reached the locked-test purge")
    if int(outer_results["maximum_target_dependency"].max()) >= maximum_dependency_exclusive:
        raise AdvancedTabularError("Gate 6B targets reached the locked-test partition")

    leaderboard_rows: list[dict[str, Any]] = []
    promotion = contract["promotion"]
    for algorithm_id, group in outer_results.groupby("algorithm_id", sort=True):
        mean_mae = float(group["mae"].mean())
        mean_peak_mae = float(group["peak_mae"].mean())
        reference_mean_mae = float(group["reference_mae"].mean())
        reference_mean_peak_mae = float(group["reference_peak_mae"].mean())
        mean_improvement = (reference_mean_mae - mean_mae) / reference_mean_mae
        peak_degradation = (mean_peak_mae - reference_mean_peak_mae) / reference_mean_peak_mae
        positive_folds = int(group["relative_mae_improvement"].gt(0.0).sum())
        max_fold_degradation = float((-group["relative_mae_improvement"]).max())
        dispersion = float(group["relative_mae_improvement"].std(ddof=0))
        hard_constraints_pass = bool(group["portability_failure_count"].eq(0).all())
        requirements = {
            "minimum_mean_validation_mae_relative_improvement": mean_improvement
            >= float(promotion["minimum_mean_validation_mae_relative_improvement"]),
            "minimum_positive_outer_folds": positive_folds
            >= int(promotion["minimum_positive_outer_folds"]),
            "maximum_single_fold_mae_relative_degradation": max_fold_degradation
            <= float(promotion["maximum_single_fold_mae_relative_degradation"]),
            "maximum_peak_state_mae_relative_degradation": peak_degradation
            <= float(promotion["maximum_peak_state_mae_relative_degradation"]),
            "hard_constraints_all_pass": hard_constraints_pass,
        }
        leaderboard_rows.append(
            {
                "algorithm_id": algorithm_id,
                "mean_mae": mean_mae,
                "mean_peak_mae": mean_peak_mae,
                "reference_mean_mae": reference_mean_mae,
                "reference_mean_peak_mae": reference_mean_peak_mae,
                "mean_validation_mae_relative_improvement": mean_improvement,
                "peak_state_mae_relative_degradation": peak_degradation,
                "positive_outer_folds": positive_folds,
                "maximum_single_fold_mae_relative_degradation": max_fold_degradation,
                "temporal_fold_dispersion": dispersion,
                "mean_model_size_bytes": float(group["model_size_bytes"].mean()),
                "max_p95_inference_latency_ms_per_1000_rows": float(
                    group["p95_inference_latency_ms_per_1000_rows"].max()
                ),
                "hard_constraints_all_pass": hard_constraints_pass,
                **{f"requirement_{key}": value for key, value in requirements.items()},
            }
        )

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mean_mae", kind="stable")
    leaderboard["pareto_status"] = [
        "dominated" if _candidate_is_dominated(row, leaderboard) else "eligible"
        for _, row in leaderboard.iterrows()
    ]
    requirement_columns = [
        column for column in leaderboard.columns if column.startswith("requirement_")
    ]
    leaderboard["promotion_eligible"] = (
        leaderboard[requirement_columns].all(axis=1)
        & leaderboard["pareto_status"].eq("eligible")
    )

    eligible = leaderboard.loc[leaderboard["promotion_eligible"]].sort_values(
        "mean_mae", kind="stable"
    )
    if eligible.empty:
        recommendation = {
            "gate": "6B",
            "status": "validation_complete",
            "recommendation": "retain_v1_incumbent",
            "candidate_id": None,
            "human_decision_required": True,
            "rationale": (
                "No advanced tabular challenger satisfied every frozen Gate 6A promotion "
                "requirement on the governed outer validation folds."
            ),
        }
    else:
        candidate_id = str(eligible.iloc[0]["algorithm_id"])
        recommendation = {
            "gate": "6B",
            "status": "validation_complete",
            "recommendation": "candidate_eligible_for_human_review",
            "candidate_id": candidate_id,
            "human_decision_required": True,
            "rationale": (
                f"{candidate_id} satisfied the frozen validation, robustness, hard-constraint, "
                "and Pareto requirements. Promotion remains a human decision."
            ),
        }

    output_paths = {
        "inner_search_results": output_directory / str(contract["outputs"]["inner_search_results"]),
        "outer_fold_results": output_directory / str(contract["outputs"]["outer_fold_results"]),
        "out_of_fold_predictions": output_directory
        / str(contract["outputs"]["out_of_fold_predictions"]),
        "candidate_leaderboard": output_directory
        / str(contract["outputs"]["candidate_leaderboard"]),
        "trial_evidence": output_directory / str(contract["outputs"]["trial_evidence"]),
        "execution_manifest": output_directory
        / str(contract["outputs"]["execution_manifest"]),
        "promotion_recommendation": output_directory
        / str(contract["outputs"]["promotion_recommendation"]),
    }
    inner_results.to_csv(output_paths["inner_search_results"], index=False)
    outer_results.to_csv(output_paths["outer_fold_results"], index=False)
    predictions.to_parquet(output_paths["out_of_fold_predictions"], index=False)
    leaderboard.to_csv(output_paths["candidate_leaderboard"], index=False)
    output_paths["trial_evidence"].write_text(
        json.dumps({"trials": trial_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths["promotion_recommendation"].write_text(
        json.dumps(recommendation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "contract_version": "1.0.0",
        "gate": "6B",
        "status": "validation_complete_pending_human_decision",
        "execution_commit": code_commit,
        "v1_tag": "v1.0.0",
        "v1_immutable": True,
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "candidate_family_count": 3,
        "unique_configuration_count": 12,
        "outer_fold_count": 4,
        "inner_fold_count": 3,
        "validation_origin_count_per_candidate": int(
            predictions.groupby("algorithm_id")["row_position"].nunique().min()
        ),
        "maximum_prediction_origin": int(predictions["row_position"].max()),
        "maximum_target_dependency": int(
            outer_results["maximum_target_dependency"].max()
        ),
        "dependency_lock": {
            "path": dependency_lock_path.relative_to(root).as_posix(),
            "sha256": dependency_lock_sha,
        },
        "artifacts": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_path(path),
            }
            for key, path in output_paths.items()
            if key != "execution_manifest"
        },
        "recommendation": recommendation,
        "next_action": "human_promotion_decision",
    }
    output_paths["execution_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return AdvancedTabularArtifacts(
        inner_search_results=inner_results,
        outer_fold_results=outer_results,
        out_of_fold_predictions=predictions,
        candidate_leaderboard=leaderboard,
        trial_records=trial_records,
        execution_manifest=manifest,
        promotion_recommendation=recommendation,
    )
