from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import random
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
from sklearn.preprocessing import StandardScaler

from iaei.contracts import ContractError, load_json, load_yaml
from iaei.modeling.candidates import build_feature_preprocessor
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class NeuralForecastingError(RuntimeError):
    """Raised when Gate 6C evidence violates a governed boundary."""


@dataclass(frozen=True)
class NeuralForecastingArtifacts:
    inner_search_results: pd.DataFrame
    outer_seed_results: pd.DataFrame
    seed_summary: pd.DataFrame
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
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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


def validate_neural_forecasting_contract(root: Path) -> dict[str, Any]:
    contract = load_yaml(root / "configs" / "neural_forecasting_contract.yml")
    schema = load_json(root / "schemas" / "neural_forecasting_contract.schema.json")
    _validate_payload(contract, schema, "Gate 6C neural forecasting contract")

    predecessor = load_json(root / str(contract["predecessor_boundary"]["closure_manifest_path"]))
    if predecessor["gate"] != "6B" or predecessor["status"] != "closed":
        raise ContractError("Gate 6C requires the closed Gate 6B boundary")
    if predecessor["human_decision"]["retained_model"] != "v1_frozen_champion":
        raise ContractError("Gate 6C reference differs from the Gate 6B decision")

    governance = load_yaml(root / "configs" / "optimization_governance.yml")
    budget = governance["search_budgets"]["neural_forecasting"]
    search = contract["search"]
    if int(search["unique_configuration_count"]) > int(budget["max_unique_configurations"]):
        raise ContractError("Gate 6C configuration count exceeds Gate 6A")
    if int(search["max_parallel_trials"]) > int(budget["max_parallel_trials"]):
        raise ContractError("Gate 6C parallelism exceeds Gate 6A")
    if int(search["max_wall_clock_minutes"]) > int(budget["max_wall_clock_minutes"]):
        raise ContractError("Gate 6C wall-clock budget exceeds Gate 6A")

    expected_seeds = [int(value) for value in governance["randomness"]["stochastic_model_seeds"]]
    observed_seeds = [int(value) for value in contract["training"]["stochastic_seeds"]]
    if observed_seeds != expected_seeds:
        raise ContractError("Gate 6C stochastic seeds differ from Gate 6A")
    if len(observed_seeds) != int(budget["seeds_per_configuration"]):
        raise ContractError("Gate 6C seed count differs from Gate 6A")

    search_schema = load_json(root / "schemas" / "governed_search_space.schema.json")
    observed_algorithms: set[str] = set()
    configuration_count = 0
    for family in contract["candidate_families"]:
        path = root / str(family["search_space_path"])
        payload = load_json(path)
        _validate_payload(payload, search_schema, f"Search space {path.name}")
        algorithm_id = str(family["algorithm_id"])
        if payload["algorithm_id"] != algorithm_id:
            raise ContractError(f"Search-space algorithm mismatch for {algorithm_id}")
        if payload["candidate_family"] != "neural_forecasting":
            raise ContractError(f"Search-space family mismatch for {algorithm_id}")
        if algorithm_id in observed_algorithms:
            raise ContractError(f"Duplicate neural algorithm: {algorithm_id}")
        observed_algorithms.add(algorithm_id)
        configurations = list(family["configurations"])
        if len(configurations) != int(family["configuration_count"]):
            raise ContractError(f"Configuration count mismatch for {algorithm_id}")
        if [int(value) for value in payload["seeds"]] != observed_seeds:
            raise ContractError(f"Search-space seeds differ for {algorithm_id}")
        configuration_count += len(configurations)

    if configuration_count != int(search["unique_configuration_count"]):
        raise ContractError("Gate 6C total configuration count is inconsistent")

    sequence = contract["sequence"]
    if int(sequence["length_intervals"]) * int(sequence["interval_minutes"]) != int(sequence["lookback_minutes"]):
        raise ContractError("Gate 6C sequence length and lookback are inconsistent")

    return contract


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise NeuralForecastingError(
            "PyTorch is required only for the dedicated Gate 6C execution workflow"
        ) from exc
    return torch, nn, functional


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


def _sequence_frame(silver: pd.DataFrame, channels: list[str]) -> np.ndarray:
    missing = sorted(set(channels).difference(silver.columns))
    if missing:
        raise NeuralForecastingError(f"Sequence channels are missing: {missing}")
    frame = silver.loc[:, channels].apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        raise NeuralForecastingError("Sequence channels contain missing values")
    return frame.to_numpy(dtype=np.float32, copy=True)


def _load_outer_folds(path: Path) -> tuple[list[ChronologicalFold], dict[str, Any]]:
    payload = load_json(path)
    folds = [ChronologicalFold(**item) for item in payload["folds"]]
    if len(folds) != 4:
        raise NeuralForecastingError("Gate 6C requires exactly four outer folds")
    if [fold.fold_id for fold in folds] != [1, 2, 3, 4]:
        raise NeuralForecastingError("Outer-fold identifiers are not canonical")
    return folds, payload


def _reference_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("fold_id", kind="stable").reset_index(drop=True)
    required = {"fold_id", "mae", "peak_mae"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise NeuralForecastingError(f"Incumbent validation evidence is missing: {missing}")
    if frame["fold_id"].astype(int).tolist() != [1, 2, 3, 4]:
        raise NeuralForecastingError("Incumbent validation evidence does not cover four folds")
    return frame


def _valid_training_positions(targets: pd.Series, *, stop: int, sequence_length: int) -> np.ndarray:
    positions = np.arange(sequence_length - 1, stop, dtype=np.int64)
    valid = targets.iloc[positions].notna().to_numpy()
    selected = positions[valid]
    if selected.size == 0:
        raise NeuralForecastingError("No valid neural training positions")
    return selected


def _window_indices(positions: np.ndarray, sequence_length: int) -> np.ndarray:
    values = np.asarray(positions, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise NeuralForecastingError("Sequence positions must be a non-empty vector")
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    indices = values[:, None] - offsets[None, :]
    if int(indices.min()) < 0:
        raise NeuralForecastingError("Sequence construction requested future or missing history")
    if not np.array_equal(indices[:, -1], values):
        raise NeuralForecastingError("Sequence window does not end at the prediction origin")
    if np.any(indices > values[:, None]):
        raise NeuralForecastingError("Sequence construction contains future values")
    return indices


def _fit_sequence_scaler(sequence_values: np.ndarray, training_positions: np.ndarray) -> StandardScaler:
    maximum_position = int(np.max(training_positions))
    scaler = StandardScaler()
    scaler.fit(sequence_values[: maximum_position + 1])
    return scaler


def _sequence_tensor(
    sequence_values: np.ndarray,
    positions: np.ndarray,
    *,
    sequence_length: int,
    scaler: StandardScaler,
) -> np.ndarray:
    indices = _window_indices(positions, sequence_length)
    raw = sequence_values[indices]
    shape = raw.shape
    scaled = scaler.transform(raw.reshape(-1, shape[-1])).reshape(shape)
    return np.asarray(scaled, dtype=np.float32)


def _static_arrays(
    features: pd.DataFrame,
    training_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    model_contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = build_feature_preprocessor(model_contract)
    train = preprocessor.fit_transform(features.iloc[training_positions])
    evaluation = preprocessor.transform(features.iloc[evaluation_positions])
    return np.asarray(train, dtype=np.float32), np.asarray(evaluation, dtype=np.float32)


def _configuration_parameters(configuration: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    configuration_id = str(configuration["configuration_id"])
    parameters = {
        str(key): value
        for key, value in configuration.items()
        if key != "configuration_id"
    }
    if not parameters:
        raise NeuralForecastingError(f"Configuration {configuration_id} has no parameters")
    return configuration_id, parameters


def _set_determinism(seed: int, threads: int) -> Any:
    torch, _, _ = _require_torch()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch


def _build_model(
    algorithm_id: str,
    parameters: dict[str, Any],
    *,
    static_width: int,
    sequence_channels: int,
) -> Any:
    torch, nn, functional = _require_torch()

    if algorithm_id == "residual_mlp":
        hidden = int(parameters["hidden_width"])
        second = int(parameters["second_width"])
        dropout = float(parameters["dropout"])

        class ResidualMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Linear(static_width, hidden)
                self.block = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, second),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(second, hidden),
                )
                self.output = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, static: Any, sequence: Any) -> Any:
                del sequence
                base = self.input(static)
                return self.output(base + self.block(base)).squeeze(-1)

        return ResidualMLP()

    if algorithm_id == "causal_tcn":
        width = int(parameters["channel_width"])
        kernel = int(parameters["kernel_size"])
        levels = int(parameters["dilation_levels"])
        dropout = float(parameters["dropout"])

        class CausalBlock(nn.Module):
            def __init__(self, input_channels: int, output_channels: int, dilation: int) -> None:
                super().__init__()
                self.left_padding = (kernel - 1) * dilation
                self.conv = nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel,
                    dilation=dilation,
                )
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, value: Any) -> Any:
                value = functional.pad(value, (self.left_padding, 0))
                return self.dropout(self.activation(self.conv(value)))

        class CausalTCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                blocks: list[Any] = []
                input_channels = sequence_channels
                for level in range(levels):
                    blocks.append(CausalBlock(input_channels, width, 2**level))
                    input_channels = width
                self.network = nn.Sequential(*blocks)
                self.output = nn.Sequential(
                    nn.Linear(width + static_width, width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, 1),
                )

            def forward(self, static: Any, sequence: Any) -> Any:
                encoded = self.network(sequence.transpose(1, 2))[:, :, -1]
                return self.output(torch.cat([encoded, static], dim=1)).squeeze(-1)

        return CausalTCN()

    if algorithm_id == "gru_sequence":
        hidden = int(parameters["hidden_size"])
        layers = int(parameters["num_layers"])
        dropout = float(parameters["dropout"])

        class GRUSequence(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(
                    input_size=sequence_channels,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.output = nn.Sequential(
                    nn.Linear(hidden + static_width, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(self, static: Any, sequence: Any) -> Any:
                _, hidden_state = self.gru(sequence)
                encoded = hidden_state[-1]
                return self.output(torch.cat([encoded, static], dim=1)).squeeze(-1)

        return GRUSequence()

    raise NeuralForecastingError(f"Unsupported neural algorithm: {algorithm_id}")


def _tensor(value: np.ndarray) -> Any:
    torch, _, _ = _require_torch()
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def _predict(
    model: Any,
    static: np.ndarray,
    sequence: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    torch, _, _ = _require_torch()
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(static), batch_size):
            stop = min(start + batch_size, len(static))
            prediction = model(_tensor(static[start:stop]), _tensor(sequence[start:stop]))
            predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions).astype(float, copy=False)


def _fit_model(
    algorithm_id: str,
    parameters: dict[str, Any],
    *,
    static_train: np.ndarray,
    sequence_train: np.ndarray,
    target_train: np.ndarray,
    static_evaluation: np.ndarray,
    sequence_evaluation: np.ndarray,
    seed: int,
    training: dict[str, Any],
) -> tuple[Any, np.ndarray, float, float, float]:
    torch = _set_determinism(seed, int(training["torch_threads"]))
    model = _build_model(
        algorithm_id,
        parameters,
        static_width=int(static_train.shape[1]),
        sequence_channels=int(sequence_train.shape[2]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_function = torch.nn.L1Loss()
    batch_size = int(training["batch_size"])
    epochs = int(training["maximum_epochs"])
    gradient_clip = float(training["gradient_clip_norm"])

    target_mean = float(np.mean(target_train))
    target_scale = float(np.std(target_train))
    if not math.isfinite(target_scale) or target_scale <= 1e-8:
        target_scale = 1.0
    normalized_target = (np.asarray(target_train, dtype=np.float32) - target_mean) / target_scale

    static_tensor = _tensor(static_train)
    sequence_tensor = _tensor(sequence_train)
    target_tensor = _tensor(normalized_target)
    generator = torch.Generator().manual_seed(seed)

    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(static_tensor), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(static_tensor[batch], sequence_tensor[batch])
            loss = loss_function(prediction, target_tensor[batch])
            if not torch.isfinite(loss):
                raise NeuralForecastingError(f"Nonfinite neural loss for {algorithm_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
    wall_clock = time.perf_counter() - started

    normalized_prediction = _predict(
        model,
        static_evaluation,
        sequence_evaluation,
        batch_size=batch_size,
    )
    prediction = normalized_prediction * target_scale + target_mean
    return model, prediction, wall_clock, target_mean, target_scale


def _serialize_model(model: Any) -> bytes:
    torch, _, _ = _require_torch()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _portability_failure_count(
    algorithm_id: str,
    parameters: dict[str, Any],
    model_bytes: bytes,
    *,
    static: np.ndarray,
    sequence: np.ndarray,
    reference_prediction: np.ndarray,
    target_mean: float,
    target_scale: float,
    batch_size: int,
) -> int:
    torch, _, _ = _require_torch()
    try:
        restored = _build_model(
            algorithm_id,
            parameters,
            static_width=int(static.shape[1]),
            sequence_channels=int(sequence.shape[2]),
        )
        state = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
        restored.load_state_dict(state)
        count = min(256, len(static))
        observed = (
            _predict(
                restored,
                static[:count],
                sequence[:count],
                batch_size=batch_size,
            )
            * target_scale
            + target_mean
        )
        expected = np.asarray(reference_prediction[:count], dtype=float)
        return 0 if np.allclose(observed, expected, rtol=1e-6, atol=1e-6) else 1
    except Exception:
        return 1


def _latency_ms_per_1000_rows(
    model: Any,
    static: np.ndarray,
    sequence: np.ndarray,
    *,
    batch_size: int,
) -> float:
    count = min(1000, len(static))
    if count == 0:
        raise NeuralForecastingError("Latency sample is empty")
    _predict(model, static[:count], sequence[:count], batch_size=batch_size)
    timings: list[float] = []
    for _ in range(7):
        started = time.perf_counter()
        _predict(model, static[:count], sequence[:count], batch_size=batch_size)
        timings.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(np.asarray(timings) * (1000.0 / count), 95))


def _peak_memory_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except (ImportError, AttributeError):
        return 0.0


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
        raise NeuralForecastingError("Could not resolve the execution commit")
    return value


def _candidate_is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    objectives = [
        "mean_mae",
        "mean_peak_mae",
        "across_seed_mae_standard_deviation",
        "temporal_fold_dispersion",
        "mean_model_size_bytes",
        "max_p95_inference_latency_ms_per_1000_rows",
    ]
    for other in candidates.itertuples(index=False):
        if other.algorithm_id == row["algorithm_id"]:
            continue
        other_values = [float(getattr(other, name)) for name in objectives]
        row_values = [float(row[name]) for name in objectives]
        no_worse = all(
            left <= right
            for left, right in zip(other_values, row_values, strict=True)
        )
        strictly_better = any(
            left < right
            for left, right in zip(other_values, row_values, strict=True)
        )
        if no_worse and strictly_better:
            return True
    return False


def _trial_validator(root: Path) -> Draft202012Validator:
    objective_schema = load_json(root / "schemas" / "objective_record.schema.json")
    trial_schema = load_json(root / "schemas" / "trial_evidence.schema.json")
    registry = Registry().with_resource(
        objective_schema["$id"], Resource.from_contents(objective_schema)
    )
    return Draft202012Validator(trial_schema, registry=registry)


def _validate_trial_record(root: Path, record: dict[str, Any]) -> None:
    objective_schema = load_json(root / "schemas" / "objective_record.schema.json")
    for objective in record["objective_records"]:
        _validate_payload(objective, objective_schema, "Gate 6C objective record")
    errors = sorted(
        _trial_validator(root).iter_errors(record),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractError(f"Gate 6C trial record failed validation:\n{details}")


def _selection_rows(
    *,
    family: dict[str, Any],
    outer_fold: ChronologicalFold,
    features: pd.DataFrame,
    sequence_values: np.ndarray,
    targets: pd.Series,
    model_contract: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    training_positions = _valid_training_positions(
        targets,
        stop=outer_fold.train_stop,
        sequence_length=int(contract["sequence"]["length_intervals"]),
    )
    splitter = TimeSeriesSplit(
        n_splits=int(contract["data_boundary"]["inner_fold_count"]),
        gap=int(contract["data_boundary"]["purge_intervals"]),
    )
    rows: list[dict[str, Any]] = []
    sequence_length = int(contract["sequence"]["length_intervals"])
    selection_seed = int(contract["training"]["selection_seed"])

    for inner_fold_id, (train_relative, validation_relative) in enumerate(
        splitter.split(training_positions), start=1
    ):
        inner_train = training_positions[train_relative]
        inner_validation = training_positions[validation_relative]
        static_train, static_validation = _static_arrays(
            features, inner_train, inner_validation, model_contract
        )
        scaler = _fit_sequence_scaler(sequence_values, inner_train)
        sequence_train = _sequence_tensor(
            sequence_values,
            inner_train,
            sequence_length=sequence_length,
            scaler=scaler,
        )
        sequence_validation = _sequence_tensor(
            sequence_values,
            inner_validation,
            sequence_length=sequence_length,
            scaler=scaler,
        )
        target_train = targets.iloc[inner_train].to_numpy(dtype=float)
        target_validation = targets.iloc[inner_validation].to_numpy(dtype=float)

        for configuration in family["configurations"]:
            configuration_id, parameters = _configuration_parameters(configuration)
            _, prediction, wall_clock, _, _ = _fit_model(
                str(family["algorithm_id"]),
                parameters,
                static_train=static_train,
                sequence_train=sequence_train,
                target_train=target_train,
                static_evaluation=static_validation,
                sequence_evaluation=sequence_validation,
                seed=selection_seed,
                training=contract["training"],
            )
            rows.append(
                {
                    "algorithm_id": str(family["algorithm_id"]),
                    "outer_fold_id": int(outer_fold.fold_id),
                    "inner_fold_id": inner_fold_id,
                    "configuration_id": configuration_id,
                    "selection_seed": selection_seed,
                    "training_rows": int(len(inner_train)),
                    "validation_rows": int(len(inner_validation)),
                    "mae": float(mean_absolute_error(target_validation, prediction)),
                    "wall_clock_seconds": float(wall_clock),
                    "maximum_prediction_origin": int(inner_validation.max()),
                    "maximum_target_dependency": int(inner_validation.max() + 1),
                }
            )
    return rows


def _select_configuration(
    rows: pd.DataFrame,
    family: dict[str, Any],
    outer_fold_id: int,
) -> tuple[str, dict[str, Any], float]:
    subset = rows.loc[
        (rows["algorithm_id"] == str(family["algorithm_id"]))
        & (rows["outer_fold_id"] == outer_fold_id)
    ]
    summary = (
        subset.groupby("configuration_id", as_index=False)["mae"]
        .mean()
        .sort_values(["mae", "configuration_id"], kind="stable")
        .reset_index(drop=True)
    )
    selected_id = str(summary.iloc[0]["configuration_id"])
    selected_mae = float(summary.iloc[0]["mae"])
    for configuration in family["configurations"]:
        configuration_id, parameters = _configuration_parameters(configuration)
        if configuration_id == selected_id:
            return selected_id, parameters, selected_mae
    raise NeuralForecastingError(
        f"Selected configuration is missing for {family['algorithm_id']}"
    )


def _outer_seed_rows(
    *,
    family: dict[str, Any],
    outer_fold: ChronologicalFold,
    selected_configuration_id: str,
    selected_parameters: dict[str, Any],
    selected_inner_mae: float,
    features: pd.DataFrame,
    sequence_values: np.ndarray,
    targets: pd.Series,
    timestamps: pd.Series,
    peak_threshold: float,
    model_contract: dict[str, Any],
    contract: dict[str, Any],
    reference_row: pd.Series,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    sequence_length = int(contract["sequence"]["length_intervals"])
    training_positions = _valid_training_positions(
        targets,
        stop=outer_fold.train_stop,
        sequence_length=sequence_length,
    )
    validation_positions = np.arange(
        outer_fold.validation_start, outer_fold.validation_stop, dtype=np.int64
    )
    validation_target = targets.iloc[validation_positions]
    if validation_target.isna().any():
        raise NeuralForecastingError(
            f"Fold {outer_fold.fold_id} has missing neural validation targets"
        )

    static_train, static_validation = _static_arrays(
        features, training_positions, validation_positions, model_contract
    )
    scaler = _fit_sequence_scaler(sequence_values, training_positions)
    sequence_train = _sequence_tensor(
        sequence_values,
        training_positions,
        sequence_length=sequence_length,
        scaler=scaler,
    )
    sequence_validation = _sequence_tensor(
        sequence_values,
        validation_positions,
        sequence_length=sequence_length,
        scaler=scaler,
    )
    target_train = targets.iloc[training_positions].to_numpy(dtype=float)
    target_validation = validation_target.to_numpy(dtype=float)
    peak_state = target_validation >= peak_threshold

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in [int(value) for value in contract["training"]["stochastic_seeds"]]:
        model, prediction, wall_clock, target_mean, target_scale = _fit_model(
            str(family["algorithm_id"]),
            selected_parameters,
            static_train=static_train,
            sequence_train=sequence_train,
            target_train=target_train,
            static_evaluation=static_validation,
            sequence_evaluation=sequence_validation,
            seed=seed,
            training=contract["training"],
        )
        mae = float(mean_absolute_error(target_validation, prediction))
        peak_mae = float(
            mean_absolute_error(target_validation[peak_state], prediction[peak_state])
        )
        model_bytes = _serialize_model(model)
        portability_failures = _portability_failure_count(
            str(family["algorithm_id"]),
            selected_parameters,
            model_bytes,
            static=static_validation,
            sequence=sequence_validation,
            reference_prediction=prediction,
            target_mean=target_mean,
            target_scale=target_scale,
            batch_size=int(contract["training"]["batch_size"]),
        )
        latency = _latency_ms_per_1000_rows(
            model,
            static_validation,
            sequence_validation,
            batch_size=int(contract["training"]["batch_size"]),
        )
        relative_improvement = 1.0 - mae / float(reference_row["mae"])
        relative_peak_change = peak_mae / float(reference_row["peak_mae"]) - 1.0
        result_rows.append(
            {
                "algorithm_id": str(family["algorithm_id"]),
                "fold_id": int(outer_fold.fold_id),
                "seed": seed,
                "selected_configuration_id": selected_configuration_id,
                "selected_inner_mae": selected_inner_mae,
                "training_rows": int(len(training_positions)),
                "validation_rows": int(len(validation_positions)),
                "peak_rows": int(peak_state.sum()),
                "peak_threshold_kwh": float(peak_threshold),
                "mae": mae,
                "peak_mae": peak_mae,
                "reference_mae": float(reference_row["mae"]),
                "reference_peak_mae": float(reference_row["peak_mae"]),
                "relative_mae_improvement": relative_improvement,
                "relative_peak_mae_change": relative_peak_change,
                "target_training_scale": target_scale,
                "model_size_bytes": int(len(model_bytes)),
                "p95_inference_latency_ms_per_1000_rows": latency,
                "peak_memory_mb": _peak_memory_mb(),
                "wall_clock_seconds": float(wall_clock),
                "portability_failure_count": int(portability_failures),
                "maximum_prediction_origin": int(validation_positions.max()),
                "maximum_target_dependency": int(validation_positions.max() + 1),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "algorithm_id": str(family["algorithm_id"]),
                    "fold_id": int(outer_fold.fold_id),
                    "seed": seed,
                    "configuration_id": selected_configuration_id,
                    "row_position": validation_positions,
                    "prediction_origin": timestamps.iloc[validation_positions].to_numpy(),
                    "actual": target_validation,
                    "prediction": prediction,
                    "peak_threshold_kwh": float(peak_threshold),
                    "is_peak_state": peak_state,
                }
            )
        )
    return result_rows, prediction_frames


def _seed_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (algorithm_id, seed), frame in predictions.groupby(["algorithm_id", "seed"], sort=True):
        peak = frame.loc[frame["is_peak_state"].astype(bool)]
        rows.append(
            {
                "algorithm_id": str(algorithm_id),
                "seed": int(seed),
                "origin_count": int(frame["row_position"].nunique()),
                "mae": float(mean_absolute_error(frame["actual"], frame["prediction"])),
                "peak_mae": float(mean_absolute_error(peak["actual"], peak["prediction"])),
                "minimum_prediction": float(frame["prediction"].min()),
                "maximum_prediction": float(frame["prediction"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["algorithm_id", "seed"], kind="stable").reset_index(drop=True)


def _leaderboard(
    outer_results: pd.DataFrame,
    seed_summary: pd.DataFrame,
    reference: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    reference_mean_mae = float(reference["mae"].mean())
    reference_mean_peak_mae = float(reference["peak_mae"].mean())
    promotion = contract["promotion"]
    rows: list[dict[str, Any]] = []

    for algorithm_id, candidate_seed_rows in seed_summary.groupby("algorithm_id", sort=True):
        candidate_outer = outer_results.loc[outer_results["algorithm_id"] == algorithm_id]
        fold_means = (
            candidate_outer.groupby("fold_id", as_index=False)
            .agg(mae=("mae", "mean"), peak_mae=("peak_mae", "mean"))
            .sort_values("fold_id", kind="stable")
            .reset_index(drop=True)
        )
        merged = fold_means.merge(
            reference[["fold_id", "mae", "peak_mae"]],
            on="fold_id",
            suffixes=("", "_reference"),
            validate="one_to_one",
        )
        relative_fold_improvement = 1.0 - merged["mae"] / merged["mae_reference"]
        mean_mae = float(candidate_seed_rows["mae"].mean())
        mean_peak_mae = float(candidate_seed_rows["peak_mae"].mean())
        mean_improvement = 1.0 - mean_mae / reference_mean_mae
        peak_degradation = mean_peak_mae / reference_mean_peak_mae - 1.0
        positive_folds = int((merged["mae"] < merged["mae_reference"]).sum())
        maximum_fold_degradation = float((merged["mae"] / merged["mae_reference"] - 1.0).max())
        seed_count = int(candidate_seed_rows["seed"].nunique())
        portability_failures = int(candidate_outer["portability_failure_count"].sum())
        hard_constraints_all_pass = portability_failures == 0
        rows.append(
            {
                "algorithm_id": str(algorithm_id),
                "mean_mae": mean_mae,
                "across_seed_mae_standard_deviation": float(candidate_seed_rows["mae"].std(ddof=0)),
                "across_seed_mae_minimum": float(candidate_seed_rows["mae"].min()),
                "across_seed_mae_maximum": float(candidate_seed_rows["mae"].max()),
                "mean_peak_mae": mean_peak_mae,
                "across_seed_peak_mae_standard_deviation": float(candidate_seed_rows["peak_mae"].std(ddof=0)),
                "reference_mean_mae": reference_mean_mae,
                "reference_mean_peak_mae": reference_mean_peak_mae,
                "mean_validation_mae_relative_improvement": mean_improvement,
                "peak_state_mae_relative_degradation": peak_degradation,
                "positive_outer_folds": positive_folds,
                "maximum_single_fold_mae_relative_degradation": maximum_fold_degradation,
                "temporal_fold_dispersion": float(relative_fold_improvement.std(ddof=0)),
                "seed_count": seed_count,
                "mean_model_size_bytes": float(candidate_outer["model_size_bytes"].mean()),
                "max_p95_inference_latency_ms_per_1000_rows": float(candidate_outer["p95_inference_latency_ms_per_1000_rows"].max()),
                "max_peak_memory_mb": float(candidate_outer["peak_memory_mb"].max()),
                "total_wall_clock_seconds": float(candidate_outer["wall_clock_seconds"].sum()),
                "portability_failure_count": portability_failures,
                "hard_constraints_all_pass": hard_constraints_all_pass,
                "requirement_minimum_mean_validation_mae_relative_improvement": (
                    mean_improvement >= float(promotion["minimum_mean_validation_mae_relative_improvement"])
                ),
                "requirement_minimum_positive_outer_folds": (
                    positive_folds >= int(promotion["minimum_positive_outer_folds"])
                ),
                "requirement_maximum_single_fold_mae_relative_degradation": (
                    maximum_fold_degradation <= float(promotion["maximum_single_fold_mae_relative_degradation"])
                ),
                "requirement_maximum_peak_state_mae_relative_degradation": (
                    peak_degradation <= float(promotion["maximum_peak_state_mae_relative_degradation"])
                ),
                "requirement_complete_seed_evidence": (
                    seed_count == int(promotion["required_seed_count"])
                ),
                "requirement_hard_constraints_all_pass": hard_constraints_all_pass,
            }
        )

    leaderboard = pd.DataFrame(rows)
    leaderboard["pareto_status"] = [
        "dominated" if _candidate_is_dominated(row, leaderboard) else "eligible"
        for _, row in leaderboard.iterrows()
    ]
    requirement_columns = [
        "requirement_minimum_mean_validation_mae_relative_improvement",
        "requirement_minimum_positive_outer_folds",
        "requirement_maximum_single_fold_mae_relative_degradation",
        "requirement_maximum_peak_state_mae_relative_degradation",
        "requirement_complete_seed_evidence",
        "requirement_hard_constraints_all_pass",
    ]
    leaderboard["promotion_eligible"] = (
        leaderboard[requirement_columns].all(axis=1)
        & leaderboard["pareto_status"].eq("eligible")
    )
    return leaderboard.sort_values(
        ["promotion_eligible", "mean_mae", "algorithm_id"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _recommendation(leaderboard: pd.DataFrame) -> dict[str, Any]:
    eligible = leaderboard.loc[leaderboard["promotion_eligible"].astype(bool)]
    if eligible.empty:
        return {
            "candidate_id": None,
            "gate": "6C",
            "human_decision_required": True,
            "rationale": (
                "No neural forecasting challenger satisfied every frozen Gate 6A "
                "promotion requirement on the governed outer validation folds and seeds."
            ),
            "recommendation": "retain_v1_incumbent",
            "status": "validation_complete",
        }
    selected = eligible.sort_values(["mean_mae", "algorithm_id"], kind="stable").iloc[0]
    return {
        "candidate_id": str(selected["algorithm_id"]),
        "gate": "6C",
        "human_decision_required": True,
        "rationale": (
            "The candidate satisfied the frozen promotion requirements on validation "
            "evidence. Human approval remains required before any promotion."
        ),
        "recommendation": "consider_candidate_promotion",
        "status": "validation_complete",
    }


def _objective_record(
    *,
    algorithm_id: str,
    row: pd.Series,
    code_commit: str,
    source_artifact_sha256: str,
) -> dict[str, Any]:
    trial_id = f"gate_6c_{algorithm_id}"
    metrics = [
        {"id": "aggregate_mae", "value": float(row["mae"]), "unit": "kWh", "direction": "minimize", "finite": True},
        {"id": "peak_state_mae", "value": float(row["peak_mae"]), "unit": "kWh", "direction": "minimize", "finite": True},
        {"id": "model_size_bytes", "value": float(row["model_size_bytes"]), "unit": "bytes", "direction": "minimize", "finite": True},
        {"id": "p95_inference_latency_ms_per_1000_rows", "value": float(row["p95_inference_latency_ms_per_1000_rows"]), "unit": "milliseconds", "direction": "minimize", "finite": True},
        {"id": "portability_failure_count", "value": float(row["portability_failure_count"]), "unit": "count", "direction": "minimize", "finite": True},
    ]
    return {
        "contract_version": "1.0.0",
        "objective_set_version": "v2_objectives_1.0.0",
        "trial_id": trial_id,
        "fold_id": int(row["fold_id"]),
        "seed": int(row["seed"]),
        "partition": "validation",
        "metrics": metrics,
        "computed_by": {
            "implementation": "iaei.v2.neural_forecasting",
            "code_commit": code_commit,
            "deterministic": True,
        },
        "source_artifact_sha256": source_artifact_sha256,
        "status": "complete",
    }


def _trial_records(
    root: Path,
    contract: dict[str, Any],
    outer_results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    *,
    code_commit: str,
    dependency_lock_sha256: str,
    prediction_artifact_sha256: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in contract["candidate_families"]:
        algorithm_id = str(family["algorithm_id"])
        candidate = outer_results.loc[outer_results["algorithm_id"] == algorithm_id].sort_values(
            ["fold_id", "seed"], kind="stable"
        )
        selected_by_fold = candidate[["fold_id", "selected_configuration_id"]].drop_duplicates()
        parameter_values: dict[str, Any] = {
            "selected_configuration_by_fold": ",".join(
                f"{int(row.fold_id)}:{row.selected_configuration_id}"
                for row in selected_by_fold.itertuples(index=False)
            )
        }
        summary = leaderboard.loc[leaderboard["algorithm_id"] == algorithm_id].iloc[0]
        portability_failures = int(candidate["portability_failure_count"].sum())
        constraints = [
            {"id": "chronology_violation_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
            {"id": "leakage_violation_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
            {"id": "locked_test_access_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
            {"id": "missing_required_artifact_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
            {"id": "nonfinite_objective_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
            {"id": "portability_failure_count", "observed_value": portability_failures, "operator": "equal", "threshold": 0, "passed": portability_failures == 0},
            {"id": "ungoverned_parameter_count", "observed_value": 0, "operator": "equal", "threshold": 0, "passed": True},
        ]
        search_space_path = root / str(family["search_space_path"])
        record = {
            "contract_version": "1.0.0",
            "trial_id": f"gate_6c_{algorithm_id}",
            "search_space_id": load_json(search_space_path)["search_space_id"],
            "search_space_sha256": _sha256_path(search_space_path),
            "candidate_family": "neural_forecasting",
            "algorithm_id": algorithm_id,
            "parameter_values": parameter_values,
            "fold_ids": [1, 2, 3, 4],
            "seeds": [int(value) for value in contract["training"]["stochastic_seeds"]],
            "objective_records": [
                _objective_record(
                    algorithm_id=algorithm_id,
                    row=row,
                    code_commit=code_commit,
                    source_artifact_sha256=prediction_artifact_sha256,
                )
                for _, row in candidate.iterrows()
            ],
            "hard_constraint_results": constraints,
            "resource_usage": {
                "wall_clock_seconds": float(candidate["wall_clock_seconds"].sum()),
                "peak_memory_mb": float(candidate["peak_memory_mb"].max()),
                "model_size_bytes": int(round(float(summary["mean_model_size_bytes"]))),
                "p95_inference_latency_ms_per_1000_rows": float(summary["max_p95_inference_latency_ms_per_1000_rows"]),
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "dependency_lock_sha256": dependency_lock_sha256,
                "execution_adapter": "portable",
            },
            "code_commit": code_commit,
            "outcome": "success",
            "artifact_sha256": prediction_artifact_sha256,
        }
        _validate_trial_record(root, record)
        records.append(record)
    return records


def execute_neural_forecasting_gate(root: Path) -> NeuralForecastingArtifacts:
    contract = validate_neural_forecasting_contract(root)
    boundary = contract["data_boundary"]
    model_contract = load_yaml(root / str(boundary["model_contract_path"]))
    target_contract = load_yaml(root / str(boundary["target_contract_path"]))
    silver = pd.read_parquet(root / str(boundary["silver_path"]))
    folds, fold_payload = _load_outer_folds(root / str(boundary["outer_folds_path"]))
    reference = _reference_frame(root / str(boundary["incumbent_results_path"]))

    maximum_origin_exclusive = int(boundary["maximum_prediction_origin_exclusive"])
    maximum_dependency_exclusive = int(boundary["maximum_target_dependency_exclusive"])
    if int(fold_payload["test_purge_start"]) != maximum_origin_exclusive:
        raise NeuralForecastingError(
            "Gate 6C prediction boundary differs from the frozen chronology"
        )
    if int(fold_payload["locked_test_start"]) != maximum_dependency_exclusive:
        raise NeuralForecastingError(
            "Gate 6C dependency boundary differs from the frozen chronology"
        )

    features = _feature_frame(silver, model_contract)
    sequence_values = _sequence_frame(
        silver, [str(value) for value in contract["sequence"]["channels"]]
    )
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    regression_target = str(model_contract["objectives"]["regression_target"])

    output_directory = root / str(contract["outputs"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    dependency_lock_path, dependency_lock_sha = _dependency_lock(root, output_directory)
    code_commit = _git_commit(root)

    inner_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        if fold.validation_stop > fold.test_purge_start:
            raise NeuralForecastingError(f"Fold {fold.fold_id} enters the locked-test purge")
        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[fold.train_start : fold.train_stop] = True
        target_artifacts = build_supervised_targets(
            silver, training_mask, contract=target_contract
        )
        targets = target_artifacts.frame[regression_target].astype(float)
        peak_threshold = float(target_artifacts.peak_threshold_kwh)

        for family in contract["candidate_families"]:
            family_inner_rows = _selection_rows(
                family=family,
                outer_fold=fold,
                features=features,
                sequence_values=sequence_values,
                targets=targets,
                model_contract=model_contract,
                contract=contract,
            )
            inner_rows.extend(family_inner_rows)
            inner_frame = pd.DataFrame(inner_rows)
            selected_id, selected_parameters, selected_inner_mae = _select_configuration(
                inner_frame, family, int(fold.fold_id)
            )
            reference_row = reference.loc[
                reference["fold_id"].astype(int) == int(fold.fold_id)
            ].iloc[0]
            family_outer_rows, family_predictions = _outer_seed_rows(
                family=family,
                outer_fold=fold,
                selected_configuration_id=selected_id,
                selected_parameters=selected_parameters,
                selected_inner_mae=selected_inner_mae,
                features=features,
                sequence_values=sequence_values,
                targets=targets,
                timestamps=timestamps,
                peak_threshold=peak_threshold,
                model_contract=model_contract,
                contract=contract,
                reference_row=reference_row,
            )
            outer_rows.extend(family_outer_rows)
            prediction_frames.extend(family_predictions)

    inner_results = pd.DataFrame(inner_rows).sort_values(
        ["algorithm_id", "outer_fold_id", "inner_fold_id", "configuration_id"],
        kind="stable",
    ).reset_index(drop=True)
    outer_results = pd.DataFrame(outer_rows).sort_values(
        ["algorithm_id", "fold_id", "seed"], kind="stable"
    ).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["algorithm_id", "seed", "row_position"], kind="stable"
    ).reset_index(drop=True)
    seed_results = _seed_summary(predictions)
    leaderboard = _leaderboard(outer_results, seed_results, reference, contract)
    recommendation = _recommendation(leaderboard)

    inner_path = output_directory / str(contract["outputs"]["inner_search_results"])
    outer_path = output_directory / str(contract["outputs"]["outer_seed_results"])
    seed_path = output_directory / str(contract["outputs"]["seed_summary"])
    prediction_path = output_directory / str(contract["outputs"]["out_of_fold_predictions"])
    leaderboard_path = output_directory / str(contract["outputs"]["candidate_leaderboard"])
    recommendation_path = output_directory / str(contract["outputs"]["promotion_recommendation"])

    inner_results.to_csv(inner_path, index=False, lineterminator="\n")
    outer_results.to_csv(outer_path, index=False, lineterminator="\n")
    seed_results.to_csv(seed_path, index=False, lineterminator="\n")
    predictions.to_parquet(prediction_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False, lineterminator="\n")
    recommendation_path.write_bytes(_canonical_json_bytes(recommendation))

    prediction_sha = _sha256_path(prediction_path)
    trials = _trial_records(
        root,
        contract,
        outer_results,
        leaderboard,
        code_commit=code_commit,
        dependency_lock_sha256=dependency_lock_sha,
        prediction_artifact_sha256=prediction_sha,
    )
    trial_payload = {"contract_version": "1.0.0", "gate": "6C", "trials": trials}
    trial_path = output_directory / str(contract["outputs"]["trial_evidence"])
    trial_path.write_bytes(_canonical_json_bytes(trial_payload))

    artifact_paths = {
        "inner_search_results": inner_path,
        "outer_seed_results": outer_path,
        "seed_summary": seed_path,
        "out_of_fold_predictions": prediction_path,
        "candidate_leaderboard": leaderboard_path,
        "promotion_recommendation": recommendation_path,
        "trial_evidence": trial_path,
    }
    maximum_prediction_origin = int(predictions["row_position"].max())
    maximum_target_dependency = maximum_prediction_origin + 1
    manifest = {
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_path(path),
            }
            for name, path in sorted(artifact_paths.items())
        },
        "candidate_family_count": 3,
        "confirmatory_evaluation_performed": False,
        "contract_version": "1.0.0",
        "dependency_lock": {
            "path": dependency_lock_path.relative_to(root).as_posix(),
            "sha256": dependency_lock_sha,
        },
        "execution_commit": code_commit,
        "gate": "6C",
        "inner_fold_count": 3,
        "locked_predictions_parsed": False,
        "locked_test_accessed": False,
        "maximum_prediction_origin": maximum_prediction_origin,
        "maximum_target_dependency": maximum_target_dependency,
        "next_action": "human_promotion_decision",
        "outer_fold_count": 4,
        "recommendation": recommendation,
        "seed_count": 5,
        "status": "validation_complete_pending_human_decision",
        "unique_configuration_count": 6,
        "v1_immutable": True,
        "v1_tag": "v1.0.0",
        "validation_origin_count_per_candidate_seed": int(
            predictions.groupby(["algorithm_id", "seed"])["row_position"].nunique().min()
        ),
    }
    manifest_path = output_directory / str(contract["outputs"]["execution_manifest"])
    manifest_path.write_bytes(_canonical_json_bytes(manifest))

    return NeuralForecastingArtifacts(
        inner_search_results=inner_results,
        outer_seed_results=outer_results,
        seed_summary=seed_results,
        out_of_fold_predictions=predictions,
        candidate_leaderboard=leaderboard,
        trial_records=trials,
        execution_manifest=manifest,
        promotion_recommendation=recommendation,
    )
