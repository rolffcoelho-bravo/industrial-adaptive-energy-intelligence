from __future__ import annotations

import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from iaei.modeling.candidates import (
    build_feature_preprocessor,
    build_inner_time_series_split,
)
from iaei.modeling.hist_gradient_boosting import (
    _feature_frame,
    build_hist_gradient_boosting_estimator,
)
from iaei.targets import build_supervised_targets
from iaei.v2.uncertainty_calibration import evaluate_intervals

try:
    import resource
except ImportError:  # pragma: no cover - Windows runtime
    resource = None

ROOT = Path(__file__).resolve().parents[3]


class Gate6E2ExecutionError(RuntimeError):
    """Raised when Gate 6E2 execution violates the frozen contract."""


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6E2ExecutionError(f"Expected an object in {path}")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise Gate6E2ExecutionError("Could not resolve execution commit")
    return value


def _pip_freeze() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.replace("\r\n", "\n")


def _directory_size(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _peak_rss_mb() -> float:
    if resource is not None:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise Gate6E2ExecutionError("Could not read Windows process memory")
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)
    raise Gate6E2ExecutionError("Unsupported process-memory platform")


def _resolve_silver_path(contract: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    contract_path = ROOT / str(contract["evidence_boundary"]["silver_path"])
    canonical_path = ROOT / "data" / "processed" / "steel_energy_silver.parquet"
    if not canonical_path.exists():
        raise Gate6E2ExecutionError("Canonical Silver analytical table is absent")
    canonical_hash = _sha256_path(canonical_path)
    if contract_path != canonical_path:
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        if contract_path.exists():
            contract_path.unlink()
        shutil.copyfile(canonical_path, contract_path)
    if not contract_path.exists():
        raise Gate6E2ExecutionError("Contract Silver path could not be materialized")
    contract_hash = _sha256_path(contract_path)
    if contract_hash != canonical_hash:
        raise Gate6E2ExecutionError("Silver path alias is not byte-identical")
    return contract_path, {
        "contract_path": str(contract_path.relative_to(ROOT).as_posix()),
        "canonical_path": str(canonical_path.relative_to(ROOT).as_posix()),
        "sha256": canonical_hash,
        "byte_identical_alias": True,
    }


def _fixed_pipeline(
    model_contract: dict[str, Any],
    parameters: dict[str, Any],
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor(model_contract)),
            ("model", build_hist_gradient_boosting_estimator(model_contract)),
        ]
    )
    pipeline.set_params(
        model__learning_rate=float(parameters["learning_rate"]),
        model__max_iter=int(parameters["max_iter"]),
        model__max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        model__l2_regularization=float(parameters["l2_regularization"]),
    )
    return pipeline


def _build_calibration_residuals(
    silver: pd.DataFrame,
    folds: list[Any],
    model_contract: dict[str, Any],
    target_contract: dict[str, Any],
    point_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, float], pd.DataFrame, dict[str, Any]]:
    features = _feature_frame(silver, model_contract)
    target_name = str(model_contract["objectives"]["regression_target"])
    tail_fraction = float(contract["calibration_protocol"]["calibration_tail_fraction"])
    minimum_origins = int(
        contract["calibration_protocol"]["minimum_calibration_origins"]
    )
    selected_by_fold = point_manifest["selected_parameters_by_fold"]
    scores_by_fold: dict[int, np.ndarray] = {}
    iqr_by_fold: dict[int, float] = {}
    residual_frames: list[pd.DataFrame] = []
    lineage_folds: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold.fold_id)
        parameters = dict(selected_by_fold[str(fold_id)])
        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[fold.train_start : fold.train_stop] = True
        target_artifacts = build_supervised_targets(
            silver,
            training_mask,
            contract=target_contract,
        )
        training_index = silver.index[fold.train_start : fold.train_stop]
        training_target = target_artifacts.frame.loc[
            training_index,
            target_name,
        ].astype(float)
        valid = training_target.notna()
        x_train = features.loc[training_index].loc[valid]
        y_train = training_target.loc[valid]
        if len(x_train) != len(y_train) or len(x_train) < minimum_origins:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} has insufficient valid training rows"
            )
        iqr = float(y_train.quantile(0.75) - y_train.quantile(0.25))
        if not math.isfinite(iqr) or iqr <= 0.0:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} has an invalid training-target IQR"
            )
        iqr_by_fold[fold_id] = iqr

        split_records: list[dict[str, Any]] = []
        fold_residuals: list[pd.DataFrame] = []
        splitter = build_inner_time_series_split(model_contract)
        for inner_id, (train_positions, validation_positions) in enumerate(
            splitter.split(x_train),
            start=1,
        ):
            inner_train_rows = x_train.index.to_numpy()[train_positions]
            inner_validation_rows = x_train.index.to_numpy()[validation_positions]
            maximum_train_dependency = int(inner_train_rows.max()) + 1
            minimum_validation_origin = int(inner_validation_rows.min())
            if maximum_train_dependency >= minimum_validation_origin:
                raise Gate6E2ExecutionError(
                    f"Fold {fold_id} inner split {inner_id} violates chronology"
                )
            pipeline = _fixed_pipeline(model_contract, parameters)
            fit_started = time.perf_counter()
            pipeline.fit(
                x_train.iloc[train_positions],
                y_train.iloc[train_positions],
            )
            predictions = pipeline.predict(x_train.iloc[validation_positions])
            fit_seconds = time.perf_counter() - fit_started
            actual = y_train.iloc[validation_positions].to_numpy(dtype=float)
            residuals = np.abs(actual - np.asarray(predictions, dtype=float))
            fold_residuals.append(
                pd.DataFrame(
                    {
                        "fold_id": fold_id,
                        "inner_split_id": inner_id,
                        "row_position": inner_validation_rows.astype(int),
                        "actual": actual,
                        "prediction": np.asarray(predictions, dtype=float),
                        "absolute_residual": residuals,
                    }
                )
            )
            split_records.append(
                {
                    "inner_split_id": inner_id,
                    "training_origin_start": int(inner_train_rows.min()),
                    "training_origin_stop_exclusive": int(inner_train_rows.max()) + 1,
                    "maximum_training_target_dependency": maximum_train_dependency,
                    "validation_origin_start": minimum_validation_origin,
                    "validation_origin_stop_exclusive": int(inner_validation_rows.max())
                    + 1,
                    "training_origin_count": int(len(inner_train_rows)),
                    "validation_origin_count": int(len(inner_validation_rows)),
                    "fit_wall_clock_seconds": float(fit_seconds),
                }
            )

        residual_frame = pd.concat(fold_residuals, ignore_index=True).sort_values(
            "row_position",
            kind="stable",
        )
        if residual_frame["row_position"].duplicated().any():
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} contains duplicate inner OOF residual origins"
            )
        eligible_count = int(len(residual_frame))
        tail_count = max(
            int(math.ceil(eligible_count * tail_fraction)),
            minimum_origins,
        )
        if tail_count > eligible_count:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} cannot satisfy the calibration-tail minimum"
            )
        calibration = residual_frame.tail(tail_count).copy()
        calibration["selected_for_initial_calibration"] = True
        calibration["calibration_rank"] = np.arange(1, tail_count + 1)
        residual_frame["selected_for_initial_calibration"] = False
        residual_frame.loc[
            calibration.index,
            "selected_for_initial_calibration",
        ] = True
        residual_frame["calibration_rank"] = pd.Series(
            pd.NA,
            index=residual_frame.index,
            dtype="Int64",
        )
        residual_frame.loc[
            calibration.index,
            "calibration_rank",
        ] = np.arange(1, tail_count + 1)
        selected_scores = calibration["absolute_residual"].to_numpy(dtype=float)
        if not np.isfinite(selected_scores).all() or (selected_scores < 0.0).any():
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} calibration residuals are invalid"
            )
        scores_by_fold[fold_id] = selected_scores
        residual_frames.append(residual_frame)
        lineage_folds.append(
            {
                "fold_id": fold_id,
                "outer_training_start": int(fold.train_start),
                "outer_training_stop_exclusive": int(fold.train_stop),
                "outer_validation_start": int(fold.validation_start),
                "outer_validation_stop_exclusive": int(fold.validation_stop),
                "selected_parameters": {
                    "learning_rate": float(parameters["learning_rate"]),
                    "max_iter": int(parameters["max_iter"]),
                    "max_leaf_nodes": int(parameters["max_leaf_nodes"]),
                    "l2_regularization": float(parameters["l2_regularization"]),
                },
                "inner_split_count": len(split_records),
                "inner_splits": split_records,
                "eligible_inner_oof_residual_count": eligible_count,
                "calibration_tail_fraction": tail_fraction,
                "initial_calibration_origin_count": tail_count,
                "initial_calibration_origin_start": int(
                    calibration["row_position"].min()
                ),
                "initial_calibration_origin_stop_exclusive": int(
                    calibration["row_position"].max()
                )
                + 1,
                "initial_calibration_score_sha256": _sha256_array(selected_scores),
                "outer_training_target_iqr": iqr,
            }
        )

    residuals = pd.concat(residual_frames, ignore_index=True).sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    lineage = {
        "schema_version": "1.0.0",
        "gate": "6E",
        "subgate": "6E2",
        "status": "calibration_residuals_complete",
        "point_model_id": "v1_frozen_champion",
        "residual_score": "absolute_error",
        "inner_oof_residuals_required": True,
        "in_sample_residuals_used": False,
        "random_split_used": False,
        "sequential_update_rule": (
            "revealed_targets_strictly_before_current_target"
        ),
        "folds": lineage_folds,
    }
    return scores_by_fold, iqr_by_fold, residuals, lineage


def _evaluate_one_configuration(
    point_predictions: pd.DataFrame,
    scores_by_fold: dict[int, np.ndarray],
    configuration: Any,
    coverage_levels: list[float],
    lower_support_bound: float,
) -> tuple[pd.DataFrame, list[float], float, float]:
    frames: list[pd.DataFrame] = []
    normalized_latencies: list[float] = []
    start_rss = _peak_rss_mb()
    started = time.perf_counter()
    for fold_id, fold in point_predictions.groupby("fold_id", sort=True):
        fold_started = time.perf_counter()
        evaluated = evaluate_intervals(
            fold,
            {int(fold_id): scores_by_fold[int(fold_id)]},
            configuration,
            coverage_levels,
            lower_support_bound=lower_support_bound,
        )
        fold_elapsed = time.perf_counter() - fold_started
        normalized_latencies.append(
            float(fold_elapsed * 1000.0 * 1000.0 / len(evaluated))
        )
        frames.append(evaluated)
    wall_clock = time.perf_counter() - started
    peak_memory = max(_peak_rss_mb(), start_rss)
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    return predictions, normalized_latencies, wall_clock, peak_memory


def _point_parity_violations(
    interval_predictions: pd.DataFrame,
    point_predictions: pd.DataFrame,
    tolerance: float,
) -> int:
    left = interval_predictions.sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    right = point_predictions.sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    if len(left) != len(right):
        return abs(len(left) - len(right)) + min(len(left), len(right))
    if not np.array_equal(
        left["row_position"].to_numpy(dtype=int),
        right["row_position"].to_numpy(dtype=int),
    ):
        return len(left)
    difference = np.abs(
        left["point_prediction"].to_numpy(dtype=float)
        - right["prediction"].to_numpy(dtype=float)
    )
    return int(np.sum(difference > tolerance))


def _nonfinite_count(frame: pd.DataFrame) -> int:
    count = 0
    for column in frame.select_dtypes(include=[np.number]).columns:
        count += int((~np.isfinite(frame[column].to_numpy(dtype=float))).sum())
    return count
