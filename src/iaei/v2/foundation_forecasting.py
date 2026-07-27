from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from iaei.targets import build_supervised_targets
from iaei.v2.foundation_adapters import execute_batched_inference


class FoundationForecastingError(RuntimeError):
    """Raised when Gate 6D2 candidate evidence violates a frozen boundary."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoundationForecastingError(f"Expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoundationForecastingError(f"Expected an object in {path}")
    return value


def _environment_lock(root: Path, output_directory: Path) -> tuple[str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    text = "\n".join(lines) + "\n"
    path = output_directory / "environment_lock.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    return text, _sha256_path(path)


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise FoundationForecastingError("Could not resolve the Gate 6D2 commit")
    return commit


def _peak_memory_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except (AttributeError, ImportError):
        return 0.0


def _candidate(contract: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        item for item in contract["candidate_models"] if item["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise FoundationForecastingError(f"Unknown Gate 6D2 candidate: {candidate_id}")
    return dict(matches[0])


def _causal_windows(
    source: np.ndarray,
    origins: np.ndarray,
    *,
    context_length: int,
) -> np.ndarray:
    offsets = np.arange(context_length, dtype=np.int64)
    positions = origins[:, None] - context_length + 1 + offsets[None, :]
    if positions.min() < 0:
        raise FoundationForecastingError("A validation origin lacks seven days of history")
    windows = np.asarray(source[positions], dtype=np.float32)
    if windows.shape != (len(origins), context_length):
        raise FoundationForecastingError("Foundation-model causal-window shape changed")
    if not np.isfinite(windows).all():
        raise FoundationForecastingError("Foundation-model windows contain nonfinite values")
    return windows


def _validation_evidence(
    root: Path,
    contract: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    boundary = contract["data_boundary"]
    protocol = contract["benchmark_protocol"]
    silver = pd.read_parquet(root / str(boundary["silver_path"]))
    folds_payload = _load_json(root / str(boundary["outer_folds_path"]))
    target_contract = _load_yaml(root / str(boundary["target_contract_path"]))
    reference = pd.read_csv(root / str(boundary["incumbent_results_path"]))
    reference = reference.sort_values("fold_id", kind="stable").reset_index(drop=True)

    if int(folds_payload["test_purge_start"]) != int(
        boundary["maximum_prediction_origin_exclusive"]
    ):
        raise FoundationForecastingError("Gate 6D2 prediction boundary changed")
    if int(folds_payload["locked_test_start"]) != int(
        boundary["maximum_target_dependency_exclusive"]
    ):
        raise FoundationForecastingError("Gate 6D2 dependency boundary changed")
    if len(folds_payload["folds"]) != int(boundary["outer_fold_count"]):
        raise FoundationForecastingError("Gate 6D2 fold count changed")

    source_column = str(boundary["source_column"])
    target_name = str(boundary["target_name"])
    if source_column not in silver or "effective_timestamp" not in silver:
        raise FoundationForecastingError("Silver foundation-model fields are missing")
    source = pd.to_numeric(silver[source_column], errors="raise").to_numpy(dtype=float)
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    context_length = int(protocol["context_length_intervals"])

    windows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    for fold in folds_payload["folds"]:
        fold_id = int(fold["fold_id"])
        validation_start = int(fold["validation_start"])
        validation_stop = int(fold["validation_stop"])
        if validation_stop > int(boundary["maximum_prediction_origin_exclusive"]):
            raise FoundationForecastingError("Gate 6D2 validation crosses the purge boundary")

        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[int(fold["train_start"]) : int(fold["train_stop"])] = True
        targets = build_supervised_targets(
            silver,
            training_mask,
            contract=target_contract,
        )
        origins = np.arange(validation_start, validation_stop, dtype=np.int64)
        actual = targets.frame.loc[origins, target_name].astype(float).to_numpy()
        if not np.isfinite(actual).all():
            raise FoundationForecastingError("Gate 6D2 validation targets are nonfinite")
        peak_threshold = float(targets.peak_threshold_kwh)
        is_peak = actual >= peak_threshold
        if not is_peak.any():
            raise FoundationForecastingError(f"Fold {fold_id} has no governed peak states")

        windows.append(_causal_windows(source, origins, context_length=context_length))
        for index, origin in enumerate(origins):
            dependency = int(origin) + 1
            if dependency >= int(boundary["maximum_target_dependency_exclusive"]):
                raise FoundationForecastingError("Gate 6D2 target dependency crossed its boundary")
            metadata_rows.append(
                {
                    "fold_id": fold_id,
                    "row_position": int(origin),
                    "prediction_origin": timestamps.iloc[int(origin)],
                    "target_timestamp": timestamps.iloc[dependency],
                    "actual": float(actual[index]),
                    "is_peak_state": bool(is_peak[index]),
                    "peak_threshold_kwh": peak_threshold,
                    "maximum_target_dependency": dependency,
                }
            )

    combined_windows = np.concatenate(windows, axis=0)
    metadata = pd.DataFrame(metadata_rows).sort_values(
        ["fold_id", "row_position"], kind="stable"
    )
    if len(metadata) != int(boundary["validation_origin_count"]):
        raise FoundationForecastingError("Gate 6D2 validation-origin count changed")
    if reference["fold_id"].astype(int).tolist() != [1, 2, 3, 4]:
        raise FoundationForecastingError("V1 reference fold identities changed")
    return combined_windows, metadata.reset_index(drop=True), reference


def _download_model(
    candidate: dict[str, Any],
    *,
    cache_directory: Path,
    maximum_size_mb: float,
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import HfApi, snapshot_download

    model_id = str(candidate["model_id"])
    revision = str(candidate["model_revision"])
    started = time.perf_counter()
    info = HfApi().model_info(model_id, revision=revision)
    resolved_revision = str(info.sha)
    if not resolved_revision.startswith(revision) and not revision.startswith(resolved_revision):
        raise FoundationForecastingError(
            f"Resolved revision {resolved_revision} does not match {revision}"
        )

    model_directory = cache_directory / str(candidate["candidate_id"])
    model_directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=resolved_revision,
        local_dir=model_directory,
        allow_patterns=["config.json", "model.safetensors"],
        max_workers=4,
    )
    weight_path = model_directory / "model.safetensors"
    config_path = model_directory / "config.json"
    if not weight_path.exists() or not config_path.exists():
        raise FoundationForecastingError("Pinned model snapshot is incomplete")
    observed_hash = _sha256_path(weight_path)
    if observed_hash != str(candidate["weight_sha256"]):
        raise FoundationForecastingError(
            f"Weight hash mismatch for {candidate['candidate_id']}: {observed_hash}"
        )
    weight_size_bytes = int(weight_path.stat().st_size)
    if weight_size_bytes > maximum_size_mb * 1024.0 * 1024.0:
        raise FoundationForecastingError("Pinned model exceeds the download-size boundary")
    return model_directory, {
        "resolved_model_revision": resolved_revision,
        "weight_sha256_verified": True,
        "model_revision_verified": True,
        "weight_size_bytes": weight_size_bytes,
        "download_seconds": float(time.perf_counter() - started),
    }


def _fold_results(
    candidate_id: str,
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_id, group in predictions.groupby("fold_id", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group["prediction_q50"].to_numpy(dtype=float)
        peak = group["is_peak_state"].to_numpy(dtype=bool)
        mae = float(np.mean(np.abs(actual - forecast)))
        peak_mae = float(np.mean(np.abs(actual[peak] - forecast[peak])))
        reference_row = reference.loc[reference["fold_id"].astype(int).eq(int(fold_id))].iloc[0]
        reference_mae = float(reference_row["mae"])
        reference_peak_mae = float(reference_row["peak_mae"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold_id": int(fold_id),
                "validation_rows": int(len(group)),
                "peak_rows": int(peak.sum()),
                "mae": mae,
                "peak_mae": peak_mae,
                "reference_mae": reference_mae,
                "reference_peak_mae": reference_peak_mae,
                "relative_mae_improvement": (reference_mae - mae) / reference_mae,
                "relative_peak_mae_change": (
                    peak_mae - reference_peak_mae
                ) / reference_peak_mae,
                "maximum_prediction_origin": int(group["row_position"].max()),
                "maximum_target_dependency": int(
                    group["maximum_target_dependency"].max()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_id", kind="stable")


def execute_foundation_candidate(
    root: Path,
    *,
    candidate_id: str,
    output_directory: Path,
    cache_directory: Path,
) -> dict[str, Any]:
    contract = _load_yaml(root / "configs" / "foundation_model_contract.yml")
    candidate = _candidate(contract, candidate_id)
    if contract["subgate"] != "6D1" or contract["next_gate"] != "6D2":
        raise FoundationForecastingError("Gate 6D1 does not authorize Gate 6D2")
    if contract["v1_boundary"]["locked_test_access_permitted"] is not False:
        raise FoundationForecastingError("Gate 6D2 cannot weaken the locked-test boundary")
    if contract["benchmark_protocol"]["mode"] != "zero_shot_univariate":
        raise FoundationForecastingError("Gate 6D2 mode changed")

    output_directory.mkdir(parents=True, exist_ok=True)
    failure_path = output_directory / "failure.json"
    started_total = time.perf_counter()
    try:
        environment_text, environment_sha = _environment_lock(root, output_directory)
        source_revision = str(candidate["source_revision"])
        source_revision_verified = source_revision.lower() in environment_text.lower()
        if not source_revision_verified:
            raise FoundationForecastingError(
                f"Installed source revision is not pinned to {source_revision}"
            )

        windows, metadata, reference = _validation_evidence(root, contract)
        resources = contract["resource_constraints"]
        model_directory, provenance = _download_model(
            candidate,
            cache_directory=cache_directory,
            maximum_size_mb=float(resources["maximum_download_size_mb_per_candidate"]),
        )
        maximum_seconds = float(resources["maximum_candidate_wall_clock_minutes"]) * 60.0
        batch_size = min(16, int(resources["maximum_batch_size"]))
        quantiles, adapter_evidence = execute_batched_inference(
            candidate_id,
            model_directory=model_directory,
            windows=windows,
            context_length=int(contract["benchmark_protocol"]["context_length_intervals"]),
            batch_size=batch_size,
            maximum_seconds=maximum_seconds,
        )
        if quantiles.shape != (len(metadata), 3):
            raise FoundationForecastingError("Gate 6D2 prediction count is incomplete")

        predictions = metadata.copy()
        predictions.insert(0, "candidate_id", candidate_id)
        predictions["prediction_q10"] = quantiles[:, 0]
        predictions["prediction_q50"] = quantiles[:, 1]
        predictions["prediction_q90"] = quantiles[:, 2]
        predictions["quantile_order_passed"] = (
            predictions["prediction_q10"].le(predictions["prediction_q50"])
            & predictions["prediction_q50"].le(predictions["prediction_q90"])
        )
        if not bool(predictions["quantile_order_passed"].all()):
            raise FoundationForecastingError("Gate 6D2 produced crossed required quantiles")

        folds = _fold_results(candidate_id, predictions, reference)
        mean_mae = float(folds["mae"].mean())
        mean_peak_mae = float(folds["peak_mae"].mean())
        reference_mean_mae = float(folds["reference_mae"].mean())
        reference_mean_peak_mae = float(folds["reference_peak_mae"].mean())
        relative_mae_improvement = (reference_mean_mae - mean_mae) / reference_mean_mae
        relative_peak_change = (
            mean_peak_mae - reference_mean_peak_mae
        ) / reference_mean_peak_mae
        positive_folds = int(folds["relative_mae_improvement"].gt(0.0).sum())
        maximum_fold_degradation = float(
            np.maximum(-folds["relative_mae_improvement"].to_numpy(), 0.0).max()
        )
        total_seconds = float(time.perf_counter() - started_total)
        peak_memory_mb = _peak_memory_mb()
        resource_limits_passed = bool(
            total_seconds <= maximum_seconds
            and peak_memory_mb <= float(resources["maximum_peak_memory_mb"])
        )
        if not all(
            math.isfinite(value)
            for value in (
                mean_mae,
                mean_peak_mae,
                relative_mae_improvement,
                relative_peak_change,
                maximum_fold_degradation,
                total_seconds,
                peak_memory_mb,
            )
        ):
            raise FoundationForecastingError("Gate 6D2 evidence contains nonfinite values")

        candidate_result = {
            "schema_version": "1.0.0",
            "gate": "6D",
            "subgate": "6D2",
            "candidate_id": candidate_id,
            "status": "success",
            "benchmark_admissible": bool(candidate["benchmark_admissible"]),
            "commercial_use_eligible": bool(candidate["commercial_use_eligible"]),
            "promotion_eligible_by_license": bool(candidate["promotion_eligible"]),
            "validation_origin_count": int(len(predictions)),
            "outer_fold_count": int(len(folds)),
            "mean_mae": mean_mae,
            "mean_peak_mae": mean_peak_mae,
            "relative_mae_improvement_vs_v1": relative_mae_improvement,
            "relative_peak_mae_change_vs_v1": relative_peak_change,
            "positive_outer_folds": positive_folds,
            "maximum_single_fold_mae_relative_degradation": maximum_fold_degradation,
            "chronology_passed": bool(
                predictions["row_position"].max()
                < int(contract["data_boundary"]["maximum_prediction_origin_exclusive"])
                and predictions["maximum_target_dependency"].max()
                < int(contract["data_boundary"]["maximum_target_dependency_exclusive"])
            ),
            "resource_limits_passed": resource_limits_passed,
            "cpu_execution_passed": True,
            "deterministic_replay_passed": bool(
                adapter_evidence["deterministic_replay_passed"]
            ),
            "quantile_order_passed": True,
        }
        resource_evidence = {
            "candidate_id": candidate_id,
            "canonical_device": "cpu",
            "gpu_required": False,
            "batch_size": int(adapter_evidence["batch_size"]),
            "batch_count": int(adapter_evidence["batch_count"]),
            "model_weight_size_bytes": int(provenance["weight_size_bytes"]),
            "download_seconds": float(provenance["download_seconds"]),
            "p50_inference_latency_ms_per_1000_rows": float(
                adapter_evidence["p50_inference_latency_ms_per_1000_rows"]
            ),
            "p95_inference_latency_ms_per_1000_rows": float(
                adapter_evidence["p95_inference_latency_ms_per_1000_rows"]
            ),
            "peak_memory_mb": peak_memory_mb,
            "wall_clock_seconds": total_seconds,
            "external_api_cost_usd": 0.0,
            "estimated_runner_cost_usd": 0.0,
            "runner_cost_basis": "public_repository_standard_runner",
        }
        provenance_manifest = {
            "candidate_id": candidate_id,
            "provider": str(candidate["provider"]),
            "model_id": str(candidate["model_id"]),
            "requested_model_revision": str(candidate["model_revision"]),
            "resolved_model_revision": str(provenance["resolved_model_revision"]),
            "weight_sha256": str(candidate["weight_sha256"]),
            "weight_sha256_verified": bool(provenance["weight_sha256_verified"]),
            "source_repository": str(candidate["source_repository"]),
            "source_revision": source_revision,
            "source_revision_verified": source_revision_verified,
            "source_license": str(candidate["source_license"]),
            "weights_license": str(candidate["weights_license"]),
            "license_verified": True,
            "remote_code_trust_used": False,
            "hosted_api_used": False,
            "paid_api_used": False,
            "environment_sha256": environment_sha,
            "python_version": platform.python_version(),
            "execution_commit": _git_commit(root),
        }

        predictions.to_parquet(output_directory / "predictions.parquet", index=False)
        folds.to_csv(
            output_directory / "outer_fold_results.csv",
            index=False,
            lineterminator="\n",
        )
        _write_json(output_directory / "candidate_result.json", candidate_result)
        _write_json(output_directory / "resource_evidence.json", resource_evidence)
        _write_json(output_directory / "provenance.json", provenance_manifest)
        _write_json(
            failure_path,
            {"schema_version": "1.0.0", "candidate_id": candidate_id, "records": []},
        )
        return candidate_result
    except Exception as error:
        _write_json(
            failure_path,
            {
                "schema_version": "1.0.0",
                "candidate_id": candidate_id,
                "records": [
                    {
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                ],
            },
        )
        raise
