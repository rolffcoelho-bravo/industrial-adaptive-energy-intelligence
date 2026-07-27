from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class FoundationEvidenceError(RuntimeError):
    """Raised when Gate 6D2 aggregate evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoundationEvidenceError(f"Expected an object in {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoundationEvidenceError(f"Expected a mapping in {path}")
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


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise FoundationEvidenceError("Could not resolve the Gate 6D2 commit")
    return value


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _recommendation(
    candidate_results: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    promotion = contract["promotion"]
    candidates: list[dict[str, Any]] = []
    for row in candidate_results.itertuples(index=False):
        failed: list[str] = []
        if row.status != "success":
            failed.append("execution_status")
        if row.relative_mae_improvement_vs_v1 < float(
            promotion["minimum_mean_validation_mae_relative_improvement"]
        ):
            failed.append("aggregate_mae_improvement")
        if row.relative_peak_mae_change_vs_v1 > float(
            promotion["maximum_peak_state_mae_relative_degradation"]
        ):
            failed.append("peak_state_mae")
        if row.positive_outer_folds < int(promotion["minimum_positive_outer_folds"]):
            failed.append("positive_outer_folds")
        if row.maximum_single_fold_mae_relative_degradation > float(
            promotion["maximum_single_fold_mae_relative_degradation"]
        ):
            failed.append("single_fold_degradation")
        if not bool(row.chronology_passed):
            failed.append("chronology")
        if not bool(row.resource_limits_passed):
            failed.append("resource_limits")
        if not bool(row.cpu_execution_passed):
            failed.append("cpu_execution")
        if not bool(row.deterministic_replay_passed):
            failed.append("deterministic_replay")
        if not bool(row.quantile_order_passed):
            failed.append("quantile_order")
        if not bool(row.commercial_use_eligible):
            failed.append("commercial_use_eligibility")
        if not bool(row.promotion_eligible_by_license):
            failed.append("license_promotion_boundary")
        candidates.append(
            {
                "candidate_id": row.candidate_id,
                "benchmark_status": row.status,
                "commercial_use_eligible": bool(row.commercial_use_eligible),
                "promotion_eligible": not failed,
                "failed_requirements": failed,
            }
        )
    eligible = [item["candidate_id"] for item in candidates if item["promotion_eligible"]]
    return {
        "schema_version": "1.0.0",
        "gate": "6D",
        "subgate": "6D2",
        "status": "validation_complete_pending_human_decision",
        "automatic_promotion_permitted": False,
        "human_decision_required": True,
        "eligible_candidates": eligible,
        "candidates": candidates,
        "recommended_next_gate": "6D3",
    }


def build_foundation_evidence(root: Path, staging_root: Path) -> dict[str, Any]:
    contract = _load_yaml(root / "configs" / "foundation_model_contract.yml")
    output_directory = root / str(contract["outputs"]["directory"])
    execution_manifest_path = output_directory / str(
        contract["outputs"]["execution_manifest"]
    )
    if execution_manifest_path.exists():
        raise FoundationEvidenceError("Gate 6D2 execution evidence is write-once")
    output_directory.mkdir(parents=True, exist_ok=True)

    expected_candidates = [
        str(candidate["candidate_id"]) for candidate in contract["candidate_models"]
    ]
    candidate_records: list[dict[str, Any]] = []
    resource_records: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    environment_directory = output_directory / "environment_locks"
    environment_directory.mkdir(parents=True, exist_ok=True)

    for candidate_id in expected_candidates:
        candidate_directory = staging_root / "candidates" / candidate_id
        required = (
            "candidate_result.json",
            "resource_evidence.json",
            "provenance.json",
            "failure.json",
            "environment_lock.txt",
        )
        missing = [name for name in required if not (candidate_directory / name).exists()]
        if missing:
            raise FoundationEvidenceError(
                f"Candidate {candidate_id} is missing staged evidence: {missing}"
            )

        result = _load_json(candidate_directory / "candidate_result.json")
        if result.get("candidate_id") != candidate_id:
            raise FoundationEvidenceError("Candidate-result identity mismatch")
        candidate_records.append(result)
        resource_records.append(_load_json(candidate_directory / "resource_evidence.json"))
        provenance_records.append(_load_json(candidate_directory / "provenance.json"))
        failure = _load_json(candidate_directory / "failure.json")
        for record in failure.get("records", []):
            failure_records.append({"candidate_id": candidate_id, **dict(record)})

        if result.get("status") == "success":
            fold_path = candidate_directory / "outer_fold_results.csv"
            predictions_path = candidate_directory / "predictions.parquet"
            if not fold_path.exists() or not predictions_path.exists():
                raise FoundationEvidenceError("Successful candidate lacks prediction evidence")
            fold_frames.append(pd.read_csv(fold_path))
            prediction_frames.append(pd.read_parquet(predictions_path))
        shutil.copyfile(
            candidate_directory / "environment_lock.txt",
            environment_directory / f"{candidate_id}.txt",
        )

    candidate_results = pd.DataFrame(candidate_records).sort_values(
        "mean_mae", kind="stable"
    )
    resources = pd.DataFrame(resource_records).sort_values(
        "candidate_id", kind="stable"
    )
    folds = pd.concat(fold_frames, ignore_index=True).sort_values(
        ["candidate_id", "fold_id"], kind="stable"
    )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["candidate_id", "fold_id", "row_position"], kind="stable"
    )

    numeric_columns = [
        "mean_mae",
        "mean_peak_mae",
        "relative_mae_improvement_vs_v1",
        "relative_peak_mae_change_vs_v1",
        "maximum_single_fold_mae_relative_degradation",
    ]
    if not _finite(
        [float(value) for column in numeric_columns for value in candidate_results[column]]
    ):
        raise FoundationEvidenceError("Gate 6D2 candidate evidence is nonfinite")
    if candidate_results["candidate_id"].tolist() != sorted(
        expected_candidates,
        key=lambda value: float(
            candidate_results.loc[
                candidate_results["candidate_id"].eq(value), "mean_mae"
            ].iloc[0]
        ),
    ):
        raise FoundationEvidenceError("Gate 6D2 leaderboard ordering is inconsistent")

    boundary = contract["data_boundary"]
    expected_prediction_rows = int(boundary["validation_origin_count"]) * len(
        expected_candidates
    )
    if len(predictions) != expected_prediction_rows:
        raise FoundationEvidenceError("Gate 6D2 out-of-fold predictions are incomplete")
    if predictions["row_position"].max() >= int(
        boundary["maximum_prediction_origin_exclusive"]
    ):
        raise FoundationEvidenceError("Gate 6D2 predictions cross the locked boundary")
    if predictions["maximum_target_dependency"].max() >= int(
        boundary["maximum_target_dependency_exclusive"]
    ):
        raise FoundationEvidenceError("Gate 6D2 dependencies cross the locked boundary")
    if failure_records:
        raise FoundationEvidenceError("Gate 6D2 contains unexpected candidate failures")

    recommendation = _recommendation(candidate_results, contract)
    paths = {
        "candidate_results": output_directory
        / str(contract["outputs"]["candidate_results"]),
        "outer_fold_results": output_directory
        / str(contract["outputs"]["outer_fold_results"]),
        "out_of_fold_predictions": output_directory
        / str(contract["outputs"]["out_of_fold_predictions"]),
        "resource_evidence": output_directory
        / str(contract["outputs"]["resource_evidence"]),
        "model_provenance_manifest": output_directory
        / str(contract["outputs"]["model_provenance_manifest"]),
        "failure_records": output_directory
        / str(contract["outputs"]["failure_records"]),
        "promotion_recommendation": output_directory
        / str(contract["outputs"]["promotion_recommendation"]),
    }
    candidate_results.to_csv(paths["candidate_results"], index=False, lineterminator="\n")
    folds.to_csv(paths["outer_fold_results"], index=False, lineterminator="\n")
    predictions.to_parquet(paths["out_of_fold_predictions"], index=False)
    resources.to_csv(paths["resource_evidence"], index=False, lineterminator="\n")
    _write_json(
        paths["model_provenance_manifest"],
        {
            "schema_version": "1.0.0",
            "gate": "6D",
            "subgate": "6D2",
            "candidate_count": len(provenance_records),
            "candidates": provenance_records,
        },
    )
    _write_json(
        paths["failure_records"],
        {
            "schema_version": "1.0.0",
            "gate": "6D",
            "subgate": "6D2",
            "records": failure_records,
        },
    )
    _write_json(paths["promotion_recommendation"], recommendation)

    output_hashes = {name: _sha256_path(path) for name, path in paths.items()}
    environment_hashes = {
        path.name: _sha256_path(path)
        for path in sorted(environment_directory.glob("*.txt"))
    }
    manifest = {
        "schema_version": "1.0.0",
        "gate": "6D",
        "subgate": "6D2",
        "status": "validation_complete_pending_human_decision",
        "execution_commit": _git_commit(root),
        "contract_version": str(contract["contract_version"]),
        "candidate_ids": expected_candidates,
        "candidate_count": len(expected_candidates),
        "successful_candidate_count": int(
            candidate_results["status"].eq("success").sum()
        ),
        "outer_fold_count": int(boundary["outer_fold_count"]),
        "validation_origins_per_candidate": int(boundary["validation_origin_count"]),
        "prediction_row_count": int(len(predictions)),
        "maximum_prediction_origin": int(predictions["row_position"].max()),
        "maximum_target_dependency": int(
            predictions["maximum_target_dependency"].max()
        ),
        "common_context_length_intervals": int(
            contract["benchmark_protocol"]["context_length_intervals"]
        ),
        "forecast_horizon_intervals": int(
            contract["benchmark_protocol"]["horizon_intervals"]
        ),
        "zero_shot_only": True,
        "quantile_outputs_authoritative": False,
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "fine_tuning_performed": False,
        "calibration_performed": False,
        "hyperparameter_search_performed": False,
        "automatic_promotion_permitted": False,
        "human_decision_required": True,
        "v1_immutable": True,
        "external_api_cost_usd": 0.0,
        "total_wall_clock_seconds": float(resources["wall_clock_seconds"].sum()),
        "output_hashes": output_hashes,
        "environment_hashes": environment_hashes,
        "next_gate": "6D3",
    }
    _write_json(execution_manifest_path, manifest)
    return manifest
