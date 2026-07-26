from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator

from iaei.contracts import (
    ContractError,
    load_json,
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
    validate_neural_seed_governance_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ContractError(f"{label} is missing columns: {missing}")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.map(math.isfinite).all():
            raise ContractError(f"{label} contains nonfinite values in {column}")


def main() -> None:
    contract = validate_neural_seed_governance_alignment(
        validate_neural_forecasting_contract()
    )
    closure = validate_gate_6c1_closure_manifest()
    if closure["status"] != "closed":
        raise ContractError("Gate 6C1 closure is not closed")

    required_paths = {
        "seed_results": OUTPUT / str(contract["outputs"]["seed_results"]),
        "outer_fold_results": OUTPUT / str(contract["outputs"]["outer_fold_results"]),
        "candidate_leaderboard": OUTPUT
        / str(contract["outputs"]["candidate_leaderboard"]),
        "out_of_fold_predictions": OUTPUT
        / str(contract["outputs"]["out_of_fold_predictions"]),
        "trial_evidence": OUTPUT / str(contract["outputs"]["trial_evidence"]),
        "execution_manifest": OUTPUT
        / str(contract["outputs"]["execution_manifest"]),
        "promotion_recommendation": OUTPUT
        / str(contract["outputs"]["promotion_recommendation"]),
        "failure_records": OUTPUT / "failure_records.json",
        "environment_lock": OUTPUT / "environment_lock.txt",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise ContractError(f"Gate 6C2 evidence is incomplete: {missing}")

    seed_results = pd.read_csv(required_paths["seed_results"])
    outer = pd.read_csv(required_paths["outer_fold_results"])
    leaderboard = pd.read_csv(required_paths["candidate_leaderboard"])
    predictions = pd.read_parquet(required_paths["out_of_fold_predictions"])
    trial = load_json(required_paths["trial_evidence"])
    manifest = load_json(required_paths["execution_manifest"])
    recommendation = load_json(required_paths["promotion_recommendation"])
    failures = load_json(required_paths["failure_records"])

    candidates = [
        str(candidate["algorithm_id"]) for candidate in contract["candidate_families"]
    ]
    seeds = [int(seed) for seed in contract["search"]["seeds"]]
    expected_seed_rows = len(candidates) * len(seeds) * 4
    if len(seed_results) != expected_seed_rows:
        raise ContractError("Gate 6C2 does not contain all candidate-seed-fold rows")
    observed_keys = set(
        zip(
            seed_results["algorithm_id"].astype(str),
            seed_results["seed"].astype(int),
            seed_results["fold_id"].astype(int),
            strict=True,
        )
    )
    expected_keys = {
        (candidate, seed, fold_id)
        for candidate in candidates
        for seed in seeds
        for fold_id in range(1, 5)
    }
    if observed_keys != expected_keys:
        raise ContractError("Gate 6C2 candidate-seed-fold coverage changed")

    _require_finite(
        seed_results,
        [
            "mae",
            "peak_mae",
            "model_size_bytes",
            "p95_inference_latency_ms_per_1000_rows",
            "peak_memory_mb",
            "wall_clock_seconds",
        ],
        "Gate 6C2 seed results",
    )
    if not seed_results["status"].eq("success").all():
        raise ContractError("Gate 6C2 contains unsuccessful seed evidence")
    if not seed_results["cpu_portability_passed"].astype(bool).all():
        raise ContractError("Gate 6C2 CPU portability failed")
    if seed_results["maximum_prediction_origin"].max() >= 28028:
        raise ContractError("Gate 6C2 prediction origins cross the frozen boundary")
    if seed_results["maximum_target_dependency"].max() >= 28032:
        raise ContractError("Gate 6C2 target dependencies cross the frozen boundary")

    if len(outer) != len(candidates) * 4:
        raise ContractError("Gate 6C2 outer-fold aggregation is incomplete")
    if len(leaderboard) != len(candidates):
        raise ContractError("Gate 6C2 candidate leaderboard is incomplete")
    if set(leaderboard["algorithm_id"].astype(str)) != set(candidates):
        raise ContractError("Gate 6C2 candidate leaderboard identities changed")
    _require_finite(
        leaderboard,
        [
            "mean_mae",
            "mean_peak_mae",
            "relative_mae_improvement_vs_v1",
            "relative_peak_mae_change_vs_v1",
            "across_seed_mae_standard_deviation",
            "across_seed_peak_mae_standard_deviation",
            "outer_fold_mae_dispersion",
            "mean_model_size_bytes",
            "max_p95_inference_latency_ms_per_1000_rows",
            "max_peak_memory_mb",
            "total_wall_clock_seconds",
        ],
        "Gate 6C2 leaderboard",
    )

    validation_origins = 7004
    expected_prediction_rows = len(candidates) * len(seeds) * validation_origins
    if len(predictions) != expected_prediction_rows:
        raise ContractError("Gate 6C2 out-of-fold prediction count changed")
    if predictions["row_position"].max() >= 28028:
        raise ContractError("Gate 6C2 prediction artifact enters the locked boundary")
    if predictions[["actual", "prediction"]].isna().any().any():
        raise ContractError("Gate 6C2 predictions contain missing values")

    seed_schema = load_json(ROOT / "schemas" / "neural_seed_evidence.schema.json")
    candidate_schema = load_json(
        ROOT / "schemas" / "neural_candidate_evidence.schema.json"
    )
    seed_records = trial.get("seed_records")
    candidate_records = trial.get("candidate_records")
    if not isinstance(seed_records, list) or len(seed_records) != expected_seed_rows:
        raise ContractError("Gate 6C2 seed-record package is incomplete")
    if not isinstance(candidate_records, list) or len(candidate_records) != len(candidates):
        raise ContractError("Gate 6C2 candidate-record package is incomplete")
    for record in seed_records:
        _validate_payload(record, seed_schema, "Gate 6C2 seed evidence")
    for record in candidate_records:
        _validate_payload(record, candidate_schema, "Gate 6C2 candidate evidence")

    if failures.get("records") != [] or trial.get("failure_records") != []:
        raise ContractError("Gate 6C2 contains execution failures")
    if manifest["status"] != "validation_complete_pending_human_decision":
        raise ContractError("Gate 6C2 execution status is not pending human decision")
    if manifest["seed_fold_evaluation_count"] != expected_seed_rows:
        raise ContractError("Gate 6C2 manifest evaluation count changed")
    if manifest["prediction_row_count"] != expected_prediction_rows:
        raise ContractError("Gate 6C2 manifest prediction count changed")
    if manifest["seeds"] != seeds or manifest["candidate_ids"] != candidates:
        raise ContractError("Gate 6C2 manifest identities changed")
    if any(
        bool(manifest[field])
        for field in (
            "locked_test_accessed",
            "locked_predictions_parsed",
            "confirmatory_evaluation_performed",
            "automatic_promotion_permitted",
        )
    ):
        raise ContractError("Gate 6C2 manifest weakens a frozen control")
    if manifest["v1_immutable"] is not True or manifest["human_decision_required"] is not True:
        raise ContractError("Gate 6C2 manifest governance state changed")
    if manifest["next_gate"] != "6C3" or manifest["blocked_gates"] != ["6D"]:
        raise ContractError("Gate 6C2 gate sequence changed")

    for name, expected_hash in manifest["output_hashes"].items():
        path = required_paths[name]
        if _sha256(path) != expected_hash:
            raise ContractError(f"Gate 6C2 output hash mismatch: {name}")
    if _sha256(required_paths["environment_lock"]) != manifest["environment"][
        "environment_lock_sha256"
    ]:
        raise ContractError("Gate 6C2 environment lock hash changed")

    if recommendation["status"] != "validation_complete_pending_human_decision":
        raise ContractError("Gate 6C2 recommendation is not awaiting human decision")
    if recommendation["automatic_promotion_permitted"] is not False:
        raise ContractError("Gate 6C2 recommendation permits automatic promotion")
    if recommendation["human_decision_required"] is not True:
        raise ContractError("Gate 6C2 recommendation bypasses human authority")
    if recommendation["recommended_next_gate"] != "6C3":
        raise ContractError("Gate 6C2 recommendation skips Gate 6C3")
    if (OUTPUT / str(contract["outputs"]["promotion_decision"])).exists():
        raise ContractError("Gate 6C2 must not create a human promotion decision")
    if (OUTPUT / str(contract["outputs"]["closure_manifest"])).exists():
        raise ContractError("Gate 6C2 must not close Gate 6C")

    print(
        "Gate 6C2 evidence: PASS | evaluations={} | predictions={} | "
        "candidates={} | next_gate=6C3".format(
            expected_seed_rows,
            expected_prediction_rows,
            len(candidates),
        )
    )


if __name__ == "__main__":
    main()
