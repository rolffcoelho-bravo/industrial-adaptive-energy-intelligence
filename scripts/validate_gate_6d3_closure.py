from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6d"
SCHEMAS = ROOT / "schemas"


class Gate6D3ClosureError(RuntimeError):
    """Raised when the Gate 6D3 closure violates a frozen control."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6D3ClosureError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(payload: dict[str, Any], schema_name: str, label: str) -> None:
    schema = _load_json(SCHEMAS / schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise Gate6D3ClosureError(f"{label} failed schema validation:\n{details}")


def main() -> None:
    decision_path = OUTPUT / "promotion_decision.json"
    closure_path = OUTPUT / "gate_6d_closure_manifest.json"
    execution_path = OUTPUT / "gate_6d_execution_manifest.json"
    recommendation_path = OUTPUT / "promotion_recommendation.json"
    candidate_results_path = OUTPUT / "candidate_results.csv"
    fold_results_path = OUTPUT / "outer_fold_results.csv"

    decision = _load_json(decision_path)
    closure = _load_json(closure_path)
    execution = _load_json(execution_path)
    recommendation = _load_json(recommendation_path)
    candidate_results = pd.read_csv(candidate_results_path)
    fold_results = pd.read_csv(fold_results_path)

    _validate(
        decision,
        "gate_6d3_human_decision.schema.json",
        "Gate 6D3 decision",
    )
    _validate(
        closure,
        "gate_6d_closure_manifest.schema.json",
        "Gate 6D closure manifest",
    )

    expected_candidates = {
        "chronos_2_zero_shot",
        "timesfm_2_5_zero_shot",
        "moirai_2_research_zero_shot",
    }
    decision_candidates = {
        str(item["candidate_id"]) for item in decision["decisions"]
    }
    closure_candidates = {
        str(item["candidate_id"]) for item in closure["candidate_decisions"]
    }
    if decision_candidates != expected_candidates:
        raise Gate6D3ClosureError("Gate 6D3 decision candidate set changed")
    if closure_candidates != expected_candidates:
        raise Gate6D3ClosureError("Gate 6D closure candidate set changed")
    if any(item["decision"] != "reject" for item in decision["decisions"]):
        raise Gate6D3ClosureError("Every foundation-model challenger must be rejected")
    if any(item["promotion_eligible"] is not False for item in decision["decisions"]):
        raise Gate6D3ClosureError("A rejected foundation model is promotion eligible")

    decision_by_id = {
        str(item["candidate_id"]): item for item in decision["decisions"]
    }
    if decision_by_id["moirai_2_research_zero_shot"]["next_action"] != (
        "retain_research_negative_benchmark"
    ):
        raise Gate6D3ClosureError("Moirai research-only disposition changed")
    for candidate_id in {
        "chronos_2_zero_shot",
        "timesfm_2_5_zero_shot",
    }:
        if decision_by_id[candidate_id]["next_action"] != "retain_incumbent":
            raise Gate6D3ClosureError("Commercial challenger disposition changed")

    if set(candidate_results["candidate_id"].astype(str)) != expected_candidates:
        raise Gate6D3ClosureError("Gate 6D2 candidate-result identities changed")
    if set(fold_results["candidate_id"].astype(str)) != expected_candidates:
        raise Gate6D3ClosureError("Gate 6D2 fold-result identities changed")
    if len(candidate_results) != 3 or len(fold_results) != 12:
        raise Gate6D3ClosureError("Gate 6D2 candidate or fold count changed")
    if not candidate_results["positive_outer_folds"].astype(int).eq(0).all():
        raise Gate6D3ClosureError("A foundation model unexpectedly has a positive fold")
    if not candidate_results["relative_mae_improvement_vs_v1"].astype(float).lt(0.0).all():
        raise Gate6D3ClosureError("A foundation model unexpectedly improves aggregate MAE")
    if not candidate_results["relative_peak_mae_change_vs_v1"].astype(float).gt(0.0).all():
        raise Gate6D3ClosureError("A foundation model unexpectedly protects peak-state MAE")

    strongest = candidate_results.sort_values("mean_mae", kind="stable").iloc[0]
    if strongest["candidate_id"] != "chronos_2_zero_shot":
        raise Gate6D3ClosureError("Gate 6D2 strongest-candidate identity changed")
    if abs(float(strongest["mean_mae"]) - 4.743580802977936) > 1e-12:
        raise Gate6D3ClosureError("Chronos-2 mean MAE changed")

    rows = candidate_results.set_index("candidate_id")
    if bool(rows.loc["chronos_2_zero_shot", "deterministic_replay_passed"]):
        raise Gate6D3ClosureError("Chronos-2 replay failure was removed")
    if bool(rows.loc["moirai_2_research_zero_shot", "commercial_use_eligible"]):
        raise Gate6D3ClosureError("Moirai became commercially eligible")
    if bool(rows.loc["moirai_2_research_zero_shot", "promotion_eligible_by_license"]):
        raise Gate6D3ClosureError("Moirai license promotion boundary changed")

    if recommendation["eligible_candidates"] != []:
        raise Gate6D3ClosureError("Gate 6D2 recommendation contains an eligible candidate")
    if recommendation["human_decision_required"] is not True:
        raise Gate6D3ClosureError("Gate 6D2 bypasses human decision authority")
    if recommendation["recommended_next_gate"] != "6D3":
        raise Gate6D3ClosureError("Gate 6D2 recommendation does not lead to Gate 6D3")

    if execution["status"] != "validation_complete_pending_human_decision":
        raise Gate6D3ClosureError("Gate 6D2 execution status changed")
    if execution["candidate_count"] != 3:
        raise Gate6D3ClosureError("Gate 6D2 candidate count changed")
    if execution["prediction_row_count"] != 21012:
        raise Gate6D3ClosureError("Gate 6D2 prediction count changed")
    if execution["maximum_prediction_origin"] != 28027:
        raise Gate6D3ClosureError("Gate 6D2 maximum prediction origin changed")
    if execution["maximum_target_dependency"] != 28028:
        raise Gate6D3ClosureError("Gate 6D2 maximum target dependency changed")
    for field in (
        "locked_test_accessed",
        "locked_predictions_parsed",
        "confirmatory_evaluation_performed",
        "fine_tuning_performed",
        "calibration_performed",
        "hyperparameter_search_performed",
        "automatic_promotion_permitted",
    ):
        if execution[field] is not False:
            raise Gate6D3ClosureError(f"Frozen Gate 6D2 control changed: {field}")
    if execution["v1_immutable"] is not True:
        raise Gate6D3ClosureError("V1 immutability control changed")

    source_paths = {
        "candidate_results": candidate_results_path,
        "outer_fold_results": fold_results_path,
        "out_of_fold_predictions": OUTPUT / "out_of_fold_predictions.parquet",
        "resource_evidence": OUTPUT / "resource_evidence.csv",
        "model_provenance_manifest": OUTPUT / "model_provenance_manifest.json",
        "failure_records": OUTPUT / "failure_records.json",
        "promotion_recommendation": recommendation_path,
    }
    if execution["output_hashes"] != closure["source_hashes"]:
        raise Gate6D3ClosureError("Closure source hashes differ from execution manifest")
    for name, path in source_paths.items():
        if not path.exists():
            raise Gate6D3ClosureError(f"Missing Gate 6D source evidence: {path}")
        if _sha256(path) != closure["source_hashes"][name]:
            raise Gate6D3ClosureError(f"Gate 6D source hash mismatch: {name}")

    environment_directory = OUTPUT / "environment_locks"
    observed_environment_hashes = {
        path.name: _sha256(path)
        for path in sorted(environment_directory.glob("*.txt"))
    }
    if observed_environment_hashes != closure["environment_hashes"]:
        raise Gate6D3ClosureError("Gate 6D environment hashes changed")
    if execution["environment_hashes"] != closure["environment_hashes"]:
        raise Gate6D3ClosureError("Closure environment hashes differ from execution manifest")

    if decision["retained_model"] != closure["retained_model"]:
        raise Gate6D3ClosureError("Decision and closure retained-model identities differ")
    if decision["closed_gate"] != "6D" or closure["status"] != "closed":
        raise Gate6D3ClosureError("Gate 6D is not formally closed")
    if decision["next_gate"] != "6E" or closure["next_gate"] != "6E":
        raise Gate6D3ClosureError("Gate 6E is not the next permitted gate")
    if decision["next_gate_unblocked"] is not True:
        raise Gate6D3ClosureError("Gate 6E remains blocked in the human decision")
    if closure["controls"]["gate_6e_unblocked"] is not True:
        raise Gate6D3ClosureError("Gate 6E remains blocked in the closure manifest")
    if set(closure["blocked_gates"]) != {"6F", "6G"}:
        raise Gate6D3ClosureError("Gate 6D closure blocked-gate set changed")

    print(
        "Gate 6D3 closure: PASS | candidates=3 | rejected=3 | "
        "retained=v1_frozen_champion | gate_6d=closed | next_gate=6E"
    )


if __name__ == "__main__":
    main()
