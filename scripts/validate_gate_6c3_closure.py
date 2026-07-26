from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6c"
SCHEMAS = ROOT / "schemas"


class Gate6C3ClosureError(RuntimeError):
    """Raised when the Gate 6C3 closure violates a frozen control."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6C3ClosureError(f"Expected a JSON object in {path}")
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
        raise Gate6C3ClosureError(f"{label} failed schema validation:\n{details}")


def main() -> None:
    decision_path = OUTPUT / "promotion_decision.json"
    closure_path = OUTPUT / "gate_6c_closure_manifest.json"
    execution_path = OUTPUT / "gate_6c_execution_manifest.json"
    recommendation_path = OUTPUT / "promotion_recommendation.json"
    leaderboard_path = OUTPUT / "candidate_leaderboard.csv"

    decision = _load_json(decision_path)
    closure = _load_json(closure_path)
    execution = _load_json(execution_path)
    recommendation = _load_json(recommendation_path)
    leaderboard = pd.read_csv(leaderboard_path)

    _validate(decision, "gate_6c3_human_decision.schema.json", "Gate 6C3 decision")
    _validate(
        closure,
        "gate_6c_closure_manifest.schema.json",
        "Gate 6C closure manifest",
    )

    expected_candidates = {
        "nhits_compact",
        "tide_compact",
        "patchtst_compact",
    }
    decision_candidates = {
        str(item["candidate_id"]) for item in decision["decisions"]
    }
    closure_candidates = {
        str(item["candidate_id"]) for item in closure["candidate_decisions"]
    }
    if decision_candidates != expected_candidates:
        raise Gate6C3ClosureError("Gate 6C3 decision candidate set changed")
    if closure_candidates != expected_candidates:
        raise Gate6C3ClosureError("Gate 6C closure candidate set changed")
    if any(item["decision"] != "reject" for item in decision["decisions"]):
        raise Gate6C3ClosureError("Every neural challenger must be rejected")
    if any(item["next_action"] != "retain_incumbent" for item in decision["decisions"]):
        raise Gate6C3ClosureError("Gate 6C3 must retain the frozen incumbent")

    if set(leaderboard["algorithm_id"].astype(str)) != expected_candidates:
        raise Gate6C3ClosureError("Gate 6C2 leaderboard candidate set changed")
    if leaderboard["promotion_eligible"].astype(bool).any():
        raise Gate6C3ClosureError("A rejected candidate is marked promotion eligible")
    if not leaderboard["positive_outer_folds"].astype(int).eq(0).all():
        raise Gate6C3ClosureError("Gate 6C2 positive-fold evidence changed")
    if not leaderboard["relative_mae_improvement_vs_v1"].astype(float).lt(0.0).all():
        raise Gate6C3ClosureError("A neural candidate unexpectedly improves aggregate MAE")
    if not leaderboard["relative_peak_mae_change_vs_v1"].astype(float).gt(0.0).all():
        raise Gate6C3ClosureError("A neural candidate unexpectedly protects peak-state MAE")

    if recommendation["eligible_candidates"] != []:
        raise Gate6C3ClosureError("Gate 6C2 recommendation contains an eligible candidate")
    if recommendation["human_decision_required"] is not True:
        raise Gate6C3ClosureError("Gate 6C2 bypasses human decision authority")
    if recommendation["recommended_next_gate"] != "6C3":
        raise Gate6C3ClosureError("Gate 6C2 recommendation does not lead to Gate 6C3")

    if execution["status"] != "validation_complete_pending_human_decision":
        raise Gate6C3ClosureError("Gate 6C2 execution status changed")
    if execution["seed_fold_evaluation_count"] != 60:
        raise Gate6C3ClosureError("Gate 6C2 evaluation count changed")
    if execution["prediction_row_count"] != 105060:
        raise Gate6C3ClosureError("Gate 6C2 prediction count changed")
    if execution["maximum_prediction_origin"] != 28027:
        raise Gate6C3ClosureError("Gate 6C2 maximum prediction origin changed")
    if execution["maximum_target_dependency"] != 28028:
        raise Gate6C3ClosureError("Gate 6C2 maximum target dependency changed")
    for field in (
        "locked_test_accessed",
        "locked_predictions_parsed",
        "confirmatory_evaluation_performed",
        "automatic_promotion_permitted",
    ):
        if execution[field] is not False:
            raise Gate6C3ClosureError(f"Frozen Gate 6C2 control changed: {field}")
    if execution["v1_immutable"] is not True:
        raise Gate6C3ClosureError("V1 immutability control changed")

    source_paths = {
        "candidate_leaderboard": leaderboard_path,
        "promotion_recommendation": recommendation_path,
        "seed_results": OUTPUT / "seed_results.csv",
        "outer_fold_results": OUTPUT / "outer_fold_results.csv",
        "out_of_fold_predictions": OUTPUT / "out_of_fold_predictions.parquet",
        "trial_evidence": OUTPUT / "trial_evidence.json",
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise Gate6C3ClosureError(f"Missing Gate 6C source evidence: {path}")
        if _sha256(path) != closure["source_hashes"][name]:
            raise Gate6C3ClosureError(f"Gate 6C source hash mismatch: {name}")

    if decision["retained_model"] != closure["retained_model"]:
        raise Gate6C3ClosureError("Decision and closure retained-model identities differ")
    if decision["next_gate"] != "6D" or closure["next_gate"] != "6D":
        raise Gate6C3ClosureError("Gate 6D is not the next permitted gate")
    if closure["blocked_gates"] != []:
        raise Gate6C3ClosureError("Gate 6C closure leaves an unexpected gate blocked")

    print(
        "Gate 6C3 closure: PASS | candidates=3 | rejected=3 | "
        "retained=v1_frozen_champion | next_gate=6D"
    )


if __name__ == "__main__":
    main()
