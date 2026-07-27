from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6e"
SCHEMAS = ROOT / "schemas"


class Gate6E3ClosureError(RuntimeError):
    """Raised when Gate 6E3 closure violates a frozen control."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6E3ClosureError(f"Expected a JSON object in {path}")
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
        raise Gate6E3ClosureError(f"{label} failed schema validation:\n{details}")


def main() -> None:
    decision_path = OUTPUT / "promotion_decision.json"
    closure_path = OUTPUT / "gate_6e_closure_manifest.json"
    execution_path = OUTPUT / "gate_6e_execution_manifest.json"
    recommendation_path = OUTPUT / "promotion_recommendation.json"
    configuration_results_path = OUTPUT / "configuration_results.csv"
    fold_results_path = OUTPUT / "outer_fold_results.csv"

    decision = _load_json(decision_path)
    closure = _load_json(closure_path)
    execution = _load_json(execution_path)
    recommendation = _load_json(recommendation_path)
    configurations = pd.read_csv(configuration_results_path)
    folds = pd.read_csv(fold_results_path)

    _validate(
        decision,
        "gate_6e3_human_decision.schema.json",
        "Gate 6E3 decision",
    )
    _validate(
        closure,
        "gate_6e_closure_manifest.schema.json",
        "Gate 6E closure manifest",
    )

    expected_configurations = {
        "expanding_all",
        "rolling_672",
        "rolling_2688",
        "aci_672_0p005",
        "aci_672_0p01",
        "aci_672_0p02",
        "aci_2688_0p005",
        "aci_2688_0p01",
        "aci_2688_0p02",
    }
    if set(decision["rejected_configurations"]) != expected_configurations:
        raise Gate6E3ClosureError("Gate 6E3 rejected-configuration set changed")
    if set(closure["rejected_configurations"]) != expected_configurations:
        raise Gate6E3ClosureError("Gate 6E closure configuration set changed")
    if set(configurations["configuration_id"].astype(str)) != expected_configurations:
        raise Gate6E3ClosureError("Gate 6E2 configuration identities changed")
    if set(folds["configuration_id"].astype(str)) != expected_configurations:
        raise Gate6E3ClosureError("Gate 6E2 fold identities changed")
    if len(configurations) != 9 or len(folds) != 36:
        raise Gate6E3ClosureError("Gate 6E2 configuration or fold count changed")

    if configurations["hard_constraints_passed"].astype(bool).any():
        raise Gate6E3ClosureError("A Gate 6E2 configuration unexpectedly passes all controls")
    if recommendation["eligible_configurations"] != []:
        raise Gate6E3ClosureError("Gate 6E2 recommendation contains an eligible configuration")
    if recommendation["pareto_eligible_configurations"] != []:
        raise Gate6E3ClosureError("Gate 6E2 recommendation contains a Pareto candidate")
    if recommendation["outcome"] != "no_action":
        raise Gate6E3ClosureError("Gate 6E2 no-action outcome changed")
    if recommendation["recommended_configuration"] is not None:
        raise Gate6E3ClosureError("Gate 6E2 unexpectedly recommends a configuration")
    if recommendation["human_decision_required"] is not True:
        raise Gate6E3ClosureError("Gate 6E2 bypasses human authority")
    if recommendation["recommended_next_gate"] != "6E3":
        raise Gate6E3ClosureError("Gate 6E2 recommendation does not lead to Gate 6E3")

    strongest = configurations.sort_values(
        "weighted_interval_score",
        kind="stable",
    ).iloc[0]
    if strongest["configuration_id"] != "aci_672_0p02":
        raise Gate6E3ClosureError("Strongest Gate 6E2 configuration identity changed")
    if abs(float(strongest["weighted_interval_score"]) - 2.2834847109806047) > 1e-12:
        raise Gate6E3ClosureError("Strongest configuration WIS changed")
    if abs(float(strongest["peak_state_weighted_interval_score"]) - 9.55071623587265) > 1e-12:
        raise Gate6E3ClosureError("Strongest configuration peak WIS changed")
    if abs(float(strongest["peak_state_coverage_90"]) - 0.6058823529411764) > 1e-12:
        raise Gate6E3ClosureError("Strongest configuration peak coverage changed")
    if "minimum_aggregate_peak_90_coverage" not in str(
        strongest["failed_hard_constraints"]
    ):
        raise Gate6E3ClosureError("Strongest configuration failure reason changed")

    if execution["status"] != "validation_complete_pending_human_decision":
        raise Gate6E3ClosureError("Gate 6E2 execution status changed")
    if execution["configuration_count"] != 9:
        raise Gate6E3ClosureError("Gate 6E2 configuration count changed")
    if execution["interval_prediction_row_count"] != 63036:
        raise Gate6E3ClosureError("Gate 6E2 interval row count changed")
    if execution["calibration_residual_row_count"] != 70935:
        raise Gate6E3ClosureError("Gate 6E2 calibration row count changed")
    if execution["validation_origins_per_configuration"] != 7004:
        raise Gate6E3ClosureError("Gate 6E2 validation-origin count changed")
    if execution["maximum_prediction_origin"] != 28027:
        raise Gate6E3ClosureError("Gate 6E2 maximum prediction origin changed")
    if execution["maximum_target_dependency"] != 28028:
        raise Gate6E3ClosureError("Gate 6E2 maximum dependency changed")
    if execution["recommendation_outcome"] != "no_action":
        raise Gate6E3ClosureError("Gate 6E2 manifest outcome changed")
    if execution["recommended_configuration"] is not None:
        raise Gate6E3ClosureError("Gate 6E2 manifest recommends a configuration")
    if execution["point_prediction_parity_violation_count"] != 0:
        raise Gate6E3ClosureError("Point-prediction parity boundary changed")

    false_controls = (
        "automatic_promotion_permitted",
        "confirmatory_evaluation_performed",
        "locked_predictions_parsed",
        "locked_test_accessed",
        "point_model_mutated",
        "point_model_search_performed",
        "probabilistic_authority_claimed",
    )
    for field in false_controls:
        if execution[field] is not False:
            raise Gate6E3ClosureError(f"Frozen Gate 6E2 control changed: {field}")
    if execution["v1_immutable"] is not True:
        raise Gate6E3ClosureError("V1 immutability changed")

    source_paths = {
        "calibration_lineage": OUTPUT / "calibration_lineage.json",
        "calibration_residuals": OUTPUT / "calibration_residuals.parquet",
        "configuration_results": configuration_results_path,
        "coverage_results": OUTPUT / "coverage_results.csv",
        "environment_lock": OUTPUT / "environment_lock.txt",
        "failure_records": OUTPUT / "failure_records.json",
        "interval_predictions": OUTPUT / "interval_predictions.parquet",
        "outer_fold_results": fold_results_path,
        "promotion_recommendation": recommendation_path,
        "resource_evidence": OUTPUT / "resource_evidence.csv",
    }
    if execution["output_hashes"] != closure["source_hashes"]:
        raise Gate6E3ClosureError("Closure source hashes differ from execution manifest")
    for name, path in source_paths.items():
        if not path.exists():
            raise Gate6E3ClosureError(f"Missing Gate 6E source evidence: {path}")
        if _sha256(path) != closure["source_hashes"][name]:
            raise Gate6E3ClosureError(f"Gate 6E source hash mismatch: {name}")

    if decision["decision_outcome"] != closure["decision_outcome"]:
        raise Gate6E3ClosureError("Decision and closure outcomes differ")
    if decision["retained_model"] != closure["retained_model"]:
        raise Gate6E3ClosureError("Decision and closure retained models differ")
    if decision["promoted_configuration"] is not None:
        raise Gate6E3ClosureError("Human decision promoted an uncertainty configuration")
    if closure["promoted_configuration"] is not None:
        raise Gate6E3ClosureError("Closure manifest promoted an uncertainty configuration")
    if decision["closed_gate"] != "6E" or closure["status"] != "closed":
        raise Gate6E3ClosureError("Gate 6E is not formally closed")
    if decision["next_gate"] != "6F" or closure["next_gate"] != "6F":
        raise Gate6E3ClosureError("Gate 6F is not the next permitted gate")
    if decision["next_gate_unblocked"] is not True:
        raise Gate6E3ClosureError("Gate 6F remains blocked in the human decision")
    if closure["controls"]["gate_6f_unblocked"] is not True:
        raise Gate6E3ClosureError("Gate 6F remains blocked in the closure manifest")
    if closure["blocked_gates"] != ["6G"]:
        raise Gate6E3ClosureError("Gate 6E closure blocked-gate sequence changed")

    print(
        "Gate 6E3 closure: PASS | configurations=9 | promoted=0 | "
        "outcome=no_action | retained=v1_frozen_champion | "
        "gate_6e=closed | next_gate=6F"
    )


if __name__ == "__main__":
    main()
