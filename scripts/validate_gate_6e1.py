from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from iaei.contracts import validate_optimization_governance
from iaei.modeling.uncertainty_governance import (
    APPROVED_UNCERTAINTY_METHODS,
    REFERENCE_CONFIGURATION,
    assert_no_gate_6e_execution_artifacts,
    build_gate_6e1_plan,
)
from iaei.uncertainty_contracts import (
    validate_gate_6e1_closure_manifest,
    validate_uncertainty_contract,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "iaei" / "modeling" / "uncertainty_governance.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "gate-6e1-uncertainty-contract.yml"
PROTOCOL_PATH = ROOT / "docs" / "GATE_6E_UNCERTAINTY_PROTOCOL.md"
SCOPE_PATH = ROOT / "docs" / "GATE_6E1_SCOPE_LOCK.md"


class Gate6E1ValidationError(RuntimeError):
    """Raised when the Gate 6E1 contract lock is inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Gate6E1ValidationError(f"Expected a JSON object in {path}")
    return value


def _validate_gate_6d_closure() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6d3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise Gate6E1ValidationError(
            "Gate 6D closure validation failed:\n"
            f"{completed.stdout}{completed.stderr}"
        )


def _validate_source_boundary() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    prohibited_functions = {
        "fit",
        "train",
        "predict",
        "forecast",
        "calibrate",
        "optimize",
        "evaluate",
        "score",
    }
    observed_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = sorted(observed_functions & prohibited_functions)
    if overlap:
        raise Gate6E1ValidationError(
            f"Gate 6E1 contains execution functions: {overlap}"
        )

    prohibited_import_roots = {
        "sklearn",
        "mapie",
        "torch",
        "xgboost",
        "lightgbm",
        "catboost",
        "chronos",
        "timesfm",
        "uni2ts",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    overlap = sorted(imports & prohibited_import_roots)
    if overlap:
        raise Gate6E1ValidationError(
            f"Gate 6E1 imported execution frameworks: {overlap}"
        )


def _validate_contract_alignment() -> None:
    contract = validate_uncertainty_contract()
    parent = validate_optimization_governance()
    plan = build_gate_6e1_plan(contract)

    if plan.method_ids != APPROVED_UNCERTAINTY_METHODS:
        raise Gate6E1ValidationError("Gate 6E1 method identities changed")
    if plan.configuration_count != 9:
        raise Gate6E1ValidationError("Gate 6E1 must contain exactly nine configurations")
    if plan.target_coverage_levels != (0.8, 0.9, 0.95):
        raise Gate6E1ValidationError("Gate 6E1 coverage levels changed")
    if contract["optimization"]["reference_configuration"] != REFERENCE_CONFIGURATION:
        raise Gate6E1ValidationError("Gate 6E1 reference configuration changed")
    if (
        contract["calibration_protocol"]["calibration_tail_fraction"]
        != parent["evidence_boundary"]["calibration_tail_fraction"]
    ):
        raise Gate6E1ValidationError("Gate 6E1 calibration tail conflicts with Gate 6A")
    if (
        contract["calibration_protocol"]["minimum_calibration_origins"]
        != parent["evidence_boundary"]["minimum_calibration_origins"]
    ):
        raise Gate6E1ValidationError(
            "Gate 6E1 minimum calibration size conflicts with Gate 6A"
        )
    if (
        contract["calibration_protocol"]["target_coverage_levels"]
        != parent["uncertainty"]["target_coverage_levels"]
    ):
        raise Gate6E1ValidationError("Gate 6E1 coverage levels conflict with Gate 6A")
    if (
        contract["evidence_boundary"]["outer_fold_count"]
        != parent["evidence_boundary"]["outer_fold_count"]
    ):
        raise Gate6E1ValidationError("Gate 6E1 outer folds conflict with Gate 6A")
    if (
        contract["evidence_boundary"]["inner_fold_count"]
        != parent["evidence_boundary"]["inner_fold_count"]
    ):
        raise Gate6E1ValidationError("Gate 6E1 inner folds conflict with Gate 6A")
    if (
        contract["evidence_boundary"]["purge_intervals"]
        != parent["evidence_boundary"]["purge_intervals"]
    ):
        raise Gate6E1ValidationError("Gate 6E1 purge conflicts with Gate 6A")
    if plan.point_model_search_permitted:
        raise Gate6E1ValidationError("Gate 6E1 reopens point-model search")
    if plan.locked_test_access_permitted:
        raise Gate6E1ValidationError("Gate 6E1 weakens the locked-test boundary")
    if plan.automatic_promotion_permitted:
        raise Gate6E1ValidationError("Gate 6E1 permits automatic promotion")


def _validate_cross_artifact_conformance() -> None:
    contract = validate_uncertainty_contract()
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    display_names = {
        "expanding_absolute_conformal": "Expanding absolute-residual conformal",
        "rolling_absolute_conformal": "Rolling absolute-residual conformal",
        "adaptive_absolute_conformal": "Adaptive conformal inference",
    }
    for method_id, display_name in display_names.items():
        if method_id not in APPROVED_UNCERTAINTY_METHODS:
            raise Gate6E1ValidationError(f"Unexpected method mapping: {method_id}")
        if display_name not in protocol or display_name not in scope:
            raise Gate6E1ValidationError(
                f"Missing cross-artifact uncertainty method: {display_name}"
            )

    if "scripts/validate_gate_6e1.py" not in workflow:
        raise Gate6E1ValidationError("Gate 6E1 workflow does not invoke its validator")
    prohibited_workflow_fragments = (
        "run_gate_6e2",
        "build_gate_6e2",
        "contents: write",
        "pip install mapie",
    )
    hits = [fragment for fragment in prohibited_workflow_fragments if fragment in workflow]
    if hits:
        raise Gate6E1ValidationError(
            f"Gate 6E1 workflow contains execution fragments: {hits}"
        )

    config_ids = [
        config["configuration_id"]
        for family in contract["candidate_families"]
        for config in family["configurations"]
    ]
    if len(config_ids) != len(set(config_ids)):
        raise Gate6E1ValidationError("Gate 6E1 configuration IDs are not unique")


def main() -> None:
    contract = validate_uncertainty_contract()
    closure = validate_gate_6e1_closure_manifest()
    plan = build_gate_6e1_plan(contract)

    _validate_gate_6d_closure()
    _validate_source_boundary()
    _validate_contract_alignment()
    _validate_cross_artifact_conformance()
    assert_no_gate_6e_execution_artifacts()

    if closure["status"] != "closed":
        raise Gate6E1ValidationError("Gate 6E1 closure is not closed")
    if closure["retained_point_model"] != "v1_frozen_champion":
        raise Gate6E1ValidationError("Gate 6E1 changed the retained point model")
    if tuple(closure["candidate_method_ids"]) != APPROVED_UNCERTAINTY_METHODS:
        raise Gate6E1ValidationError("Gate 6E1 closure method identities changed")
    if closure["configuration_count"] != plan.configuration_count:
        raise Gate6E1ValidationError("Gate 6E1 closure configuration count changed")
    if closure["next_gate"] != "6E2" or closure["next_gate_authorized"] is not False:
        raise Gate6E1ValidationError("Gate 6E2 requires separate explicit authorization")
    if closure["blocked_gates"] != ["6E2", "6E3", "6F", "6G"]:
        raise Gate6E1ValidationError("Gate 6E1 blocked-gate sequence changed")

    controls = closure["controls"]
    prohibited_true = (
        "point_model_search_performed",
        "point_model_fit_performed",
        "calibration_performed",
        "intervals_generated",
        "uncertainty_metrics_calculated",
        "optimization_executed",
        "locked_test_accessed",
        "locked_predictions_parsed",
        "confirmatory_evaluation_performed",
        "automatic_promotion_permitted",
        "probabilistic_authority_claimed",
    )
    if any(controls[field] is not False for field in prohibited_true):
        raise Gate6E1ValidationError("Gate 6E1 closure contains execution evidence")
    if controls["implementation_only"] is not True:
        raise Gate6E1ValidationError("Gate 6E1 is not implementation-only")
    if controls["v1_immutable"] is not True or controls["gate_6d_closed"] is not True:
        raise Gate6E1ValidationError("Gate 6E1 predecessor boundary changed")

    print(
        "Gate 6E1 validation: PASS | methods=3 | configurations=9 | "
        "coverage=0.80,0.90,0.95 | calibration=false | intervals=false | "
        "locked_test=false | point_model=v1_frozen_champion | "
        "next_gate=6E2_authorization_required"
    )


if __name__ == "__main__":
    main()
