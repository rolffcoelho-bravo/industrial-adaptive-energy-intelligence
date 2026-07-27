from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from iaei.modeling.uncertainty_governance import (
    APPROVED_UNCERTAINTY_METHODS,
    PROHIBITED_EXECUTION_ARTIFACTS,
    REFERENCE_CONFIGURATION,
    assert_gate_6e1_execution_boundary,
    build_gate_6e1_plan,
)
from iaei.paths import ROOT, SCHEMAS
from iaei.uncertainty_contracts import (
    validate_gate_6e1_closure_manifest,
    validate_uncertainty_contract,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6e1_contract_and_closure_schemas_are_valid() -> None:
    contract_schema = _load_json(
        SCHEMAS / "uncertainty_contract.schema.json"
    )
    closure_schema = _load_json(
        SCHEMAS / "gate_6e1_closure_manifest.schema.json"
    )
    Draft202012Validator.check_schema(contract_schema)
    Draft202012Validator.check_schema(closure_schema)
    validate_uncertainty_contract()
    validate_gate_6e1_closure_manifest()


def test_gate_6e1_freezes_method_and_configuration_set() -> None:
    contract = validate_uncertainty_contract()
    plan = build_gate_6e1_plan(contract)
    assert plan.method_ids == APPROVED_UNCERTAINTY_METHODS
    assert plan.configuration_count == 9
    assert plan.target_coverage_levels == (0.8, 0.9, 0.95)
    assert (
        contract["optimization"]["reference_configuration"]
        == REFERENCE_CONFIGURATION
    )
    assert contract["execution_budget"]["unique_configuration_count"] == 9


def test_gate_6e1_preserves_point_model_and_locked_test_boundaries() -> None:
    contract = validate_uncertainty_contract()
    closure = validate_gate_6e1_closure_manifest()
    assert (
        contract["point_model_boundary"]["model_id"]
        == "v1_frozen_champion"
    )
    assert (
        contract["point_model_boundary"]["selected_model"]
        == "hist_gradient_boosting"
    )
    assert (
        contract["point_model_boundary"]["point_model_search_permitted"]
        is False
    )
    assert (
        contract["point_model_boundary"][
            "point_prediction_mutation_permitted"
        ]
        is False
    )
    assert (
        contract["evidence_boundary"][
            "locked_test_partition_permitted"
        ]
        is False
    )
    assert (
        contract["evidence_boundary"][
            "locked_prediction_parsing_permitted"
        ]
        is False
    )
    assert (
        contract["evidence_boundary"][
            "confirmatory_evaluation_permitted"
        ]
        is False
    )
    assert closure["controls"]["v1_immutable"] is True
    assert closure["controls"]["gate_6d_closed"] is True


def test_gate_6e1_calibration_chronology_is_frozen() -> None:
    contract = validate_uncertainty_contract()
    protocol = contract["calibration_protocol"]
    assert protocol["separate_from_point_fit"] is True
    assert protocol["inner_oof_residuals_required"] is True
    assert protocol["in_sample_residuals_permitted"] is False
    assert protocol["calibration_tail_fraction"] == 0.15
    assert protocol["minimum_calibration_origins"] == 672
    assert protocol["first_outer_origin_uses_training_only_residuals"] is True
    assert (
        protocol["update_information_rule"]
        == "revealed_targets_strictly_before_current_target"
    )


def test_gate_6e1_promotion_is_fail_closed_and_human_authorized() -> None:
    contract = validate_uncertainty_contract()
    optimization = contract["optimization"]
    assert optimization["no_action_when_no_configuration_feasible"] is True
    assert optimization["reference_may_be_selected_when_feasible"] is True
    assert optimization["automatic_promotion_permitted"] is False
    assert optimization["final_authority"] == "human"
    assert optimization["genai_vote_permitted"] is False
    assert contract["next_gate_authorized"] is False
    assert contract["blocked_gates"] == ["6E2", "6E3", "6F", "6G"]


def test_gate_6e1_boundary_accepts_no_successor_artifacts(
    tmp_path: Path,
) -> None:
    assert_gate_6e1_execution_boundary(tmp_path)


def test_gate_6e1_boundary_rejects_partial_successor_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs" / "v2" / "gate_6e"
    output.mkdir(parents=True)
    (output / "configuration_results.csv").write_text(
        "placeholder\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="partial or inconsistent"):
        assert_gate_6e1_execution_boundary(tmp_path)


def test_gate_6e1_boundary_accepts_authorized_gate_6e2_successor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs" / "v2" / "gate_6e"
    output.mkdir(parents=True)
    for name in PROHIBITED_EXECUTION_ARTIFACTS:
        if name != "gate_6e_execution_manifest.json":
            (output / name).write_text("placeholder\n", encoding="utf-8")
    manifest = {
        "gate": "6E",
        "subgate": "6E2",
        "status": "validation_complete_pending_human_decision",
        "retained_point_model": "v1_frozen_champion",
        "next_gate": "6E3",
        "next_gate_authorized": False,
        "automatic_promotion_permitted": False,
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "point_model_search_performed": False,
        "point_model_mutated": False,
        "v1_immutable": True,
        "blocked_gates": ["6E3", "6F", "6G"],
    }
    (output / "gate_6e_execution_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert_gate_6e1_execution_boundary(tmp_path)


def test_gate_6e1_validator_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6e1.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Gate 6E1 validation: PASS" in completed.stdout
