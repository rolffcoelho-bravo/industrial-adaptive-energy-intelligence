from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from iaei.paths import ROOT, SCHEMAS


OUTPUT = ROOT / "outputs" / "v2" / "gate_6e"


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6e3_schemas_and_records_are_valid() -> None:
    decision_schema = _load_json(SCHEMAS / "gate_6e3_human_decision.schema.json")
    closure_schema = _load_json(SCHEMAS / "gate_6e_closure_manifest.schema.json")
    Draft202012Validator.check_schema(decision_schema)
    Draft202012Validator.check_schema(closure_schema)
    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6e_closure_manifest.json")
    Draft202012Validator(decision_schema).validate(decision)
    Draft202012Validator(closure_schema).validate(closure)


def test_gate_6e3_records_governed_no_action() -> None:
    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6e_closure_manifest.json")
    assert decision["decision_outcome"] == "no_action"
    assert decision["promoted_configuration"] is None
    assert len(decision["rejected_configurations"]) == 9
    assert closure["decision_outcome"] == "no_action"
    assert closure["promoted_configuration"] is None
    assert closure["validation_summary"]["eligible_configuration_count"] == 0
    assert closure["validation_summary"]["pareto_eligible_configuration_count"] == 0


def test_gate_6e3_preserves_v1_and_locked_test_boundaries() -> None:
    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6e_closure_manifest.json")
    assert decision["retained_model"] == "v1_frozen_champion"
    assert decision["locked_test_accessed"] is False
    assert decision["locked_predictions_parsed"] is False
    assert decision["confirmatory_evaluation_performed"] is False
    controls = closure["controls"]
    assert controls["v1_immutable"] is True
    assert controls["point_model_mutated"] is False
    assert controls["point_model_search_performed"] is False
    assert controls["uncertainty_layer_promoted"] is False
    assert controls["probabilistic_authority_claimed"] is False


def test_gate_6e3_closes_gate_and_unblocks_only_gate_6f() -> None:
    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6e_closure_manifest.json")
    assert decision["closed_gate"] == "6E"
    assert decision["next_gate"] == "6F"
    assert decision["next_gate_unblocked"] is True
    assert closure["status"] == "closed"
    assert closure["next_gate"] == "6F"
    assert closure["blocked_gates"] == ["6G"]


def test_gate_6e3_validator_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6e3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Gate 6E3 closure: PASS" in completed.stdout
