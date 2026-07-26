from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6c"
SCHEMAS = ROOT / "schemas"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6c3_schemas_and_records_are_valid() -> None:
    decision_schema = _load_json(SCHEMAS / "gate_6c3_human_decision.schema.json")
    closure_schema = _load_json(SCHEMAS / "gate_6c_closure_manifest.schema.json")
    Draft202012Validator.check_schema(decision_schema)
    Draft202012Validator.check_schema(closure_schema)

    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6c_closure_manifest.json")
    assert list(Draft202012Validator(decision_schema).iter_errors(decision)) == []
    assert list(Draft202012Validator(closure_schema).iter_errors(closure)) == []


def test_gate_6c3_rejects_every_neural_challenger() -> None:
    decision = _load_json(OUTPUT / "promotion_decision.json")
    observed = {item["candidate_id"]: item for item in decision["decisions"]}
    assert set(observed) == {
        "nhits_compact",
        "tide_compact",
        "patchtst_compact",
    }
    assert all(item["decision"] == "reject" for item in observed.values())
    assert all(item["promotion_eligible"] is False for item in observed.values())
    assert all(item["next_action"] == "retain_incumbent" for item in observed.values())
    assert decision["retained_model"] == "v1_frozen_champion"
    assert decision["automatic_promotion_permitted"] is False
    assert decision["human_authority"] is True
    assert decision["next_gate"] == "6D"


def test_gate_6c3_decision_matches_validation_evidence() -> None:
    leaderboard = pd.read_csv(OUTPUT / "candidate_leaderboard.csv")
    assert len(leaderboard) == 3
    assert leaderboard["promotion_eligible"].astype(bool).eq(False).all()
    assert leaderboard["positive_outer_folds"].astype(int).eq(0).all()
    assert leaderboard["relative_mae_improvement_vs_v1"].astype(float).lt(0.0).all()
    assert leaderboard["relative_peak_mae_change_vs_v1"].astype(float).gt(0.0).all()


def test_gate_6c3_closure_preserves_frozen_boundaries() -> None:
    closure = _load_json(OUTPUT / "gate_6c_closure_manifest.json")
    controls = closure["controls"]
    assert controls["human_decision_recorded"] is True
    assert controls["automatic_promotion_permitted"] is False
    assert controls["locked_test_accessed"] is False
    assert controls["locked_predictions_parsed"] is False
    assert controls["confirmatory_evaluation_performed"] is False
    assert controls["v1_immutable"] is True
    assert controls["quarantined_evidence_used"] is False
    assert closure["status"] == "closed"
    assert closure["next_gate"] == "6D"
    assert closure["blocked_gates"] == []


def test_gate_6c3_validator_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6c3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Gate 6C3 closure: PASS" in completed.stdout


def test_gate_6c3_does_not_mutate_gate_6c2_execution_state() -> None:
    execution = _load_json(OUTPUT / "gate_6c_execution_manifest.json")
    assert execution["status"] == "validation_complete_pending_human_decision"
    assert execution["subgate"] == "6C2"
    assert execution["human_decision_required"] is True
    assert execution["locked_test_accessed"] is False
    assert execution["confirmatory_evaluation_performed"] is False
    assert execution["v1_immutable"] is True
