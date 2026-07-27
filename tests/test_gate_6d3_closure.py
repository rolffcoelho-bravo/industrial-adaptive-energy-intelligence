from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6d"
SCHEMAS = ROOT / "schemas"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6d3_schemas_and_records_are_valid() -> None:
    decision_schema = _load_json(SCHEMAS / "gate_6d3_human_decision.schema.json")
    closure_schema = _load_json(SCHEMAS / "gate_6d_closure_manifest.schema.json")
    Draft202012Validator.check_schema(decision_schema)
    Draft202012Validator.check_schema(closure_schema)

    decision = _load_json(OUTPUT / "promotion_decision.json")
    closure = _load_json(OUTPUT / "gate_6d_closure_manifest.json")
    assert list(Draft202012Validator(decision_schema).iter_errors(decision)) == []
    assert list(Draft202012Validator(closure_schema).iter_errors(closure)) == []


def test_gate_6d3_rejects_every_foundation_challenger() -> None:
    decision = _load_json(OUTPUT / "promotion_decision.json")
    observed = {item["candidate_id"]: item for item in decision["decisions"]}
    assert set(observed) == {
        "chronos_2_zero_shot",
        "timesfm_2_5_zero_shot",
        "moirai_2_research_zero_shot",
    }
    assert all(item["decision"] == "reject" for item in observed.values())
    assert all(item["promotion_eligible"] is False for item in observed.values())
    assert observed["chronos_2_zero_shot"]["next_action"] == "retain_incumbent"
    assert observed["timesfm_2_5_zero_shot"]["next_action"] == "retain_incumbent"
    assert observed["moirai_2_research_zero_shot"]["next_action"] == (
        "retain_research_negative_benchmark"
    )
    assert decision["retained_model"] == "v1_frozen_champion"
    assert decision["automatic_promotion_permitted"] is False
    assert decision["human_authority"] is True
    assert decision["closed_gate"] == "6D"
    assert decision["next_gate"] == "6E"
    assert decision["next_gate_unblocked"] is True


def test_gate_6d3_decision_matches_validation_evidence() -> None:
    candidates = pd.read_csv(OUTPUT / "candidate_results.csv")
    assert len(candidates) == 3
    assert candidates["positive_outer_folds"].astype(int).eq(0).all()
    assert candidates["relative_mae_improvement_vs_v1"].astype(float).lt(0.0).all()
    assert candidates["relative_peak_mae_change_vs_v1"].astype(float).gt(0.0).all()
    assert candidates.sort_values("mean_mae", kind="stable").iloc[0][
        "candidate_id"
    ] == "chronos_2_zero_shot"


def test_gate_6d3_preserves_replay_and_license_rejection_evidence() -> None:
    candidates = pd.read_csv(OUTPUT / "candidate_results.csv").set_index("candidate_id")
    assert bool(
        candidates.loc["chronos_2_zero_shot", "deterministic_replay_passed"]
    ) is False
    assert bool(
        candidates.loc["moirai_2_research_zero_shot", "commercial_use_eligible"]
    ) is False
    assert bool(
        candidates.loc[
            "moirai_2_research_zero_shot", "promotion_eligible_by_license"
        ]
    ) is False


def test_gate_6d3_closure_preserves_frozen_boundaries() -> None:
    closure = _load_json(OUTPUT / "gate_6d_closure_manifest.json")
    controls = closure["controls"]
    assert controls["human_decision_recorded"] is True
    assert controls["automatic_promotion_permitted"] is False
    assert controls["locked_test_accessed"] is False
    assert controls["locked_predictions_parsed"] is False
    assert controls["confirmatory_evaluation_performed"] is False
    assert controls["fine_tuning_performed"] is False
    assert controls["calibration_performed"] is False
    assert controls["hyperparameter_search_performed"] is False
    assert controls["v1_immutable"] is True
    assert controls["rescue_lane_opened"] is False
    assert controls["probabilistic_authority_claimed"] is False
    assert controls["commercial_license_boundary_preserved"] is True
    assert controls["gate_6e_unblocked"] is True
    assert closure["status"] == "closed"
    assert closure["next_gate"] == "6E"
    assert set(closure["blocked_gates"]) == {"6F", "6G"}


def test_gate_6d3_validator_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6d3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Gate 6D3 closure: PASS" in completed.stdout


def test_gate_6d3_does_not_mutate_gate_6d2_execution_state() -> None:
    execution = _load_json(OUTPUT / "gate_6d_execution_manifest.json")
    assert execution["status"] == "validation_complete_pending_human_decision"
    assert execution["subgate"] == "6D2"
    assert execution["human_decision_required"] is True
    assert execution["locked_test_accessed"] is False
    assert execution["locked_predictions_parsed"] is False
    assert execution["confirmatory_evaluation_performed"] is False
    assert execution["fine_tuning_performed"] is False
    assert execution["calibration_performed"] is False
    assert execution["hyperparameter_search_performed"] is False
    assert execution["v1_immutable"] is True
