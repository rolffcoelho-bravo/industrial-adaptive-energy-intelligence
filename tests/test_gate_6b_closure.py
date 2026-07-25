from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "gate_6b"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_6b_human_decision_is_schema_valid() -> None:
    decision = _load(OUTPUT / "promotion_decision.json")
    schema = _load(ROOT / "schemas" / "promotion_decision.schema.json")

    Draft202012Validator(schema).validate(decision)

    assert decision["decision"] == "reject"
    assert decision["next_action"] == "retain_incumbent"
    assert decision["decided_by"]["authority"] == "human"
    assert decision["genai_role"]["vote_permitted"] is False


def test_gate_6b_closure_manifest_is_schema_valid() -> None:
    closure = _load(OUTPUT / "gate_6b_closure_manifest.json")
    schema = _load(ROOT / "schemas" / "gate_6b_closure_manifest.schema.json")

    Draft202012Validator(schema).validate(closure)

    assert closure["status"] == "closed"
    assert closure["human_decision"]["decision"] == "reject_all_challengers"
    assert closure["human_decision"]["retained_model"] == "v1_frozen_champion"
    assert closure["next_gate"] == "6C"


def test_gate_6b_closure_hashes_are_exact() -> None:
    closure = _load(OUTPUT / "gate_6b_closure_manifest.json")

    assert closure["execution"]["execution_manifest_sha256"] == _sha256(
        OUTPUT / "gate_6b_execution_manifest.json"
    )
    assert closure["execution"]["leaderboard_sha256"] == _sha256(
        OUTPUT / "candidate_leaderboard.csv"
    )
    assert closure["human_decision"]["decision_artifact_sha256"] == _sha256(
        OUTPUT / "promotion_decision.json"
    )


def test_gate_6b_closure_preserves_v1_and_locked_test_boundary() -> None:
    closure = _load(OUTPUT / "gate_6b_closure_manifest.json")

    assert closure["v1_boundary"] == {
        "confirmatory_evaluation_performed": False,
        "immutable": True,
        "locked_predictions_parsed": False,
        "locked_test_accessed": False,
        "tag": "v1.0.0",
    }
    assert closure["evidence_boundary"]["maximum_prediction_origin"] == 28027
    assert closure["evidence_boundary"]["maximum_target_dependency"] == 28028
    assert closure["evidence_boundary"]["locked_test_start"] == 28032


def test_all_gate_6b_challengers_are_rejected() -> None:
    closure = _load(OUTPUT / "gate_6b_closure_manifest.json")
    outcomes = closure["candidate_outcomes"]

    assert {item["algorithm_id"] for item in outcomes} == {
        "lightgbm_l1",
        "xgboost_hist",
        "catboost_mae",
    }
    assert all(item["decision"] == "rejected" for item in outcomes)
    assert all(item["promotion_eligible"] is False for item in outcomes)
    assert all(item["hard_constraints_all_pass"] is True for item in outcomes)
