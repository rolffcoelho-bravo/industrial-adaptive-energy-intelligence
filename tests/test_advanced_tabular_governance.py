from __future__ import annotations

from pathlib import Path

from iaei.contracts import validate_advanced_tabular_contract
from iaei.v2.advanced_tabular import (
    _configuration_parameters,
    validate_advanced_tabular_contract as validate_execution_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gate_6b_contract_validates_from_repository_checks() -> None:
    contract = validate_advanced_tabular_contract()

    assert contract["gate"] == "6B"
    assert contract["status"] == "closed"
    assert contract["v1_boundary"]["locked_test_access_permitted"] is False
    assert contract["data_boundary"]["admissible_partitions"] == [
        "training",
        "validation",
    ]


def test_gate_6b_execution_contract_matches_gate_6a_budget() -> None:
    contract = validate_execution_contract(ROOT)

    assert contract["search"]["unique_configuration_count"] == 12
    assert contract["search"]["max_parallel_trials"] <= 4
    assert contract["search"]["max_wall_clock_minutes"] <= 360


def test_gate_6b_has_three_bounded_candidate_families() -> None:
    contract = validate_execution_contract(ROOT)
    families = contract["candidate_families"]

    assert [family["algorithm_id"] for family in families] == [
        "xgboost_hist",
        "lightgbm_l1",
        "catboost_mae",
    ]
    assert all(family["configuration_count"] == 4 for family in families)
    assert sum(len(family["configurations"]) for family in families) == 12


def test_configuration_parameters_separate_identity_from_model_values() -> None:
    configuration_id, parameters = _configuration_parameters(
        {
            "configuration_id": "candidate_01",
            "n_estimators": 200,
            "learning_rate": 0.05,
        }
    )

    assert configuration_id == "candidate_01"
    assert parameters == {"n_estimators": 200, "learning_rate": 0.05}


def test_gate_6b_source_never_reads_locked_prediction_rows() -> None:
    source = (ROOT / "src" / "iaei" / "v2" / "advanced_tabular.py").read_text(
        encoding="utf-8"
    )

    assert "locked_test_predictions.csv" not in source
    assert "parse_locked" not in source
    assert "maximum_prediction_origin_exclusive" in source
    assert "maximum_target_dependency_exclusive" in source


def test_gate_6b_claim_controls_remain_false() -> None:
    contract = validate_execution_contract(ROOT)

    assert set(contract["claim_controls"].values()) == {False, True}
    assert contract["claim_controls"]["production_claim_permitted"] is False
    assert contract["claim_controls"]["savings_claim_permitted"] is False
    assert contract["claim_controls"]["drift_claim_permitted"] is False
    assert contract["claim_controls"]["causal_claim_permitted"] is False
    assert contract["claim_controls"]["confirmatory_claim_permitted"] is False
    assert contract["claim_controls"]["provisional_validation_language_required"] is True
