from __future__ import annotations

from pathlib import Path

import numpy as np

from iaei.contracts import validate_neural_forecasting_contract
from iaei.v2.neural_forecasting import (
    _configuration_parameters,
    _window_indices,
    validate_neural_forecasting_contract as validate_execution_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gate_6c_contract_validates_from_repository_checks() -> None:
    contract = validate_neural_forecasting_contract()

    assert contract["gate"] == "6C"
    assert contract["status"] == "approved_for_execution"
    assert contract["v1_boundary"]["locked_test_access_permitted"] is False
    assert contract["data_boundary"]["admissible_partitions"] == [
        "training",
        "validation",
    ]
    assert contract["predecessor_boundary"]["gate_6b_status"] == "closed"


def test_gate_6c_execution_contract_matches_gate_6a_budget_and_seeds() -> None:
    contract = validate_execution_contract(ROOT)

    assert contract["search"]["unique_configuration_count"] == 6
    assert contract["search"]["max_parallel_trials"] <= 4
    assert contract["search"]["max_wall_clock_minutes"] <= 720
    assert contract["training"]["stochastic_seeds"] == [
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
    ]


def test_gate_6c_has_three_bounded_neural_families() -> None:
    contract = validate_execution_contract(ROOT)
    families = contract["candidate_families"]

    assert [family["algorithm_id"] for family in families] == [
        "residual_mlp",
        "causal_tcn",
        "gru_sequence",
    ]
    assert all(family["configuration_count"] == 2 for family in families)
    assert sum(len(family["configurations"]) for family in families) == 6


def test_gate_6c_sequence_windows_end_at_origin_without_future_values() -> None:
    positions = np.asarray([15, 16, 27], dtype=np.int64)
    indices = _window_indices(positions, sequence_length=16)

    assert indices.shape == (3, 16)
    assert np.array_equal(indices[:, -1], positions)
    assert np.all(indices <= positions[:, None])
    assert indices[0, 0] == 0


def test_configuration_parameters_separate_identity_from_model_values() -> None:
    configuration_id, parameters = _configuration_parameters(
        {
            "configuration_id": "mlp_01",
            "hidden_width": 64,
            "learning_rate": 0.001,
        }
    )

    assert configuration_id == "mlp_01"
    assert parameters == {"hidden_width": 64, "learning_rate": 0.001}


def test_gate_6c_source_never_reads_locked_prediction_rows() -> None:
    source = (ROOT / "src" / "iaei" / "v2" / "neural_forecasting.py").read_text(
        encoding="utf-8"
    )

    assert "locked_test_predictions.csv" not in source
    assert "parse_locked" not in source
    assert "maximum_prediction_origin_exclusive" in source
    assert "maximum_target_dependency_exclusive" in source
    assert 'fold_payload["locked_test_start"]' in source


def test_gate_6c_claim_controls_remain_false() -> None:
    contract = validate_execution_contract(ROOT)
    controls = contract["claim_controls"]

    assert controls["production_claim_permitted"] is False
    assert controls["savings_claim_permitted"] is False
    assert controls["drift_claim_permitted"] is False
    assert controls["causal_claim_permitted"] is False
    assert controls["optimization_impact_claim_permitted"] is False
    assert controls["confirmatory_claim_permitted"] is False
    assert controls["provisional_validation_language_required"] is True


def test_gate_6c_training_protocol_is_bounded_and_deterministic() -> None:
    contract = validate_execution_contract(ROOT)
    training = contract["training"]

    assert training["device"] == "cpu"
    assert training["maximum_epochs"] == 6
    assert training["batch_size"] == 1024
    assert training["internal_early_stopping_permitted"] is False
    assert training["deterministic_algorithms_required"] is True
    assert training["torch_threads"] == 1
