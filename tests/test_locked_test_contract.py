from __future__ import annotations

from pathlib import Path

from iaei.contracts import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "locked_test_contract.yml"


def _contract() -> dict:
    return load_yaml(CONTRACT_PATH)


def test_locked_test_contract_is_pending_execution() -> None:
    contract = _contract()
    governance = contract["governance"]

    assert governance["decision_gate"] == "4E"
    assert governance["status"] == "locked_pending_execution"
    assert governance["maximum_evaluation_count"] == 1
    assert governance["repeated_test_evaluation_prohibited"] is True
    assert contract["selected_model_source"]["parameters_immutable"] is True


def test_locked_test_origin_boundary_is_exact() -> None:
    boundary = _contract()["locked_test_boundary"]

    assert boundary["locked_test_start"] == 28_032
    assert boundary["locked_test_stop_exclusive"] == 35_040
    assert boundary["maximum_target_horizon_steps"] == 4
    assert boundary["evaluation_origin_start"] == 28_032
    assert boundary["evaluation_origin_stop_exclusive"] == 35_036
    assert boundary["evaluation_origin_count"] == 7_004
    assert boundary["maximum_evaluation_origin"] == 35_035
    assert boundary["maximum_target_dependency"] == 35_039
    assert (
        boundary["evaluation_origin_stop_exclusive"]
        - boundary["evaluation_origin_start"]
        == boundary["evaluation_origin_count"]
    )


def test_temporal_blocks_are_contiguous_and_complete() -> None:
    temporal = _contract()["metrics"]["temporal_stability"]
    blocks = temporal["blocks"]

    assert temporal["block_count"] == 4
    assert temporal["equal_origin_count_per_block"] == 1_751
    assert len(blocks) == 4
    assert blocks[0]["origin_start"] == 28_032
    assert blocks[-1]["origin_stop_exclusive"] == 35_036

    total = 0

    for index, block in enumerate(blocks):
        count = (
            block["origin_stop_exclusive"]
            - block["origin_start"]
        )

        assert count == 1_751
        total += count

        if index > 0:
            assert block["origin_start"] == (
                blocks[index - 1]["origin_stop_exclusive"]
            )

    assert total == 7_004


def test_locked_test_metrics_and_reference_are_frozen() -> None:
    contract = _contract()

    assert contract["formal_reference"]["name"] == "persistence"
    assert contract["metrics"]["primary"]["name"] == "aggregate_mae"
    assert contract["metrics"]["peak_state"]["name"] == "peak_state_mae"
    assert contract["targets"]["common_origin_set_required"] is True


def test_execution_controls_prohibit_adaptation() -> None:
    controls = _contract()["execution_controls"]

    assert controls["hyperparameter_changes_allowed"] is False
    assert controls["feature_changes_allowed"] is False
    assert controls["target_changes_allowed"] is False
    assert controls["threshold_changes_allowed"] is False
    assert controls["benchmark_changes_allowed"] is False
    assert controls["temporal_block_changes_allowed"] is False
    assert controls["second_evaluation_allowed"] is False


def test_gate_outputs_are_declared_but_not_created() -> None:
    outputs = _contract()["outputs"]

    assert outputs["write_once"] is True
    assert len(outputs["predictions_required_columns"]) == 9
    assert not (ROOT / outputs["predictions_path"]).exists()
    assert not (ROOT / outputs["results_path"]).exists()
