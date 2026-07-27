from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from iaei.contracts import (
    ContractError,
    validate_foundation_model_contract,
    validate_gate_6d1_closure_manifest,
)
from iaei.modeling.foundation_governance import (
    APPROVED_FOUNDATION_MODELS,
    RESEARCH_ONLY_MODELS,
    Gate6D1ExecutionProhibited,
    assert_no_gate_6d_execution_artifacts,
    build_gate_6d1_plan,
    causal_foundation_window,
    commercial_promotion_eligible,
    prohibit_gate_6d1_execution,
)
from iaei.paths import SCHEMAS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6d1_contract_and_plan_are_frozen() -> None:
    contract = validate_foundation_model_contract()
    plan = build_gate_6d1_plan()

    assert contract["contract_version"] == "1.0.0"
    assert contract["gate"] == "6D"
    assert contract["subgate"] == "6D1"
    assert tuple(candidate.candidate_id for candidate in plan.candidates) == (
        APPROVED_FOUNDATION_MODELS
    )
    assert plan.context_length_intervals == 672
    assert plan.horizon_intervals == 1
    assert plan.outer_fold_count == 4
    assert plan.purge_intervals == 4
    assert plan.validation_origin_count == 7004
    assert plan.maximum_prediction_origin_exclusive == 28028
    assert plan.maximum_target_dependency_exclusive == 28032
    assert plan.canonical_device == "cpu"
    assert plan.model_download_permitted is False
    assert plan.inference_permitted is False
    assert plan.fine_tuning_permitted is False
    assert plan.locked_test_access_permitted is False
    assert plan.final_authority == "human"


def test_gate_6d1_model_identity_and_weight_hashes_are_frozen() -> None:
    contract = validate_foundation_model_contract()
    observed = {item["candidate_id"]: item for item in contract["candidate_models"]}

    assert observed["chronos_2_zero_shot"]["model_id"] == "amazon/chronos-2"
    assert observed["chronos_2_zero_shot"]["weight_sha256"] == (
        "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
    )
    assert observed["timesfm_2_5_zero_shot"]["model_id"] == (
        "google/timesfm-2.5-200m-pytorch"
    )
    assert observed["timesfm_2_5_zero_shot"]["weight_sha256"] == (
        "2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8"
    )
    assert observed["moirai_2_research_zero_shot"]["model_id"] == (
        "Salesforce/moirai-2.0-R-small"
    )
    assert observed["moirai_2_research_zero_shot"]["weight_sha256"] == (
        "fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669"
    )


def test_gate_6d1_license_boundary_blocks_noncommercial_promotion() -> None:
    contract = validate_foundation_model_contract()
    observed = {item["candidate_id"]: item for item in contract["candidate_models"]}

    assert RESEARCH_ONLY_MODELS == ("moirai_2_research_zero_shot",)
    moirai = observed["moirai_2_research_zero_shot"]
    assert moirai["weights_license"] == "CC-BY-NC-4.0"
    assert moirai["benchmark_admissible"] is True
    assert moirai["commercial_use_eligible"] is False
    assert moirai["promotion_eligible"] is False
    assert commercial_promotion_eligible("moirai_2_research_zero_shot") is False
    assert commercial_promotion_eligible("chronos_2_zero_shot") is True
    assert commercial_promotion_eligible("timesfm_2_5_zero_shot") is True

    with pytest.raises(ContractError):
        commercial_promotion_eligible("unknown_model")


def test_causal_foundation_window_respects_context_and_boundaries() -> None:
    window = causal_foundation_window(prediction_origin=28027)
    assert window.context_start == 27356
    assert window.context_end_exclusive == 28028
    assert window.prediction_origin == 28027
    assert window.target_index == 28028

    with pytest.raises(ContractError):
        causal_foundation_window(prediction_origin=28028)
    with pytest.raises(ContractError):
        causal_foundation_window(prediction_origin=500)
    with pytest.raises(ContractError):
        causal_foundation_window(prediction_origin=1000, context_length=96)
    with pytest.raises(ContractError):
        causal_foundation_window(prediction_origin=1000, horizon=2)


def test_gate_6d1_fails_closed_on_execution_actions() -> None:
    for action in (
        "download",
        "load_weights",
        "infer",
        "forecast",
        "fit",
        "fine_tune",
        "calibrate",
        "evaluate",
        "score",
        "promote",
    ):
        with pytest.raises(Gate6D1ExecutionProhibited):
            prohibit_gate_6d1_execution(action)

    prohibit_gate_6d1_execution("validate_contract")


def test_gate_6d1_execution_artifact_guard(tmp_path: Path) -> None:
    assert_no_gate_6d_execution_artifacts(tmp_path)

    output_directory = tmp_path / "outputs" / "v2" / "gate_6d"
    output_directory.mkdir(parents=True)
    (output_directory / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(Gate6D1ExecutionProhibited):
        assert_no_gate_6d_execution_artifacts(tmp_path)


def test_gate_6d1_schemas_are_valid_json_schemas() -> None:
    for schema_name in (
        "foundation_model_contract.schema.json",
        "gate_6d1_closure_manifest.schema.json",
    ):
        Draft202012Validator.check_schema(_load_json(SCHEMAS / schema_name))


def test_gate_6d1_closure_is_implementation_only() -> None:
    closure = validate_gate_6d1_closure_manifest()

    assert closure["status"] == "closed"
    assert closure["implementation_only"] is True
    assert closure["candidate_ids"] == list(APPROVED_FOUNDATION_MODELS)
    assert closure["research_only_candidate_ids"] == list(RESEARCH_ONLY_MODELS)
    assert closure["controls"]["model_weights_downloaded"] is False
    assert closure["controls"]["model_inference_performed"] is False
    assert closure["controls"]["validation_predictions_generated"] is False
    assert closure["controls"]["validation_metrics_calculated"] is False
    assert closure["controls"]["fine_tuning_performed"] is False
    assert closure["controls"]["locked_test_accessed"] is False
    assert closure["controls"]["confirmatory_evaluation_performed"] is False
    assert closure["controls"]["automatic_promotion_permitted"] is False
    assert closure["controls"]["v1_immutable"] is True
    assert closure["controls"]["gate_6c_closed"] is True
    assert closure["next_gate"] == "6D2"
    assert closure["next_gate_authorized"] is False
    assert closure["blocked_gates"] == ["6D2", "6D3", "6E"]
