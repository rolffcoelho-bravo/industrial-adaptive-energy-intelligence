from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from iaei.contracts import (
    ContractError,
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
)
from iaei.modeling.neural_governance import (
    APPROVED_ALGORITHMS,
    APPROVED_SEEDS,
    Gate6C1ExecutionProhibited,
    assert_no_gate_6c_execution_artifacts,
    build_gate_6c1_plan,
    causal_window,
    deterministic_cpu_environment,
    prohibit_gate_6c1_execution,
)
from iaei.paths import SCHEMAS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _errors(schema_name: str, payload: dict[str, Any]) -> list[Any]:
    schema = _load_json(SCHEMAS / schema_name)
    return list(Draft202012Validator(schema).iter_errors(payload))


def test_gate_6c1_contract_and_plan_are_frozen() -> None:
    contract = validate_neural_forecasting_contract()
    plan = build_gate_6c1_plan()

    assert contract["contract_version"] == "1.1.0"
    assert contract["status"] == "approved_for_implementation"
    assert tuple(candidate.algorithm_id for candidate in plan.candidates) == APPROVED_ALGORITHMS
    assert plan.seeds == APPROVED_SEEDS
    assert len(plan.seeds) == 5
    assert plan.outer_fold_count == 4
    assert plan.purge_intervals == 4
    assert plan.maximum_prediction_origin_exclusive == 28028
    assert plan.maximum_target_dependency_exclusive == 28032
    assert plan.canonical_device == "cpu"
    assert plan.fitting_permitted is False


def test_gate_6c1_contract_preserves_human_decision_and_v1_boundary() -> None:
    contract = validate_neural_forecasting_contract()

    assert contract["promotion"]["automatic_promotion_permitted"] is False
    assert contract["promotion"]["final_authority"] == "human"
    assert contract["v1_boundary"]["locked_test_access_permitted"] is False
    assert contract["v1_boundary"]["locked_prediction_parsing_permitted"] is False
    assert contract["v1_boundary"]["confirmatory_evaluation_permitted"] is False
    assert contract["data_boundary"]["admissible_partitions"] == [
        "training",
        "validation",
    ]


def test_deterministic_cpu_environment_is_seed_bounded() -> None:
    for seed in APPROVED_SEEDS:
        environment = deterministic_cpu_environment(seed)
        assert environment["PYTHONHASHSEED"] == str(seed)
        assert environment["IAEI_GATE_6C_SEED"] == str(seed)
        assert environment["IAEI_CANONICAL_DEVICE"] == "cpu"
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["MKL_NUM_THREADS"] == "1"
        assert environment["OPENBLAS_NUM_THREADS"] == "1"
        assert environment["NUMEXPR_NUM_THREADS"] == "1"

    with pytest.raises(ContractError):
        deterministic_cpu_environment(7)


def test_causal_window_respects_context_and_locked_boundary() -> None:
    window = causal_window(
        prediction_origin=28027,
        context_length=96,
        horizon=1,
    )
    assert window.context_start == 27932
    assert window.context_end_exclusive == 28028
    assert window.prediction_origin == 28027
    assert window.target_index == 28028

    with pytest.raises(ContractError):
        causal_window(prediction_origin=28028, context_length=96)
    with pytest.raises(ContractError):
        causal_window(prediction_origin=40, context_length=96)
    with pytest.raises(ContractError):
        causal_window(prediction_origin=100, context_length=96, horizon=2)


def test_gate_6c1_fails_closed_on_model_execution_actions() -> None:
    for action in ("fit", "train", "predict", "evaluate", "score", "optimize", "search"):
        with pytest.raises(Gate6C1ExecutionProhibited):
            prohibit_gate_6c1_execution(action)

    prohibit_gate_6c1_execution("validate_contract")


def test_gate_6c1_execution_artifact_guard(tmp_path: Path) -> None:
    assert_no_gate_6c_execution_artifacts(tmp_path)

    output_directory = tmp_path / "outputs" / "v2" / "gate_6c"
    output_directory.mkdir(parents=True)
    (output_directory / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(Gate6C1ExecutionProhibited):
        assert_no_gate_6c_execution_artifacts(tmp_path)


def test_gate_6c1_schemas_are_valid_json_schemas() -> None:
    schema_names = (
        "neural_forecasting_contract.schema.json",
        "neural_seed_evidence.schema.json",
        "neural_candidate_evidence.schema.json",
        "neural_promotion_decision.schema.json",
        "gate_6c1_closure_manifest.schema.json",
    )
    for schema_name in schema_names:
        Draft202012Validator.check_schema(_load_json(SCHEMAS / schema_name))


def test_future_seed_evidence_schema_enforces_cpu_and_boundaries() -> None:
    payload = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "nhits_compact",
        "seed": 20260721,
        "fold_id": 1,
        "partition": "validation",
        "prediction_origin_max": 28027,
        "target_dependency_max": 28028,
        "metrics": {"mae": 3.8, "peak_mae": 13.9},
        "resources": {
            "wall_clock_seconds": 10.0,
            "peak_memory_mb": 512.0,
            "model_size_bytes": 4096,
            "p95_inference_latency_ms_per_1000_rows": 2.0,
        },
        "portability": {
            "device": "cpu",
            "gpu_required": False,
            "cpu_portability_passed": True,
        },
        "status": "success",
    }
    assert _errors("neural_seed_evidence.schema.json", payload) == []

    invalid = dict(payload)
    invalid["seed"] = 20260726
    assert _errors("neural_seed_evidence.schema.json", invalid)

    invalid = dict(payload)
    invalid["prediction_origin_max"] = 28028
    assert _errors("neural_seed_evidence.schema.json", invalid)


def test_future_candidate_and_decision_schemas_preserve_governance() -> None:
    candidate = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "tide_compact",
        "seed_count": 5,
        "outer_fold_count": 4,
        "validation_origin_count_per_seed": 7004,
        "aggregate": {
            "mean_mae": 3.8,
            "mean_peak_mae": 14.0,
            "relative_mae_improvement_vs_v1": 0.02,
            "positive_outer_folds": 3,
        },
        "stability": {
            "across_seed_mae_standard_deviation": 0.02,
            "across_seed_peak_mae_standard_deviation": 0.04,
            "outer_fold_mae_dispersion": 0.1,
        },
        "constraints": {
            "chronology_passed": True,
            "locked_test_excluded": True,
            "v1_immutable": True,
            "resource_limits_passed": True,
            "cpu_portability_passed": True,
        },
        "promotion_eligible": True,
    }
    assert _errors("neural_candidate_evidence.schema.json", candidate) == []

    decision = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "tide_compact",
        "decision": "defer",
        "human_authority": True,
        "automatic_promotion_permitted": False,
        "locked_test_accessed": False,
        "confirmatory_evaluation_performed": False,
        "rationale": "Additional validation evidence is required before any promotion decision.",
        "next_action": "request_additional_validation",
    }
    assert _errors("neural_promotion_decision.schema.json", decision) == []

    invalid = dict(decision)
    invalid["automatic_promotion_permitted"] = True
    assert _errors("neural_promotion_decision.schema.json", invalid)


def test_gate_6c1_closure_manifest_is_design_only() -> None:
    closure = validate_gate_6c1_closure_manifest()
    assert closure["status"] in {"implementation_complete_pending_ci", "closed"}
    assert closure["implementation_only"] is True
    assert closure["candidate_ids"] == list(APPROVED_ALGORITHMS)
    assert closure["seeds"] == list(APPROVED_SEEDS)
    assert closure["controls"]["model_fitting_performed"] is False
    assert closure["controls"]["validation_predictions_generated"] is False
    assert closure["controls"]["validation_metrics_calculated"] is False
    assert closure["controls"]["locked_test_accessed"] is False
    assert closure["controls"]["confirmatory_evaluation_performed"] is False
    assert closure["controls"]["v1_immutable"] is True
    assert closure["next_gate"] == "6C2"
    assert closure["blocked_gates"] == ["6C3", "6D"]
