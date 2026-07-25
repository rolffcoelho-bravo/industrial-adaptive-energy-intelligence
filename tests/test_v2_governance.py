from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from iaei.contracts import (
    validate_gate_6a_closure_manifest,
    validate_optimization_governance,
    validate_v2_architecture_contract,
)
from iaei.paths import ROOT, SCHEMAS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _errors(schema_name: str, payload: dict[str, Any]) -> list[Any]:
    schema = _load_json(SCHEMAS / schema_name)
    return list(Draft202012Validator(schema).iter_errors(payload))


def _valid_objective_record() -> dict[str, Any]:
    return {
        "contract_version": "1.0.0",
        "objective_set_version": "v2_objectives_1.0.0",
        "trial_id": "trial-001",
        "fold_id": 1,
        "seed": 20260725,
        "partition": "validation",
        "metrics": [
            {
                "id": "aggregate_mae",
                "value": 3.8,
                "unit": "kWh",
                "direction": "minimize",
                "finite": True,
            }
        ],
        "computed_by": {
            "implementation": "iaei.metrics.aggregate_mae",
            "code_commit": "a" * 40,
            "deterministic": True,
        },
        "source_artifact_sha256": "b" * 64,
        "status": "complete",
    }


def test_gate_6a_contracts_validate() -> None:
    architecture = validate_v2_architecture_contract()
    optimization = validate_optimization_governance()
    closure = validate_gate_6a_closure_manifest()

    assert architecture["gate"] == "6A"
    assert optimization["gate"] == "6A"
    assert closure["status"] == "closed"
    assert closure["next_gate"] == "6B"


def test_v2_layer_order_is_frozen() -> None:
    architecture = validate_v2_architecture_contract()
    observed = [layer["id"] for layer in architecture["layers"]]

    assert observed == [
        "data_evidence",
        "predictive_models",
        "governed_model_optimization",
        "uncertainty",
        "robust_selection",
        "generative_interpretation",
        "human_decision",
    ]


def test_locked_test_is_excluded_from_v2() -> None:
    architecture = validate_v2_architecture_contract()
    optimization = validate_optimization_governance()

    assert architecture["v1_baseline"]["locked_test_access_permitted"] is False
    assert architecture["interfaces"]["trial_executor"][
        "admissible_partitions"
    ] == ["training", "validation"]
    assert architecture["interfaces"]["trial_executor"][
        "prohibited_partitions"
    ] == ["locked_test"]
    assert optimization["evidence_boundary"][
        "locked_test_partition_permitted"
    ] is False
    assert optimization["ensemble"]["locked_test_predictions_permitted"] is False
    assert optimization["uncertainty"][
        "locked_test_calibration_permitted"
    ] is False


def test_decision_rights_separate_execution_interpretation_and_approval() -> None:
    architecture = validate_v2_architecture_contract()
    rights = architecture["decision_rights"]

    assert "execute_trials" in rights["genai"]["prohibited"]
    assert "approve_model_promotion" in rights["genai"]["prohibited"]
    assert "calculate_objectives" in rights["deterministic_system"]["allowed"]
    assert "approve_or_reject_promotion" in rights["human"]["allowed"]
    assert architecture["interfaces"]["promotion_authority"][
        "final_authority"
    ] == "human"


def test_optimization_rules_are_prespecified() -> None:
    governance = validate_optimization_governance()

    assert governance["evidence_boundary"]["fold_strategy"] == (
        "nested_expanding_window"
    )
    assert governance["evidence_boundary"]["outer_fold_count"] == 4
    assert governance["evidence_boundary"]["inner_fold_count"] == 3
    assert governance["evidence_boundary"]["purge_intervals"] == 4
    assert governance["randomness"]["stochastic_model_seeds"] == [
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
    ]
    assert governance["promotion"]["selection_method"] == (
        "constrained_pareto_then_human_approval"
    )
    assert governance["promotion"]["automatic_promotion_permitted"] is False


def test_all_gate_6a_schemas_are_valid_json_schemas() -> None:
    schema_names = [
        "v2_architecture_contract.schema.json",
        "optimization_governance.schema.json",
        "governed_search_space.schema.json",
        "objective_record.schema.json",
        "trial_evidence.schema.json",
        "promotion_decision.schema.json",
        "gate_6a_closure_manifest.schema.json",
    ]

    for schema_name in schema_names:
        Draft202012Validator.check_schema(_load_json(SCHEMAS / schema_name))


def test_governed_search_space_accepts_only_training_and_validation() -> None:
    payload = {
        "contract_version": "1.0.0",
        "gate": "6B",
        "search_space_id": "advanced-tabular-1",
        "candidate_family": "advanced_tabular",
        "algorithm_id": "tabular_challenger",
        "data_boundary": {
            "admissible_partitions": ["training", "validation"],
            "locked_test_access_permitted": False,
            "fold_strategy": "nested_expanding_window",
            "outer_fold_count": 4,
            "inner_fold_count": 3,
            "purge_intervals": 4,
        },
        "parameter_space": [
            {
                "name": "max_depth",
                "type": "integer",
                "required": True,
                "minimum": 2,
                "maximum": 12,
                "step": 1,
                "log_scale": False,
            }
        ],
        "budget": {
            "max_unique_trials": 24,
            "max_parallel_trials": 4,
            "max_retries_per_trial": 1,
            "max_wall_clock_minutes": 180,
        },
        "seeds": [20260725],
        "objectives": ["aggregate_mae", "peak_state_mae"],
        "hard_constraints": [
            "chronology_violation_count",
            "locked_test_access_count",
        ],
        "provenance": {
            "optimization_contract_version": "1.0.0",
            "code_commit": "a" * 40,
            "created_by_role": "human",
            "human_approved": True,
            "approved_before_execution": True,
        },
    }

    assert _errors("governed_search_space.schema.json", payload) == []

    invalid = deepcopy(payload)
    invalid["data_boundary"]["admissible_partitions"] = [
        "training",
        "validation",
        "locked_test",
    ]
    invalid["data_boundary"]["locked_test_access_permitted"] = True

    assert _errors("governed_search_space.schema.json", invalid)


def test_objective_trial_and_promotion_schemas_accept_governed_records() -> None:
    objective = _valid_objective_record()
    assert _errors("objective_record.schema.json", objective) == []

    trial = {
        "contract_version": "1.0.0",
        "trial_id": "trial-001",
        "search_space_id": "advanced-tabular-1",
        "search_space_sha256": "c" * 64,
        "candidate_family": "advanced_tabular",
        "algorithm_id": "tabular_challenger",
        "parameter_values": {"max_depth": 6},
        "fold_ids": [1],
        "seeds": [20260725],
        "objective_records": [objective],
        "hard_constraint_results": [
            {
                "id": "locked_test_access_count",
                "observed_value": 0,
                "operator": "equal",
                "threshold": 0,
                "passed": True,
            }
        ],
        "resource_usage": {
            "wall_clock_seconds": 10.0,
            "peak_memory_mb": 256.0,
            "model_size_bytes": 4096,
            "p95_inference_latency_ms_per_1000_rows": 2.5,
        },
        "environment": {
            "python_version": "3.12",
            "platform": "portable",
            "dependency_lock_sha256": "d" * 64,
            "execution_adapter": "portable",
        },
        "code_commit": "a" * 40,
        "outcome": "success",
        "artifact_sha256": "e" * 64,
    }
    assert _errors("trial_evidence.schema.json", trial) == []

    promotion = {
        "contract_version": "1.0.0",
        "decision_id": "promotion-001",
        "gate": "6B",
        "candidate_id": "tabular-challenger-001",
        "reference_ids": ["persistence", "v1_frozen_champion"],
        "evidence_manifest_sha256": "f" * 64,
        "hard_constraints_all_pass": True,
        "pareto_status": "eligible",
        "promotion_requirements": [
            {
                "id": "minimum_mean_validation_mae_relative_improvement",
                "passed": True,
                "observed": 0.02,
                "required_value": 0.01,
            }
        ],
        "decision": "promote",
        "decided_by": {
            "authority": "human",
            "role": "head_of_research",
            "human_approval_recorded": True,
        },
        "genai_role": {
            "advisory_only": True,
            "vote_permitted": False,
            "authoritative_metric_calculation_permitted": False,
        },
        "rationale": (
            "The candidate passes the governed constraints and remains "
            "Pareto eligible on validation evidence."
        ),
        "evidence_commit": "a" * 40,
        "next_action": "freeze_candidate",
    }
    assert _errors("promotion_decision.schema.json", promotion) == []

    invalid_promotion = deepcopy(promotion)
    invalid_promotion["genai_role"]["vote_permitted"] = True
    assert _errors("promotion_decision.schema.json", invalid_promotion)


def test_v1_governed_hashes_remain_unchanged() -> None:
    release = _load_json(ROOT / "outputs" / "v1_release_manifest.json")
    evidence = release["governed_evidence"]

    for key in ("report_payload", "latex_source"):
        item = evidence[key]
        assert _sha256(ROOT / item["path"]) == item["sha256"]

    for group in ("figures", "tables"):
        for path_value, expected in evidence[group].items():
            assert _sha256(ROOT / path_value) == expected

    locked_test = evidence["locked_test"]
    assert _sha256(
        ROOT / "outputs" / "modeling" / "locked_test_predictions.csv"
    ) == locked_test["prediction_sha256"]
    assert _sha256(
        ROOT / "outputs" / "modeling" / "locked_test_results.json"
    ) == locked_test["results_sha256"]


def test_gate_6a_is_design_only() -> None:
    architecture = validate_v2_architecture_contract()
    closure = validate_gate_6a_closure_manifest()

    boundary = architecture["gate_boundary"]
    controls = closure["controls"]

    assert boundary["design_only"] is True
    assert boundary["model_fitting_performed"] is False
    assert boundary["confirmatory_evaluation_performed"] is False
    assert boundary["locked_test_accessed"] is False
    assert controls["v1_artifact_changed"] is False
    assert controls["optimization_impact_claim_made"] is False
