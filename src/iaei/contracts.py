from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from iaei.paths import CONFIGS, ROOT, SCHEMAS


class ContractError(RuntimeError):
    """Raised when an analytical or reporting contract is violated."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected a mapping in {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"Expected an object in {path}")
    return value


def _validate_payload(
    payload: dict[str, Any],
    schema: dict[str, Any],
    *,
    label: str,
) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ContractError(f"{label} failed validation:\n{details}")


def validate_target_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "target_contract.yml")
    schema = load_json(SCHEMAS / "target_contract.schema.json")
    _validate_payload(contract, schema, label="Target and leakage contract")
    return contract


def validate_silver_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "silver_contract.yml")
    schema = load_json(SCHEMAS / "silver_contract.schema.json")
    _validate_payload(contract, schema, label="Silver analytical-layer contract")
    return contract


def validate_report_payload(payload_path: Path) -> dict[str, Any]:
    payload = load_json(payload_path)
    schema = load_json(SCHEMAS / "report_payload.schema.json")
    _validate_payload(payload, schema, label="Report payload")

    serialized = json.dumps(payload).lower()
    forbidden = (
        "populate_",
        "placeholder",
        "todo",
        "tbd",
        "dummy",
        "synthetic",
    )
    hits = [term for term in forbidden if term in serialized]
    if hits:
        raise ContractError(
            "Report payload contains forbidden placeholder terms: " f"{hits}"
        )
    return payload


def validate_reporting_closure_manifest() -> dict[str, Any]:
    manifest = load_json(ROOT / "outputs" / "reporting_closure_manifest.json")
    schema = load_json(SCHEMAS / "reporting_closure_manifest.schema.json")
    _validate_payload(
        manifest,
        schema,
        label="Gate 5D4 reporting closure manifest",
    )
    return manifest


def validate_v1_release_manifest() -> dict[str, Any]:
    manifest = load_json(ROOT / "outputs" / "v1_release_manifest.json")
    schema = load_json(SCHEMAS / "v1_release_manifest.schema.json")
    _validate_payload(manifest, schema, label="Gate 5E V1 release manifest")
    return manifest


def validate_v2_architecture_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "v2_architecture_contract.yml")
    schema = load_json(SCHEMAS / "v2_architecture_contract.schema.json")
    _validate_payload(
        contract,
        schema,
        label="Gate 6A V2 architecture contract",
    )
    return contract


def validate_optimization_governance() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "optimization_governance.yml")
    schema = load_json(SCHEMAS / "optimization_governance.schema.json")
    _validate_payload(
        contract,
        schema,
        label="Gate 6A optimization governance contract",
    )
    return contract


def validate_gate_6a_closure_manifest() -> dict[str, Any]:
    manifest = load_json(
        ROOT / "outputs" / "v2" / "gate_6a_architecture_manifest.json"
    )
    schema = load_json(SCHEMAS / "gate_6a_closure_manifest.schema.json")
    _validate_payload(
        manifest,
        schema,
        label="Gate 6A architecture closure manifest",
    )
    return manifest


def validate_advanced_tabular_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "advanced_tabular_contract.yml")
    schema = load_json(SCHEMAS / "advanced_tabular_contract.schema.json")
    _validate_payload(
        contract,
        schema,
        label="Gate 6B advanced tabular contract",
    )

    search_schema = load_json(SCHEMAS / "governed_search_space.schema.json")
    observed_algorithms: set[str] = set()
    configuration_count = 0
    for family in contract["candidate_families"]:
        path = ROOT / str(family["search_space_path"])
        search_space = load_json(path)
        _validate_payload(
            search_space,
            search_schema,
            label=f"Gate 6B search space {path.name}",
        )
        algorithm_id = str(family["algorithm_id"])
        if search_space["algorithm_id"] != algorithm_id:
            raise ContractError(
                f"Gate 6B search-space algorithm mismatch for {algorithm_id}"
            )
        if algorithm_id in observed_algorithms:
            raise ContractError(
                f"Duplicate Gate 6B candidate algorithm: {algorithm_id}"
            )
        observed_algorithms.add(algorithm_id)
        configuration_count += len(family["configurations"])

    search = contract["search"]
    if configuration_count != int(search["unique_configuration_count"]):
        raise ContractError("Gate 6B configuration count is inconsistent")

    governance = validate_optimization_governance()
    budget = governance["search_budgets"]["advanced_tabular"]
    if configuration_count > int(budget["max_unique_trials"]):
        raise ContractError("Gate 6B configuration count exceeds Gate 6A")
    if int(search["max_parallel_trials"]) > int(budget["max_parallel_trials"]):
        raise ContractError("Gate 6B parallelism exceeds Gate 6A")
    if int(search["max_wall_clock_minutes"]) > int(
        budget["max_wall_clock_minutes"]
    ):
        raise ContractError("Gate 6B wall-clock budget exceeds Gate 6A")
    return contract


def validate_neural_forecasting_contract() -> dict[str, Any]:
    """Validate the Gate 6C proposal without resolving its seed decision."""

    contract = load_yaml(CONFIGS / "neural_forecasting_contract.yml")
    schema = load_json(SCHEMAS / "neural_forecasting_contract.schema.json")
    _validate_payload(
        contract,
        schema,
        label="Gate 6C neural forecasting contract",
    )

    architecture = validate_v2_architecture_contract()
    optimization = validate_optimization_governance()
    if architecture["v1_baseline"]["locked_test_access_permitted"] is not False:
        raise ContractError("Gate 6C cannot weaken the V1 locked-test boundary")
    if optimization["evidence_boundary"][
        "locked_test_partition_permitted"
    ] is not False:
        raise ContractError("Gate 6C cannot admit the locked-test partition")
    if contract["promotion"]["automatic_promotion_permitted"] is not False:
        raise ContractError("Gate 6C cannot permit automatic promotion")

    search = contract["search"]
    neural_budget = optimization["search_budgets"]["neural_forecasting"]
    if int(search["unique_configuration_count"]) > int(
        neural_budget["max_unique_configurations"]
    ):
        raise ContractError("Gate 6C configuration count exceeds Gate 6A")
    if int(search["max_parallel_trials"]) > int(
        neural_budget["max_parallel_trials"]
    ):
        raise ContractError("Gate 6C parallelism exceeds Gate 6A")
    if int(search["max_wall_clock_minutes"]) > int(
        neural_budget["max_wall_clock_minutes"]
    ):
        raise ContractError("Gate 6C wall-clock budget exceeds Gate 6A")
    return contract


def validate_neural_seed_governance_alignment(
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require Gate 6C seeds to match the frozen Gate 6A parent contract."""

    observed = contract or validate_neural_forecasting_contract()
    optimization = validate_optimization_governance()
    search = observed["search"]
    neural_budget = optimization["search_budgets"]["neural_forecasting"]
    parent_seeds = optimization["randomness"]["stochastic_model_seeds"]
    parent_minimum = int(optimization["randomness"]["minimum_stochastic_seed_count"])
    parent_seed_budget = int(neural_budget["seeds_per_configuration"])

    if search["seeds"] != parent_seeds:
        raise ContractError(
            "Gate 6C seed identities conflict with frozen Gate 6A governance"
        )
    if int(search["seed_count"]) < parent_minimum:
        raise ContractError(
            "Gate 6C seed count is below the frozen Gate 6A minimum"
        )
    if int(search["seed_count"]) != parent_seed_budget:
        raise ContractError(
            "Gate 6C seed count conflicts with the Gate 6A neural budget"
        )
    return observed


def validate_gate_6c1_closure_manifest() -> dict[str, Any]:
    manifest = load_json(
        ROOT / "outputs" / "v2" / "gate_6c1_closure_manifest.json"
    )
    schema = load_json(SCHEMAS / "gate_6c1_closure_manifest.schema.json")
    _validate_payload(manifest, schema, label="Gate 6C1 closure manifest")
    return manifest


def validate_repository_contracts() -> None:
    required_yaml = [
        CONFIGS / "project.yml",
        CONFIGS / "data_contract.yml",
        CONFIGS / "model_contract.yml",
        CONFIGS / "target_contract.yml",
        CONFIGS / "silver_contract.yml",
        CONFIGS / "drift_policy.yml",
        CONFIGS / "report_contract.yml",
        CONFIGS / "visualization_contract.yml",
        CONFIGS / "v2_architecture_contract.yml",
        CONFIGS / "optimization_governance.yml",
        CONFIGS / "advanced_tabular_contract.yml",
        CONFIGS / "neural_forecasting_contract.yml",
    ]
    required_json = [
        SCHEMAS / "report_payload.schema.json",
        SCHEMAS / "reporting_closure_manifest.schema.json",
        SCHEMAS / "v1_release_manifest.schema.json",
        SCHEMAS / "target_contract.schema.json",
        SCHEMAS / "silver_contract.schema.json",
        SCHEMAS / "v2_architecture_contract.schema.json",
        SCHEMAS / "optimization_governance.schema.json",
        SCHEMAS / "governed_search_space.schema.json",
        SCHEMAS / "objective_record.schema.json",
        SCHEMAS / "trial_evidence.schema.json",
        SCHEMAS / "promotion_decision.schema.json",
        SCHEMAS / "gate_6a_closure_manifest.schema.json",
        SCHEMAS / "advanced_tabular_contract.schema.json",
        SCHEMAS / "neural_forecasting_contract.schema.json",
        SCHEMAS / "neural_seed_evidence.schema.json",
        SCHEMAS / "neural_candidate_evidence.schema.json",
        SCHEMAS / "neural_promotion_decision.schema.json",
        SCHEMAS / "gate_6c1_closure_manifest.schema.json",
    ]
    required = [*required_yaml, *required_json]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ContractError(f"Missing required contracts: {missing}")

    for path in required_yaml:
        load_yaml(path)
    for path in required_json:
        load_json(path)

    validate_target_contract()
    validate_silver_contract()
    validate_reporting_closure_manifest()
    validate_v1_release_manifest()
    validate_v2_architecture_contract()
    validate_optimization_governance()
    validate_gate_6a_closure_manifest()
    validate_advanced_tabular_contract()
    validate_neural_forecasting_contract()
    validate_gate_6c1_closure_manifest()
