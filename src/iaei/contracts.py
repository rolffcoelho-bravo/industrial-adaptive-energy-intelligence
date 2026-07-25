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
            f"- {'/'.join(map(str, error.path)) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise ContractError(f"{label} failed validation:\n{details}")


def validate_target_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "target_contract.yml")
    schema = load_json(SCHEMAS / "target_contract.schema.json")

    _validate_payload(
        contract,
        schema,
        label="Target and leakage contract",
    )

    return contract


def validate_silver_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "silver_contract.yml")
    schema = load_json(SCHEMAS / "silver_contract.schema.json")

    _validate_payload(
        contract,
        schema,
        label="Silver analytical-layer contract",
    )

    return contract


def validate_report_payload(payload_path: Path) -> dict[str, Any]:
    payload = load_json(payload_path)
    schema = load_json(SCHEMAS / "report_payload.schema.json")

    _validate_payload(
        payload,
        schema,
        label="Report payload",
    )

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
            "Report payload contains forbidden placeholder terms: "
            f"{hits}"
        )

    return payload


def validate_reporting_closure_manifest() -> dict[str, Any]:
    manifest = load_json(
        ROOT / "outputs" / "reporting_closure_manifest.json"
    )
    schema = load_json(
        SCHEMAS / "reporting_closure_manifest.schema.json"
    )

    _validate_payload(
        manifest,
        schema,
        label="Gate 5D4 reporting closure manifest",
    )

    return manifest


def validate_v1_release_manifest() -> dict[str, Any]:
    manifest = load_json(ROOT / "outputs" / "v1_release_manifest.json")
    schema = load_json(SCHEMAS / "v1_release_manifest.schema.json")

    _validate_payload(
        manifest,
        schema,
        label="Gate 5E V1 release manifest",
    )

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
