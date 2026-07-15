from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from iaei.paths import CONFIGS, SCHEMAS


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


def validate_report_payload(payload_path: Path) -> dict[str, Any]:
    payload = load_json(payload_path)
    schema = load_json(SCHEMAS / "report_payload.schema.json")
    _validate_payload(payload, schema, label="Report payload")

    serialized = json.dumps(payload).lower()
    forbidden = ("populate_", "placeholder", "todo", "tbd", "dummy", "synthetic")
    hits = [term for term in forbidden if term in serialized]
    if hits:
        raise ContractError(f"Report payload contains forbidden placeholder terms: {hits}")
    return payload


def validate_repository_contracts() -> None:
    required_yaml = [
        CONFIGS / "project.yml",
        CONFIGS / "data_contract.yml",
        CONFIGS / "model_contract.yml",
        CONFIGS / "target_contract.yml",
        CONFIGS / "drift_policy.yml",
        CONFIGS / "report_contract.yml",
        CONFIGS / "visualization_contract.yml",
    ]
    required_json = [
        SCHEMAS / "report_payload.schema.json",
        SCHEMAS / "target_contract.schema.json",
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
