from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from iaei.paths import CONFIGS, SCHEMAS


class ContractError(RuntimeError):
    """Raised when a publication or analytical contract is violated."""


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


def validate_report_payload(payload_path: Path) -> dict[str, Any]:
    payload = load_json(payload_path)
    schema = load_json(SCHEMAS / "report_payload.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        details = "\n".join(f"- {'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors)
        raise ContractError(f"Report payload failed validation:\n{details}")

    serialized = json.dumps(payload).lower()
    forbidden = ("populate_", "placeholder", "todo", "tbd", "dummy", "synthetic")
    hits = [term for term in forbidden if term in serialized]
    if hits:
        raise ContractError(f"Report payload contains forbidden placeholder terms: {hits}")
    return payload


def validate_repository_contracts() -> None:
    required = [
        CONFIGS / "project.yml",
        CONFIGS / "data_contract.yml",
        CONFIGS / "model_contract.yml",
        CONFIGS / "drift_policy.yml",
        CONFIGS / "report_contract.yml",
        CONFIGS / "visualization_contract.yml",
        SCHEMAS / "report_payload.schema.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ContractError(f"Missing required contracts: {missing}")
    for path in required[:-1]:
        load_yaml(path)
    load_json(required[-1])
