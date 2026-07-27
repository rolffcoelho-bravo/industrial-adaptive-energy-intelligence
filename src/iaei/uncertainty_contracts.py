from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from iaei.paths import CONFIGS, ROOT, SCHEMAS


class UncertaintyContractError(RuntimeError):
    """Raised when the Gate 6E uncertainty contract is invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UncertaintyContractError(f"Expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UncertaintyContractError(f"Expected an object in {path}")
    return value


def _validate(payload: dict[str, Any], schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "\n".join(
            f"- {'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise UncertaintyContractError(f"{label} failed validation:\n{details}")


def validate_uncertainty_contract() -> dict[str, Any]:
    contract = _load_yaml(CONFIGS / "uncertainty_contract.yml")
    schema = _load_json(SCHEMAS / "uncertainty_contract.schema.json")
    _validate(contract, schema, "Gate 6E1 uncertainty contract")
    return contract


def validate_gate_6e1_closure_manifest() -> dict[str, Any]:
    manifest = _load_json(ROOT / "outputs" / "v2" / "gate_6e1_closure_manifest.json")
    schema = _load_json(SCHEMAS / "gate_6e1_closure_manifest.schema.json")
    _validate(manifest, schema, "Gate 6E1 closure manifest")
    return manifest
