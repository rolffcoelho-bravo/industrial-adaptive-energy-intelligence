from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from iaei.contracts import validate_report_payload
from iaei.reporting.payload import (
    HASH_ONLY_INPUTS,
    PARSED_INPUTS,
    VISUAL_PATHS,
    build_report_payload,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_payload_build_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report_payload.json"

    first = build_report_payload(output)
    first_hash = _sha256(first)

    second = build_report_payload(output)
    second_hash = _sha256(second)

    assert first_hash == second_hash
    assert validate_report_payload(second) == json.loads(
        second.read_text(encoding="utf-8")
    )


def test_payload_uses_governed_values(
    tmp_path: Path,
) -> None:
    output = build_report_payload(
        tmp_path / "report_payload.json"
    )
    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["visuals"] == VISUAL_PATHS
    assert payload["metadata"]["evidence_gate"] == "4F"
    assert payload["metadata"]["all_results_final"] is True
    assert payload["executive"]["prediction_row_count"] == 7004
    assert payload["locked_test"]["peak_state_row_count"] == 761
    assert payload["locked_test"]["temporal_block_count"] == 4
    assert (
        payload["governance"]["single_evaluation_consumed"]
        is True
    )
    assert (
        payload["governance"]["second_evaluation_allowed"]
        is False
    )
    assert (
        payload["governance"]["reestimation_performed"]
        is False
    )


def test_locked_predictions_are_hash_only_evidence() -> None:
    assert len(HASH_ONLY_INPUTS) == 1
    assert (
        HASH_ONLY_INPUTS[0].name
        == "locked_test_predictions.csv"
    )
    assert HASH_ONLY_INPUTS[0] not in PARSED_INPUTS


def test_payload_module_has_no_model_or_evaluator_imports() -> None:
    source_path = Path("src/iaei/reporting/payload.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(
                alias.name for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any(
        module.startswith("iaei.modeling")
        for module in imported_modules
    )
    assert "evaluate_locked_test_once" not in source
    assert "locked_test_harness" not in source


def test_payload_contains_no_unbounded_claims(
    tmp_path: Path,
) -> None:
    output = build_report_payload(
        tmp_path / "report_payload.json"
    )
    serialized = output.read_text(
        encoding="utf-8"
    ).lower()

    prohibited = (
        "guaranteed savings",
        "production ready",
        "caused the reduction",
        "recommended operating schedule",
        "company-specific conclusion",
    )

    assert all(
        term not in serialized
        for term in prohibited
    )
