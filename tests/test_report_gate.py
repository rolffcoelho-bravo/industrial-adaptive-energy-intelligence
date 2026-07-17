from __future__ import annotations

import json
from pathlib import Path

import pytest

from iaei.contracts import (
    ContractError,
    load_yaml,
    validate_report_payload,
)
from iaei.reporting.builder import PAGE_TITLES


EXPECTED_SECTIONS = [
    "metadata",
    "visuals",
    "executive",
    "data",
    "validation",
    "locked_test",
    "governance",
]


def test_template_cannot_be_published() -> None:
    template = Path("examples/report_payload.template.json")

    with pytest.raises(ContractError):
        validate_report_payload(template)


def test_report_schema_requires_evidence_aligned_sections() -> None:
    schema = json.loads(
        Path("schemas/report_payload.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["required"] == EXPECTED_SECTIONS
    assert set(schema["properties"]) == set(EXPECTED_SECTIONS)
    assert "drift" not in schema["properties"]
    assert "optimization" not in schema["properties"]
    assert "agents" not in schema["properties"]
    assert "impact" not in schema["properties"]


def test_report_contract_uses_terminal_evidence_only() -> None:
    report = load_yaml(
        Path("configs/report_contract.yml")
    )["report"]

    assert report["exact_pages"] == 5
    assert (
        report["locked_metrics_must_be_read_from_terminal_artifacts"]
        is True
    )
    assert report["second_locked_test_evaluation_prohibited"] is True
    assert report["required_charts"] == [
        "confirmatory_forecasting_verdict.png",
        "governed_data_architecture.png",
        "model_ladder_chronological_validation.png",
        "locked_test_temporal_stability.png",
        "evidence_governance_model_boundaries.png",
    ]
    assert len(report["required_tables"]) == 5
    assert "business_savings_claim" in report["prohibited_claims"]


def test_builder_page_titles_are_evidence_aligned() -> None:
    assert PAGE_TITLES == (
        "1. Confirmatory forecasting verdict",
        "2. Governed data and analytical architecture",
        "3. Model ladder and chronological validation",
        "4. Locked-test stability and peak-state robustness",
        "5. Evidence governance and model boundaries",
    )
