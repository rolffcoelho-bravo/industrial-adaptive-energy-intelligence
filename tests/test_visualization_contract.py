from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.visualization import plot_confirmatory_forecasting_verdict


EXPECTED_BRIEF_FIGURES = {
    "page_1": "confirmatory_forecasting_verdict.png",
    "page_2": "governed_data_architecture.png",
    "page_3": "model_ladder_chronological_validation.png",
    "page_4": "locked_test_temporal_stability.png",
    "page_5": "evidence_governance_model_boundaries.png",
}

EXPECTED_INPUTS = {
    "outputs/reporting_evidence_manifest.json",
    "outputs/tables/confirmatory_metrics.csv",
    "outputs/tables/data_quality_summary.csv",
    "outputs/tables/model_ladder_summary.csv",
    "outputs/tables/temporal_block_results.csv",
    "outputs/tables/evidence_lineage.csv",
}


def test_visualization_contract_defines_evidence_aligned_figures() -> None:
    contract = load_yaml(
        Path("configs/visualization_contract.yml")
    )["visualization"]
    figures = contract["brief_figures"]

    assert contract["engine"] == "matplotlib"
    assert contract["backend"] == "Agg"
    assert contract["background"] == "light"
    assert contract["dpi"] == 300
    assert contract["deterministic_png_metadata"] is True
    assert contract["deterministic_scope"] == "same_environment"
    assert set(contract["permitted_inputs"]) == EXPECTED_INPUTS
    assert contract["footer_fields"] == [
        "figure_id",
        "source",
        "sample",
        "evidence_id",
    ]
    assert len(figures) == 5
    assert {
        page: record["filename"]
        for page, record in figures.items()
    } == EXPECTED_BRIEF_FIGURES


def test_visualization_contract_requires_visual_approval() -> None:
    contract = load_yaml(
        Path("configs/visualization_contract.yml")
    )["visualization"]
    gate = contract["publication_gate"]

    assert gate["required_brief_figures"] == 5
    assert gate["minimum_width_pixels"] == 2400
    assert gate["minimum_height_pixels"] == 1200
    assert gate["minimum_file_size_bytes"] == 30000
    assert gate["terminal_metric_reconciliation_required"] is True
    assert gate["evidence_identity_required"] is True
    assert gate["visual_inspection_required"] is True


def test_visualization_contract_excludes_unsupported_outputs() -> None:
    contract = load_yaml(
        Path("configs/visualization_contract.yml")
    )["visualization"]
    filenames = {
        record["filename"]
        for record in contract["brief_figures"].values()
    }

    assert "drift_optimization_dashboard.png" not in filenames
    assert "business_impact_governance.png" not in filenames
    assert "executive_decision_timeline.png" not in filenames


def test_visualization_rejects_empty_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_confirmatory_forecasting_verdict(
            pd.DataFrame(),
            pd.DataFrame(),
            tmp_path / "chart.png",
            source="Controlled test",
            sample="Not applicable",
            evidence_id="test-evidence",
        )
