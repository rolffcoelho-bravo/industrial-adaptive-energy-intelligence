from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.visualization.charts import plot_industrial_load_profile


EXPECTED_BRIEF_FIGURES = {
    "page_1": "confirmatory_forecasting_verdict.png",
    "page_2": "governed_data_architecture.png",
    "page_3": "model_ladder_chronological_validation.png",
    "page_4": "locked_test_temporal_stability.png",
    "page_5": "evidence_governance_model_boundaries.png",
}


def test_visualization_contract_defines_evidence_aligned_figures() -> None:
    contract = load_yaml(
        Path("configs/visualization_contract.yml")
    )["visualization"]
    figures = contract["brief_figures"]

    assert len(figures) == 5
    assert contract["background"] == "light"
    assert contract["publication_gate"]["required_brief_figures"] == 5
    assert contract["publication_gate"][
        "terminal_metric_reconciliation_required"
    ] is True
    assert {
        page: record["filename"]
        for page, record in figures.items()
    } == EXPECTED_BRIEF_FIGURES


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


def test_visualization_rejects_empty_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_industrial_load_profile(
            pd.DataFrame(columns=["timestamp", "energy_demand"]),
            tmp_path / "chart.png",
            source="UCI",
            sample="not applicable",
        )
