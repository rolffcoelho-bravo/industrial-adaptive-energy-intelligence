from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from iaei.visualization import (
    plot_confirmatory_forecasting_verdict,
    plot_evidence_governance_model_boundaries,
    plot_governed_data_architecture,
    plot_locked_test_temporal_stability,
    plot_model_ladder_chronological_validation,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
MANIFEST_PATH = ROOT / "outputs" / "reporting_evidence_manifest.json"

SOURCE = "UCI dataset 851 and governed Gate 5B evidence"
SAMPLE = "2018-01-01 00:15 to 2019-01-01 00:00"
EVIDENCE_ID = "Gate 5B manifest 93b6ea5afed6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    confirmatory = pd.read_csv(
        TABLES / "confirmatory_metrics.csv"
    )
    data_quality = pd.read_csv(
        TABLES / "data_quality_summary.csv"
    )
    ladder = pd.read_csv(
        TABLES / "model_ladder_summary.csv"
    )
    blocks = pd.read_csv(
        TABLES / "temporal_block_results.csv"
    )
    lineage = pd.read_csv(
        TABLES / "evidence_lineage.csv"
    )
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    return (
        confirmatory,
        data_quality,
        ladder,
        blocks,
        lineage,
        manifest,
    )


def _assert_publication_png(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size >= 30_000

    with Image.open(path) as image:
        width, height = image.size
        assert width >= 2_400
        assert height >= 1_200
        assert image.mode in {"RGB", "RGBA"}


def test_all_reporting_figures_render_from_gate_5b_evidence(
    tmp_path: Path,
) -> None:
    (
        confirmatory,
        data_quality,
        ladder,
        blocks,
        lineage,
        manifest,
    ) = _inputs()

    outputs = [
        plot_confirmatory_forecasting_verdict(
            confirmatory,
            blocks,
            tmp_path / "confirmatory_forecasting_verdict.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        ),
        plot_governed_data_architecture(
            data_quality,
            manifest,
            tmp_path / "governed_data_architecture.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        ),
        plot_model_ladder_chronological_validation(
            ladder,
            tmp_path / "model_ladder_chronological_validation.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        ),
        plot_locked_test_temporal_stability(
            blocks,
            confirmatory,
            tmp_path / "locked_test_temporal_stability.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        ),
        plot_evidence_governance_model_boundaries(
            lineage,
            manifest,
            tmp_path / "evidence_governance_model_boundaries.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        ),
    ]

    assert len(outputs) == 5

    for path in outputs:
        _assert_publication_png(path)


def test_confirmatory_figure_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    confirmatory, _, _, blocks, _, _ = _inputs()

    first = plot_confirmatory_forecasting_verdict(
        confirmatory,
        blocks,
        tmp_path / "first.png",
        source=SOURCE,
        sample=SAMPLE,
        evidence_id=EVIDENCE_ID,
    )
    second = plot_confirmatory_forecasting_verdict(
        confirmatory,
        blocks,
        tmp_path / "second.png",
        source=SOURCE,
        sample=SAMPLE,
        evidence_id=EVIDENCE_ID,
    )

    assert _sha256(first) == _sha256(second)
    assert first.read_bytes() == second.read_bytes()


def test_figures_reject_placeholder_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_confirmatory_forecasting_verdict(
            pd.DataFrame(),
            pd.DataFrame(),
            tmp_path / "invalid.png",
            source=SOURCE,
            sample=SAMPLE,
            evidence_id=EVIDENCE_ID,
        )


def test_chart_module_excludes_locked_prediction_rows() -> None:
    source = (
        ROOT
        / "src"
        / "iaei"
        / "visualization"
        / "charts.py"
    ).read_text(encoding="utf-8")

    assert "locked_test_predictions.csv" not in source
    assert "evaluate_locked_test_once" not in source
    assert "fit(" not in source


def test_package_exports_only_evidence_aligned_figures() -> None:
    path = (
        ROOT
        / "src"
        / "iaei"
        / "visualization"
        / "__init__.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exported: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                exported = [
                    value.value
                    for value in node.value.elts
                    if isinstance(value, ast.Constant)
                ]

    assert set(exported) == {
        "plot_confirmatory_forecasting_verdict",
        "plot_evidence_governance_model_boundaries",
        "plot_governed_data_architecture",
        "plot_locked_test_temporal_stability",
        "plot_model_ladder_chronological_validation",
    }


def test_legacy_unsupported_chart_api_is_removed() -> None:
    chart_source = (
        ROOT
        / "src"
        / "iaei"
        / "visualization"
        / "charts.py"
    ).read_text(encoding="utf-8")

    legacy_names = (
        "plot_executive_decision_timeline",
        "plot_industrial_load_profile",
        "plot_model_validation_dashboard",
        "plot_drift_optimization_dashboard",
        "plot_business_impact_governance",
    )

    for name in legacy_names:
        assert name not in chart_source
