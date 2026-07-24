from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_reporting_figures.py"

EXPECTED_INPUTS = {
    "outputs/reporting_evidence_manifest.json",
    "outputs/tables/confirmatory_metrics.csv",
    "outputs/tables/data_quality_summary.csv",
    "outputs/tables/evidence_lineage.csv",
    "outputs/tables/model_ladder_summary.csv",
    "outputs/tables/temporal_block_results.csv",
}

EXPECTED_FIGURES = {
    "confirmatory_forecasting_verdict.png",
    "governed_data_architecture.png",
    "model_ladder_chronological_validation.png",
    "locked_test_temporal_stability.png",
    "evidence_governance_model_boundaries.png",
}


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reporting_figure_builder",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_uses_only_gate_5b_inputs() -> None:
    module = _load_builder()

    assert set(module.EXPECTED_INPUT_HASHES) == EXPECTED_INPUTS

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "locked_test_predictions.csv" not in source
    assert "evaluate_locked_test_once" not in source
    assert "locked_test_results.json" not in source
    assert "fit(" not in source


def test_builder_orchestrates_exact_figure_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_builder()
    module.CHARTS = tmp_path

    def fake_plot(*args: object, **kwargs: object) -> Path:
        del kwargs
        output_path = next(
            value
            for value in reversed(args)
            if isinstance(value, Path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"controlled-builder-test")
        return output_path

    plot_names = (
        "plot_confirmatory_forecasting_verdict",
        "plot_governed_data_architecture",
        "plot_model_ladder_chronological_validation",
        "plot_locked_test_temporal_stability",
        "plot_evidence_governance_model_boundaries",
    )

    for name in plot_names:
        monkeypatch.setattr(module, name, fake_plot)

    outputs = module.build_reporting_figures()

    assert {path.name for path in outputs} == EXPECTED_FIGURES
    assert {path.name for path in tmp_path.iterdir()} == EXPECTED_FIGURES


def test_builder_contains_no_clock_metadata() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "datetime.now" not in source
    assert "utcnow" not in source
    assert "Generated:" not in source
