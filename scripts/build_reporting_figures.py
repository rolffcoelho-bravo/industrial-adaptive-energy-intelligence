from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from iaei.visualization import (
    plot_confirmatory_forecasting_verdict,
    plot_evidence_governance_model_boundaries,
    plot_governed_data_architecture,
    plot_locked_test_temporal_stability,
    plot_model_ladder_chronological_validation,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
CHARTS = ROOT / "outputs" / "charts"
MANIFEST_PATH = ROOT / "outputs" / "reporting_evidence_manifest.json"

EXPECTED_INPUT_HASHES = {
    "outputs/reporting_evidence_manifest.json": (
        "93b6ea5afed692d3038244fb6262b51bacdc286e7a0e64a8300b64a0b4e722bd"
    ),
    "outputs/tables/confirmatory_metrics.csv": (
        "bd6816dcd518fc3a18eed95049adbf5a4d9b6d0f19252740bfc2a362a5ff3073"
    ),
    "outputs/tables/data_quality_summary.csv": (
        "43ae0a766d3afa3e49a1653bcccd9eaaa61492dff1241eb728e860b168b6566a"
    ),
    "outputs/tables/evidence_lineage.csv": (
        "ae3c7229370e556ed7e8bb22deb3d195d4e07506e4018edc9269f0c62db34507"
    ),
    "outputs/tables/model_ladder_summary.csv": (
        "a61cbab367ad83a4199919c04d5fc9285f9abffba1da9ab762134d03866d3c58"
    ),
    "outputs/tables/temporal_block_results.csv": (
        "58ab345ca592b279fc8a41e65642d663a868b334cee28a6fa8cfd0a517446d27"
    ),
}

FIGURE_FILENAMES = (
    "confirmatory_forecasting_verdict.png",
    "governed_data_architecture.png",
    "model_ladder_chronological_validation.png",
    "locked_test_temporal_stability.png",
    "evidence_governance_model_boundaries.png",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_inputs() -> None:
    for relative_path, expected_hash in EXPECTED_INPUT_HASHES.items():
        path = ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(path)

        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                "Reporting evidence identity changed: "
                f"{relative_path}"
            )


def _verify_controls(manifest: dict[str, object]) -> None:
    controls = manifest.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("Reporting evidence controls are missing")

    required_false = (
        "evaluator_imported",
        "evaluator_invoked",
        "locked_metrics_recalculated_from_predictions",
        "locked_predictions_parsed",
        "model_fitting_performed",
        "reestimation_performed",
        "second_locked_test_evaluation_performed",
    )

    for field in required_false:
        if controls.get(field) is not False:
            raise ValueError(
                f"Reporting control is not closed: {field}"
            )


def build_reporting_figures() -> list[Path]:
    _verify_inputs()

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )
    _verify_controls(manifest)

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

    if len(data_quality) != 1:
        raise ValueError(
            "Data quality summary must contain exactly one row"
        )

    data_row = data_quality.iloc[0]
    dataset_id = str(data_row["dataset_id"]).removeprefix("uci-")
    sample_start = str(data_row["effective_sample_start"]).replace(
        "T",
        " ",
    )
    sample_end = str(data_row["effective_sample_end"]).replace(
        "T",
        " ",
    )

    source = (
        f"UCI dataset {dataset_id} and governed Gate 5B evidence"
    )
    sample = f"{sample_start} to {sample_end}"
    evidence_id = (
        "Gate 5B manifest "
        + EXPECTED_INPUT_HASHES[
            "outputs/reporting_evidence_manifest.json"
        ][:12]
    )

    outputs = [
        plot_confirmatory_forecasting_verdict(
            confirmatory,
            blocks,
            CHARTS / FIGURE_FILENAMES[0],
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        ),
        plot_governed_data_architecture(
            data_quality,
            manifest,
            CHARTS / FIGURE_FILENAMES[1],
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        ),
        plot_model_ladder_chronological_validation(
            ladder,
            CHARTS / FIGURE_FILENAMES[2],
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        ),
        plot_locked_test_temporal_stability(
            blocks,
            confirmatory,
            CHARTS / FIGURE_FILENAMES[3],
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        ),
        plot_evidence_governance_model_boundaries(
            lineage,
            manifest,
            CHARTS / FIGURE_FILENAMES[4],
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        ),
    ]

    if tuple(path.name for path in outputs) != FIGURE_FILENAMES:
        raise ValueError("Unexpected reporting figure set")

    return outputs


if __name__ == "__main__":
    rendered = build_reporting_figures()
    print(
        "Reporting figures: PASS | "
        f"count={len(rendered)} | "
        f"output={CHARTS}"
    )
