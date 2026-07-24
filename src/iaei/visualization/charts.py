from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as mpl

mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from iaei.visualization.style import (
    PALETTE,
    add_figure_header,
    publication_style,
    save_publication_figure,
    style_axis,
)


def _require_frame(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    if frame.empty:
        raise ValueError(
            f"{name} cannot be empty; placeholder charts are prohibited"
        )

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> pd.DataFrame:
    data = frame.copy()

    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    return data


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise ValueError(f"Cannot interpret boolean value: {value}")


def _model_label(value: str) -> str:
    labels = {
        "persistence": "Persistence",
        "ridge": "Ridge",
        "elastic_net": "Elastic Net",
        "hist_gradient_boosting": "Histogram gradient boosting",
    }
    return labels.get(value, value.replace("_", " ").title())


def _relative_label(value: float) -> str:
    if value > 0:
        return f"{value:.1%} better"
    if value < 0:
        return f"{abs(value):.1%} worse"
    return "Reference"


def _paired_mae_panel(
    ax: plt.Axes,
    row: pd.Series,
    *,
    title: str,
) -> None:
    candidate = float(row["candidate_mae"])
    reference = float(row["reference_mae"])
    improvement = float(row["relative_mae_improvement"])

    labels = ["Frozen HGB", "Persistence"]
    values = np.array([candidate, reference])
    positions = np.array([1, 0])
    colors = [PALETTE["teal"], PALETTE["slate"]]
    bars = ax.barh(positions, values, color=colors, height=0.52)
    maximum = float(values.max())

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + maximum * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} kWh",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.text(
        0.98,
        0.92,
        f"{improvement:.1%} lower MAE",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["teal"],
    )
    ax.set_yticks(positions, labels=labels)
    ax.set_xlim(0, maximum * 1.34)
    ax.set_xlabel("MAE, kWh")
    ax.set_title(title, loc="left")
    style_axis(ax, grid_axis="x")


def plot_confirmatory_forecasting_verdict(
    confirmatory: pd.DataFrame,
    blocks: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    evidence_id: str,
) -> Path:
    _require_frame(
        confirmatory,
        [
            "metric_scope",
            "candidate_mae",
            "reference_mae",
            "relative_mae_improvement",
            "origin_count",
        ],
        "confirmatory metrics",
    )
    _require_frame(
        blocks,
        [
            "block_id",
            "candidate_mae",
            "persistence_mae",
            "relative_mae_improvement",
            "origin_count",
        ],
        "temporal blocks",
    )

    metrics = _numeric(
        confirmatory,
        [
            "candidate_mae",
            "reference_mae",
            "relative_mae_improvement",
            "origin_count",
        ],
        "confirmatory metrics",
    ).set_index("metric_scope")

    if set(metrics.index) != {"aggregate", "peak_state"}:
        raise ValueError(
            "Confirmatory metrics require aggregate and peak_state rows"
        )

    temporal = _numeric(
        blocks,
        [
            "block_id",
            "candidate_mae",
            "persistence_mae",
            "relative_mae_improvement",
            "origin_count",
        ],
        "temporal blocks",
    ).sort_values("block_id")

    if len(temporal) != 4:
        raise ValueError(
            "Exactly four prespecified temporal blocks are required"
        )
    if (temporal["relative_mae_improvement"] <= 0).any():
        raise ValueError(
            "Every temporal block must retain positive improvement"
        )

    with publication_style():
        fig = plt.figure(figsize=(11.2, 6.4))
        grid = GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.0, 1.38],
            hspace=0.47,
            wspace=0.33,
        )
        ax_aggregate = fig.add_subplot(grid[0, 0])
        ax_peak = fig.add_subplot(grid[0, 1])
        ax_blocks = fig.add_subplot(grid[1, :])

        add_figure_header(
            fig,
            "Confirmatory forecasting verdict",
            (
                "The frozen model beat persistence in aggregate, during "
                "peak-demand states, and in every prespecified time block."
            ),
        )

        _paired_mae_panel(
            ax_aggregate,
            metrics.loc["aggregate"],
            title="All confirmatory origins",
        )
        _paired_mae_panel(
            ax_peak,
            metrics.loc["peak_state"],
            title="Peak-demand states",
        )

        x = np.arange(len(temporal))
        width = 0.34
        candidate = temporal["candidate_mae"].to_numpy(dtype=float)
        reference = temporal["persistence_mae"].to_numpy(dtype=float)
        improvements = temporal["relative_mae_improvement"].to_numpy(
            dtype=float
        )

        candidate_bars = ax_blocks.bar(
            x - width / 2,
            candidate,
            width,
            color=PALETTE["teal"],
            label="Frozen HGB",
        )
        reference_bars = ax_blocks.bar(
            x + width / 2,
            reference,
            width,
            color=PALETTE["slate"],
            label="Persistence",
        )
        reference_max = float(reference.max())

        for position, improvement in enumerate(improvements):
            top = max(candidate[position], reference[position])
            ax_blocks.text(
                position,
                top + reference_max * 0.055,
                f"{improvement:.1%} lower",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color=PALETTE["teal"],
            )

        for bars in (candidate_bars, reference_bars):
            for bar in bars:
                ax_blocks.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + reference_max * 0.012,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                )

        ax_blocks.set_xticks(
            x,
            labels=[
                f"Block {int(value)}"
                for value in temporal["block_id"]
            ],
        )
        ax_blocks.set_ylabel("MAE, kWh")
        ax_blocks.set_title(
            "Prespecified temporal stability",
            loc="left",
        )
        ax_blocks.legend(loc="upper right", ncols=2)
        ax_blocks.set_ylim(0, reference_max * 1.29)
        style_axis(ax_blocks)

        fig.subplots_adjust(
            left=0.075,
            right=0.98,
            top=0.84,
            bottom=0.12,
        )
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 1",
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        )


def _architecture_box(
    ax: plt.Axes,
    *,
    x: float,
    title: str,
    lines: list[str],
    accent: str,
) -> None:
    box = FancyBboxPatch(
        (x, 0.42),
        0.17,
        0.43,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=ax.transAxes,
        linewidth=1.0,
        edgecolor=accent,
        facecolor=PALETTE["white"],
    )
    ax.add_patch(box)

    ax.add_patch(
        FancyBboxPatch(
            (x, 0.80),
            0.17,
            0.05,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            transform=ax.transAxes,
            linewidth=0,
            facecolor=accent,
        )
    )
    ax.text(
        x + 0.012,
        0.775,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        x + 0.012,
        0.725,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        linespacing=1.45,
        color=PALETTE["muted"],
    )


def plot_governed_data_architecture(
    data_quality: pd.DataFrame,
    manifest: Mapping[str, Any],
    output_path: Path,
    *,
    source: str,
    sample: str,
    evidence_id: str,
) -> Path:
    _require_frame(
        data_quality,
        [
            "dataset_id",
            "license",
            "raw_csv_sha256",
            "raw_row_count",
            "source_column_count",
            "expected_frequency_minutes",
            "silver_row_count",
            "silver_column_count",
            "dq_any_count",
            "source_order_preserved",
            "supervised_targets_present",
        ],
        "data quality summary",
    )

    if len(data_quality) != 1:
        raise ValueError(
            "Data quality summary must contain exactly one row"
        )

    row = data_quality.iloc[0]
    generated = manifest.get("generated_artifacts", {})
    sources = manifest.get("source_artifacts", {})
    controls = manifest.get("controls", {})

    if len(generated) != 5:
        raise ValueError(
            "Gate 5B manifest must govern five evidence tables"
        )
    if controls.get("locked_predictions_parsed") is not False:
        raise ValueError(
            "Locked prediction parsing control is not closed"
        )
    if not _truthy(row["source_order_preserved"]):
        raise ValueError("Source order preservation did not pass")
    if _truthy(row["supervised_targets_present"]):
        raise ValueError("Silver layer contains supervised targets")

    dataset_id = str(row["dataset_id"])
    license_name = str(row["license"])
    raw_rows = int(row["raw_row_count"])
    source_columns = int(row["source_column_count"])
    interval_minutes = int(row["expected_frequency_minutes"])
    silver_rows = int(row["silver_row_count"])
    silver_columns = int(row["silver_column_count"])
    quality_exceptions = int(row["dq_any_count"])
    raw_hash = str(row["raw_csv_sha256"])[:12]

    stages = [
        (
            "Official source",
            [
                f"Dataset {dataset_id}",
                f"{raw_rows:,} rows",
                f"{source_columns} source fields",
                license_name,
            ],
            PALETTE["navy"],
        ),
        (
            "Immutable raw layer",
            [
                "Source order preserved",
                f"SHA256 {raw_hash}...",
                "No sorting or imputation",
                "Licensed public evidence",
            ],
            PALETTE["blue"],
        ),
        (
            "Source-aware chronology",
            [
                f"{interval_minutes}-minute intervals",
                "Operational midnight corrected",
                "Continuous effective time",
                "Future values excluded",
            ],
            PALETTE["teal"],
        ),
        (
            "Governed Silver layer",
            [
                f"{silver_rows:,} rows",
                f"{silver_columns} analytical fields",
                f"{quality_exceptions} quality exceptions",
                "No supervised targets stored",
            ],
            PALETTE["amber"],
        ),
        (
            "Evidence synthesis",
            [
                f"{len(sources)} governed inputs",
                f"{len(generated)} final tables",
                "Deterministic serialization",
                "Reporting downstream only",
            ],
            PALETTE["vermillion"],
        ),
    ]

    with publication_style():
        fig, ax = plt.subplots(figsize=(11.2, 6.4))
        add_figure_header(
            fig,
            "Governed data and analytical architecture",
            (
                "The reporting layer preserves provenance, chronology, "
                "quality controls, and separation from confirmatory "
                "model execution."
            ),
        )
        ax.axis("off")

        x_positions = [0.025, 0.22, 0.415, 0.61, 0.805]

        for index, (stage, x) in enumerate(
            zip(stages, x_positions, strict=True)
        ):
            title, lines, accent = stage
            _architecture_box(
                ax,
                x=x,
                title=title,
                lines=lines,
                accent=accent,
            )

            if index < len(stages) - 1:
                arrow = FancyArrowPatch(
                    (x + 0.173, 0.635),
                    (x_positions[index + 1] - 0.004, 0.635),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.0,
                    color=PALETTE["slate"],
                )
                ax.add_patch(arrow)

        controls_text = (
            "CONTROL BOUNDARY\n"
            "Raw source unchanged   |   "
            "No future targets in Silver   |   "
            "Locked prediction rows not parsed   |   "
            "No model fitting in reporting"
        )
        ax.text(
            0.5,
            0.20,
            controls_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            linespacing=1.5,
            color=PALETTE["navy"],
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": PALETTE["light"],
                "edgecolor": PALETTE["grid"],
            },
        )

        fig.subplots_adjust(
            left=0.04,
            right=0.985,
            top=0.86,
            bottom=0.09,
        )
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 2",
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        )


def plot_model_ladder_chronological_validation(
    ladder: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    evidence_id: str,
) -> Path:
    _require_frame(
        ladder,
        [
            "model",
            "mean_validation_mae",
            "mean_peak_state_mae",
            "worst_fold_mae",
            "relative_mae_improvement_vs_persistence",
            "promotion_decision",
            "validation_fold_count",
            "validation_origin_count",
            "locked_test_used_for_selection",
        ],
        "model ladder",
    )

    data = _numeric(
        ladder,
        [
            "mean_validation_mae",
            "mean_peak_state_mae",
            "worst_fold_mae",
            "relative_mae_improvement_vs_persistence",
            "validation_fold_count",
            "validation_origin_count",
        ],
        "model ladder",
    )

    expected_models = {
        "persistence",
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
    }
    if set(data["model"]) != expected_models:
        raise ValueError("Unexpected model ladder membership")

    if any(
        _truthy(value)
        for value in data["locked_test_used_for_selection"]
    ):
        raise ValueError("Locked test was used for model selection")

    if data["validation_fold_count"].nunique() != 1:
        raise ValueError("Validation fold count is inconsistent")
    if data["validation_origin_count"].nunique() != 1:
        raise ValueError("Validation origin count is inconsistent")

    order = [
        "hist_gradient_boosting",
        "persistence",
        "ridge",
        "elastic_net",
    ]
    data = data.set_index("model").loc[order].reset_index()
    labels = [_model_label(value) for value in data["model"]]
    colors = [
        PALETTE["teal"],
        PALETTE["slate"],
        PALETTE["amber"],
        PALETTE["vermillion"],
    ]
    positions = np.arange(len(data))[::-1]

    with publication_style():
        fig = plt.figure(figsize=(11.2, 6.4))
        grid = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.0, 1.0],
            wspace=0.32,
        )
        ax_mean = fig.add_subplot(grid[0, 0])
        ax_peak = fig.add_subplot(grid[0, 1])

        add_figure_header(
            fig,
            "Model ladder and chronological validation",
            (
                "Four expanding-window folds selected the nonlinear "
                "candidate using validation evidence only. The locked "
                "test was excluded from model choice."
            ),
        )

        mean_values = data["mean_validation_mae"].to_numpy(dtype=float)
        mean_bars = ax_mean.barh(
            positions,
            mean_values,
            color=colors,
            height=0.58,
        )
        mean_max = float(mean_values.max())

        for row_index, (bar, value) in enumerate(
            zip(mean_bars, mean_values, strict=True)
        ):
            relative = float(
                data.iloc[row_index][
                    "relative_mae_improvement_vs_persistence"
                ]
            )
            weight = (
                "bold"
                if data.iloc[row_index]["model"]
                == "hist_gradient_boosting"
                else "normal"
            )
            ax_mean.text(
                value + mean_max * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}  |  {_relative_label(relative)}",
                va="center",
                fontsize=7.7,
                fontweight=weight,
            )

        persistence_mean = float(
            data.loc[
                data["model"] == "persistence",
                "mean_validation_mae",
            ].iloc[0]
        )
        ax_mean.axvline(
            persistence_mean,
            color=PALETTE["slate"],
            linestyle="--",
            linewidth=1.0,
        )
        ax_mean.set_yticks(positions, labels=labels)
        ax_mean.set_xlim(0, mean_max * 1.52)
        ax_mean.set_xlabel("Mean validation MAE, kWh")
        ax_mean.set_title("Aggregate validation", loc="left")
        style_axis(ax_mean, grid_axis="x")

        peak_values = data["mean_peak_state_mae"].to_numpy(dtype=float)
        worst_values = data["worst_fold_mae"].to_numpy(dtype=float)
        peak_bars = ax_peak.barh(
            positions,
            peak_values,
            color=colors,
            height=0.58,
            alpha=0.9,
        )
        peak_max = float(peak_values.max())

        for bar, peak_value, worst_value in zip(
            peak_bars,
            peak_values,
            worst_values,
            strict=True,
        ):
            ax_peak.text(
                peak_value + peak_max * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"Peak {peak_value:.2f}  |  Worst fold {worst_value:.2f}",
                va="center",
                fontsize=7.5,
            )

        ax_peak.set_yticks(positions, labels=[])
        ax_peak.set_xlim(0, peak_max * 1.56)
        ax_peak.set_xlabel("Mean peak-state MAE, kWh")
        ax_peak.set_title(
            "Peak-state and worst-fold evidence",
            loc="left",
        )
        style_axis(ax_peak, grid_axis="x")

        fold_count = int(data["validation_fold_count"].iloc[0])
        origin_count = int(data["validation_origin_count"].iloc[0])
        fig.text(
            0.055,
            0.082,
            (
                f"SELECTION CONTROL   {fold_count} chronological folds  |  "
                f"{origin_count:,} validation origins  |  "
                "locked test used for selection: no"
            ),
            fontsize=8.1,
            fontweight="bold",
            color=PALETTE["navy"],
        )

        fig.subplots_adjust(
            left=0.17,
            right=0.98,
            top=0.84,
            bottom=0.17,
        )
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 3",
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        )


def plot_locked_test_temporal_stability(
    blocks: pd.DataFrame,
    confirmatory: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    evidence_id: str,
) -> Path:
    _require_frame(
        blocks,
        [
            "block_id",
            "origin_start",
            "origin_stop_exclusive",
            "origin_count",
            "candidate_mae",
            "persistence_mae",
            "relative_mae_improvement",
        ],
        "temporal blocks",
    )
    _require_frame(
        confirmatory,
        [
            "metric_scope",
            "candidate_mae",
            "reference_mae",
            "relative_mae_improvement",
            "origin_count",
            "peak_threshold_kwh",
        ],
        "confirmatory metrics",
    )

    temporal = _numeric(
        blocks,
        [
            "block_id",
            "origin_start",
            "origin_stop_exclusive",
            "origin_count",
            "candidate_mae",
            "persistence_mae",
            "relative_mae_improvement",
        ],
        "temporal blocks",
    ).sort_values("block_id")

    metrics = _numeric(
        confirmatory,
        [
            "candidate_mae",
            "reference_mae",
            "relative_mae_improvement",
            "origin_count",
        ],
        "confirmatory metrics",
    ).set_index("metric_scope")

    if len(temporal) != 4:
        raise ValueError("Exactly four temporal blocks are required")
    if temporal["origin_count"].nunique() != 1:
        raise ValueError("Temporal blocks must have equal origin counts")
    if set(metrics.index) != {"aggregate", "peak_state"}:
        raise ValueError("Unexpected confirmatory metric scopes")

    peak_threshold = float(
        confirmatory.loc[
            confirmatory["metric_scope"] == "peak_state",
            "peak_threshold_kwh",
        ].iloc[0]
    )

    with publication_style():
        fig = plt.figure(figsize=(11.2, 6.4))
        grid = GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.55, 0.85],
            width_ratios=[1.35, 1.0],
            hspace=0.44,
            wspace=0.34,
        )
        ax_blocks = fig.add_subplot(grid[0, :])
        ax_metrics = fig.add_subplot(grid[1, 0])
        ax_boundary = fig.add_subplot(grid[1, 1])

        add_figure_header(
            fig,
            "Locked-test temporal stability",
            (
                "The frozen model retained lower MAE in all four equal, "
                "prespecified blocks and during the governed peak state."
            ),
        )

        x = np.arange(len(temporal))
        candidate = temporal["candidate_mae"].to_numpy(dtype=float)
        reference = temporal["persistence_mae"].to_numpy(dtype=float)
        improvement = temporal["relative_mae_improvement"].to_numpy(
            dtype=float
        )

        ax_blocks.plot(
            x,
            candidate,
            marker="o",
            markersize=7,
            color=PALETTE["teal"],
            label="Frozen HGB",
        )
        ax_blocks.plot(
            x,
            reference,
            marker="o",
            markersize=7,
            color=PALETTE["slate"],
            label="Persistence",
        )

        for position in range(len(temporal)):
            ax_blocks.fill_between(
                [position - 0.08, position + 0.08],
                [candidate[position], candidate[position]],
                [reference[position], reference[position]],
                color=PALETTE["teal"],
                alpha=0.15,
            )
            ax_blocks.text(
                position,
                reference[position] + 0.28,
                f"{improvement[position]:.1%} lower",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color=PALETTE["teal"],
            )

        ax_blocks.set_xticks(
            x,
            labels=[
                f"Block {int(value)}\n{int(count):,} origins"
                for value, count in zip(
                    temporal["block_id"],
                    temporal["origin_count"],
                    strict=True,
                )
            ],
        )
        ax_blocks.set_ylabel("MAE, kWh")
        ax_blocks.set_title(
            "Candidate and reference performance by locked block",
            loc="left",
        )
        ax_blocks.legend(loc="upper right", ncols=2)
        ax_blocks.set_ylim(0, float(reference.max()) * 1.23)
        style_axis(ax_blocks)

        aggregate = metrics.loc["aggregate"]
        peak = metrics.loc["peak_state"]

        ax_metrics.axis("off")
        ax_metrics.text(
            0.0,
            0.96,
            "Confirmatory summary",
            transform=ax_metrics.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax_metrics.text(
            0.0,
            0.68,
            f"{float(aggregate['relative_mae_improvement']):.1%}",
            transform=ax_metrics.transAxes,
            fontsize=21,
            fontweight="bold",
            color=PALETTE["teal"],
        )
        ax_metrics.text(
            0.0,
            0.48,
            (
                "aggregate MAE reduction\n"
                f"{int(aggregate['origin_count']):,} confirmatory origins"
            ),
            transform=ax_metrics.transAxes,
            fontsize=8.2,
            linespacing=1.35,
        )
        ax_metrics.text(
            0.58,
            0.68,
            f"{float(peak['relative_mae_improvement']):.1%}",
            transform=ax_metrics.transAxes,
            fontsize=21,
            fontweight="bold",
            color=PALETTE["teal"],
        )
        ax_metrics.text(
            0.58,
            0.48,
            (
                "peak-state MAE reduction\n"
                f"{int(peak['origin_count']):,} governed rows"
            ),
            transform=ax_metrics.transAxes,
            fontsize=8.2,
            linespacing=1.35,
        )

        first = temporal.iloc[0]
        last = temporal.iloc[-1]
        origin_start = int(first["origin_start"])
        origin_stop = int(last["origin_stop_exclusive"]) - 1
        block_size = int(first["origin_count"])

        ax_boundary.axis("off")
        ax_boundary.text(
            0.0,
            0.96,
            "Locked evaluation boundary",
            transform=ax_boundary.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        boundary_lines = [
            f"Evaluation origins: {origin_start:,} to {origin_stop:,}",
            f"Four equal blocks: {block_size:,} origins each",
            f"Peak threshold: {peak_threshold:.2f} kWh",
            "Evaluation count: one",
        ]

        for line_index, line in enumerate(boundary_lines):
            ax_boundary.text(
                0.0,
                0.69 - line_index * 0.17,
                line,
                transform=ax_boundary.transAxes,
                fontsize=8.5,
                color=PALETTE["muted"],
            )

        fig.subplots_adjust(
            left=0.075,
            right=0.98,
            top=0.84,
            bottom=0.12,
        )
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 4",
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        )


def plot_evidence_governance_model_boundaries(
    lineage: pd.DataFrame,
    manifest: Mapping[str, Any],
    output_path: Path,
    *,
    source: str,
    sample: str,
    evidence_id: str,
) -> Path:
    _require_frame(
        lineage,
        [
            "sequence",
            "governance_gate",
            "artifact_role",
            "status",
            "decision",
            "sha256",
        ],
        "evidence lineage",
    )

    data = _numeric(
        lineage,
        ["sequence"],
        "evidence lineage",
    ).sort_values("sequence")
    controls = manifest.get("controls", {})

    if len(data) != 8:
        raise ValueError(
            "Evidence lineage must contain eight governed records"
        )
    if controls.get("second_locked_test_evaluation_performed") is not False:
        raise ValueError("Second-evaluation control is not closed")
    if controls.get("model_fitting_performed") is not False:
        raise ValueError("Reporting model-fitting control is not closed")
    if controls.get("evaluator_invoked") is not False:
        raise ValueError("Evaluator invocation control is not closed")

    positive_decisions = {
        "promoted",
        "model_frozen",
        "success",
        "confirmatory_metrics_recorded",
        "single_evaluation_consumed",
        "persistence_reference_locked",
    }

    with publication_style():
        fig = plt.figure(figsize=(11.2, 6.4))
        grid = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.4, 1.0],
            wspace=0.28,
        )
        ax_lineage = fig.add_subplot(grid[0, 0])
        ax_controls = fig.add_subplot(grid[0, 1])

        add_figure_header(
            fig,
            "Evidence governance and model boundaries",
            (
                "Every public claim is tied to a governed artifact, "
                "while operational and causal claims remain outside "
                "the validated scope."
            ),
        )

        positions = np.arange(len(data))[::-1]
        ax_lineage.vlines(
            0.08,
            positions.min(),
            positions.max(),
            color=PALETTE["grid"],
            linewidth=2.0,
        )

        for y_value, (_, row) in zip(
            positions,
            data.iterrows(),
            strict=True,
        ):
            decision = str(row["decision"])
            color = (
                PALETTE["teal"]
                if decision in positive_decisions
                else PALETTE["slate"]
            )
            role = str(row["artifact_role"]).replace("_", " ").title()
            hash_prefix = str(row["sha256"])[:12]
            decision_label = decision.replace("_", " ")

            ax_lineage.scatter(
                0.08,
                y_value,
                s=82,
                color=color,
                edgecolor=PALETTE["white"],
                linewidth=0.8,
                zorder=3,
            )
            ax_lineage.text(
                0.0,
                y_value,
                str(row["governance_gate"]),
                ha="right",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=PALETTE["navy"],
            )
            ax_lineage.text(
                0.14,
                y_value + 0.14,
                role,
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
            ax_lineage.text(
                0.14,
                y_value - 0.14,
                f"{decision_label}  |  sha256 {hash_prefix}...",
                ha="left",
                va="center",
                fontsize=7.2,
                color=PALETTE["muted"],
            )

        ax_lineage.set_xlim(-0.03, 1.0)
        ax_lineage.set_ylim(-0.6, len(data) - 0.4)
        ax_lineage.set_title(
            "Gate 4B to Gate 4F evidence lineage",
            loc="left",
        )
        ax_lineage.axis("off")

        ax_controls.axis("off")
        ax_controls.text(
            0.0,
            0.98,
            "Closed controls",
            transform=ax_controls.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        closed_controls = [
            "Single locked evaluation consumed",
            "Second locked evaluation prohibited",
            "Locked prediction rows not parsed",
            "No model fitting or re-estimation",
            "Evaluator not imported or invoked",
            "Terminal artifact hashes preserved",
        ]

        for control_index, text in enumerate(closed_controls):
            y_position = 0.88 - control_index * 0.095
            ax_controls.scatter(
                0.025,
                y_position,
                s=46,
                color=PALETTE["teal"],
                transform=ax_controls.transAxes,
            )
            ax_controls.text(
                0.075,
                y_position,
                text,
                transform=ax_controls.transAxes,
                va="center",
                fontsize=8.2,
            )

        ax_controls.text(
            0.0,
            0.28,
            "Outside validated scope",
            transform=ax_controls.transAxes,
            va="top",
            fontsize=10,
            fontweight="bold",
        )

        outside_scope = [
            "Structural drift conclusions",
            "Optimization recommendations",
            "Savings or business-impact estimates",
            "Causal effects",
            "Live production performance",
        ]
        outside_text = "\n".join(
            f"- {item}"
            for item in outside_scope
        )
        ax_controls.text(
            0.0,
            0.21,
            outside_text,
            transform=ax_controls.transAxes,
            va="top",
            fontsize=8.2,
            linespacing=1.55,
            color=PALETTE["muted"],
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": PALETTE["light"],
                "edgecolor": PALETTE["grid"],
            },
        )

        fig.subplots_adjust(
            left=0.075,
            right=0.98,
            top=0.84,
            bottom=0.12,
        )
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 5",
            source=source,
            sample=sample,
            evidence_id=evidence_id,
        )
