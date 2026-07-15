from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from iaei.visualization.style import (
    PALETTE,
    add_figure_header,
    publication_style,
    save_publication_figure,
    style_axis,
)


def _require_frame(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} cannot be empty; placeholder charts are prohibited")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="raise", utc=True)
    return values.dt.tz_convert(None)


def _metric_text(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def plot_executive_decision_timeline(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    timestamp: str = "timestamp",
    actual: str = "actual",
    forecast: str = "forecast",
    lower: str = "forecast_lower",
    upper: str = "forecast_upper",
    peak_probability: str = "peak_probability",
    decision_state: str = "decision_state",
) -> Path:
    required = [timestamp, actual, forecast, lower, upper, peak_probability, decision_state]
    _require_frame(frame, required, "executive timeline")
    data = frame.sort_values(timestamp).copy()
    x = _datetime(data, timestamp)

    with publication_style():
        fig = plt.figure(figsize=(11, 6.4))
        grid = GridSpec(3, 1, figure=fig, height_ratios=[3.3, 1.25, 0.65], hspace=0.12)
        ax_main = fig.add_subplot(grid[0])
        ax_risk = fig.add_subplot(grid[1], sharex=ax_main)
        ax_state = fig.add_subplot(grid[2], sharex=ax_main)
        add_figure_header(
            fig,
            "Executive decision timeline",
            "Observed demand, forecast uncertainty, peak-load risk, and governed operating state.",
        )

        ax_main.fill_between(x, data[lower], data[upper], color=PALETTE["blue"], alpha=0.16, linewidth=0)
        ax_main.plot(x, data[actual], color=PALETTE["ink"], linewidth=1.55, label="Observed demand")
        ax_main.plot(x, data[forecast], color=PALETTE["blue"], linewidth=1.75, label="Forecast")
        ax_main.set_ylabel("Energy demand")
        style_axis(ax_main)
        ax_main.legend(loc="upper left", ncols=2)

        ax_risk.plot(x, data[peak_probability], color=PALETTE["amber"], linewidth=1.65)
        ax_risk.fill_between(x, 0, data[peak_probability], color=PALETTE["amber"], alpha=0.14)
        ax_risk.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=0.9)
        ax_risk.set_ylim(0, 1)
        ax_risk.set_ylabel("Peak risk")
        style_axis(ax_risk)

        state_map = {"stable": 0, "watch": 1, "adaptation_candidate": 2, "no_action": 0}
        states = data[decision_state].astype(str).str.lower().map(state_map)
        if states.isna().any():
            unknown = sorted(data.loc[states.isna(), decision_state].astype(str).unique())
            raise ValueError(f"Unknown decision states: {unknown}")
        state_colors = [PALETTE["teal"], PALETTE["amber"], PALETTE["vermillion"]]
        cmap = LinearSegmentedColormap.from_list("decision", state_colors, N=3)
        ax_state.imshow(
            states.to_numpy()[None, :],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=2,
            extent=[mdates.date2num(x.iloc[0]), mdates.date2num(x.iloc[-1]), 0, 1],
        )
        ax_state.set_yticks([0.5], labels=["Decision"])
        ax_state.set_xlabel("Time")
        ax_state.grid(False)
        ax_state.spines[["top", "right", "left"]].set_visible(False)
        ax_state.xaxis_date()
        ax_state.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax_state.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_state.xaxis.get_major_locator()))
        plt.setp(ax_main.get_xticklabels(), visible=False)
        plt.setp(ax_risk.get_xticklabels(), visible=False)
        fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.105)
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 1",
            source=source,
            sample=sample,
        )


def plot_industrial_load_profile(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    timestamp: str = "timestamp",
    value: str = "energy_demand",
) -> Path:
    _require_frame(frame, [timestamp, value], "industrial load profile")
    data = frame.copy()
    data["_timestamp"] = _datetime(data, timestamp)
    data["_weekday"] = data["_timestamp"].dt.dayofweek
    data["_hour"] = data["_timestamp"].dt.hour
    pivot = data.pivot_table(index="_weekday", columns="_hour", values=value, aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=range(24))
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hourly = data.groupby("_hour", observed=True)[value].agg(["mean", "std"]).reindex(range(24))
    daily = data.groupby("_weekday", observed=True)[value].mean().reindex(range(7))

    with publication_style():
        fig = plt.figure(figsize=(11, 6.4))
        grid = GridSpec(2, 3, figure=fig, width_ratios=[4.8, 1.55, 1.55], height_ratios=[3.5, 1.35], hspace=0.35, wspace=0.35)
        ax_heat = fig.add_subplot(grid[:, 0])
        ax_hour = fig.add_subplot(grid[0, 1:])
        ax_day = fig.add_subplot(grid[1, 1:])
        add_figure_header(
            fig,
            "Industrial load structure",
            "Average energy intensity by weekday and hour, with marginal operating profiles.",
        )

        cmap = LinearSegmentedColormap.from_list("industrial_load", [PALETTE["light"], PALETTE["blue"], PALETTE["navy"]])
        image = ax_heat.imshow(pivot.to_numpy(), aspect="auto", cmap=cmap, interpolation="nearest")
        ax_heat.set_yticks(range(7), labels=weekday_labels)
        ax_heat.set_xticks(range(0, 24, 3), labels=[f"{hour:02d}:00" for hour in range(0, 24, 3)])
        ax_heat.set_xlabel("Hour of day")
        ax_heat.set_ylabel("Weekday")
        colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.025)
        colorbar.set_label("Average energy demand", fontsize=8)

        hours = np.arange(24)
        ax_hour.plot(hours, hourly["mean"], color=PALETTE["navy"], linewidth=2.0)
        ax_hour.fill_between(
            hours,
            hourly["mean"] - hourly["std"].fillna(0),
            hourly["mean"] + hourly["std"].fillna(0),
            color=PALETTE["blue"],
            alpha=0.15,
            linewidth=0,
        )
        ax_hour.set_title("Intraday profile", loc="left", fontsize=10)
        ax_hour.set_xlim(0, 23)
        ax_hour.set_ylabel("Energy demand")
        style_axis(ax_hour)

        ax_day.barh(weekday_labels, daily.to_numpy(), color=PALETTE["teal"], alpha=0.88)
        ax_day.set_title("Weekday concentration", loc="left", fontsize=10)
        ax_day.set_xlabel("Average energy demand")
        style_axis(ax_day, grid_axis="x")
        fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.105)
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 2",
            source=source,
            sample=sample,
        )


def plot_model_validation_dashboard(
    folds: pd.DataFrame,
    calibration: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    model: str = "model",
    fold: str = "fold",
    metric_value: str = "metric_value",
    probability: str = "probability",
    target: str = "target",
) -> Path:
    _require_frame(folds, [model, fold, metric_value], "chronological fold results")
    _require_frame(calibration, [probability, target], "probability calibration")
    matrix = folds.pivot_table(index=model, columns=fold, values=metric_value, aggfunc="mean")
    matrix = matrix.sort_index()
    mean_score = matrix.mean(axis=1)
    worst_score = matrix.max(axis=1)

    bins = np.linspace(0, 1, 11)
    calibrated = calibration.copy()
    calibrated["_bin"] = pd.cut(calibrated[probability], bins=bins, include_lowest=True, duplicates="drop")
    grouped = calibrated.groupby("_bin", observed=True).agg(
        predicted=(probability, "mean"), observed=(target, "mean"), count=(target, "size")
    )
    brier = float(np.mean((calibrated[probability].to_numpy() - calibrated[target].to_numpy()) ** 2))

    with publication_style():
        fig = plt.figure(figsize=(11, 6.4))
        grid = GridSpec(2, 3, figure=fig, width_ratios=[3.9, 1.7, 2.4], height_ratios=[3.3, 1.3], hspace=0.38, wspace=0.42)
        ax_matrix = fig.add_subplot(grid[:, 0])
        ax_summary = fig.add_subplot(grid[0, 1])
        ax_cal = fig.add_subplot(grid[0, 2])
        ax_count = fig.add_subplot(grid[1, 1:])
        add_figure_header(
            fig,
            "Chronological model validation",
            "Fold-level robustness, average-versus-worst performance, and peak-risk probability calibration.",
        )

        heatmap = ax_matrix.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        ax_matrix.set_yticks(range(len(matrix.index)), labels=matrix.index)
        ax_matrix.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
        ax_matrix.set_xlabel("Chronological validation fold")
        ax_matrix.set_ylabel("Candidate model")
        fig.colorbar(heatmap, ax=ax_matrix, fraction=0.04, pad=0.025, label="Error metric")

        y = np.arange(len(matrix.index))
        ax_summary.hlines(y, mean_score, worst_score, color=PALETTE["grid"], linewidth=2.4)
        ax_summary.scatter(mean_score, y, color=PALETTE["teal"], s=38, label="Mean")
        ax_summary.scatter(worst_score, y, color=PALETTE["vermillion"], s=38, marker="D", label="Worst")
        ax_summary.set_yticks(y, labels=[])
        ax_summary.set_title("Robustness spread", loc="left", fontsize=10)
        ax_summary.set_xlabel("Error metric")
        style_axis(ax_summary, grid_axis="x")
        ax_summary.legend(loc="lower right")

        ax_cal.plot([0, 1], [0, 1], color=PALETTE["slate"], linestyle="--", linewidth=1.0)
        ax_cal.plot(grouped["predicted"], grouped["observed"], color=PALETTE["navy"], marker="o", markersize=4.5)
        ax_cal.set_xlim(0, 1)
        ax_cal.set_ylim(0, 1)
        ax_cal.set_title("Peak-risk calibration", loc="left", fontsize=10)
        ax_cal.set_xlabel("Predicted probability")
        ax_cal.set_ylabel("Observed frequency")
        ax_cal.text(0.04, 0.91, f"Brier score: {_metric_text(brier, 3)}", transform=ax_cal.transAxes, fontsize=8)
        style_axis(ax_cal)

        ax_count.bar(grouped["predicted"], grouped["count"], width=0.065, color=PALETTE["blue"], alpha=0.78)
        ax_count.set_title("Calibration-bin support", loc="left", fontsize=10)
        ax_count.set_xlabel("Mean predicted probability")
        ax_count.set_ylabel("Observations")
        style_axis(ax_count)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.12)
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 3",
            source=source,
            sample=sample,
        )


def plot_drift_optimization_dashboard(
    drift: pd.DataFrame,
    frontier: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    timestamp: str = "timestamp",
    drift_score: str = "drift_score",
    disagreement: str = "forecast_disagreement",
    state: str = "decision_state",
    cost: str = "expected_cost",
    peak_exposure: str = "peak_exposure",
    disruption: str = "schedule_disruption",
    selected: str = "selected",
) -> Path:
    _require_frame(drift, [timestamp, drift_score, disagreement, state], "drift history")
    _require_frame(frontier, [cost, peak_exposure, disruption, selected], "optimization frontier")
    timeline = drift.sort_values(timestamp).copy()
    x = _datetime(timeline, timestamp)

    with publication_style():
        fig = plt.figure(figsize=(11, 6.4))
        grid = GridSpec(2, 2, figure=fig, width_ratios=[3.25, 2.1], height_ratios=[2.1, 1.45], hspace=0.35, wspace=0.34)
        ax_drift = fig.add_subplot(grid[0, 0])
        ax_disagree = fig.add_subplot(grid[1, 0], sharex=ax_drift)
        ax_frontier = fig.add_subplot(grid[:, 1])
        add_figure_header(
            fig,
            "Structural drift and constrained operating frontier",
            "Adapt only when predictive structure moves materially and a feasible decision improves the locked objective.",
        )

        ax_drift.plot(x, timeline[drift_score], color=PALETTE["navy"], linewidth=1.8)
        ax_drift.axhline(0.6, color=PALETTE["amber"], linestyle="--", linewidth=1.0, label="Watch threshold")
        ax_drift.axhline(0.8, color=PALETTE["vermillion"], linestyle="--", linewidth=1.0, label="Adaptation threshold")
        ax_drift.fill_between(x, 0.8, timeline[drift_score], where=timeline[drift_score] >= 0.8, color=PALETTE["vermillion"], alpha=0.16)
        ax_drift.set_ylabel("Normalized drift score")
        ax_drift.set_ylim(bottom=0)
        style_axis(ax_drift)
        ax_drift.legend(loc="upper left", ncols=2)

        ax_disagree.plot(x, timeline[disagreement], color=PALETTE["teal"], linewidth=1.55)
        ax_disagree.set_ylabel("Forecast disagreement")
        ax_disagree.set_xlabel("Time")
        ax_disagree.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax_disagree.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_disagree.xaxis.get_major_locator()))
        style_axis(ax_disagree)
        plt.setp(ax_drift.get_xticklabels(), visible=False)

        sizes = 45 + 160 * (frontier[disruption] - frontier[disruption].min()) / max(
            frontier[disruption].max() - frontier[disruption].min(), 1e-12
        )
        scatter = ax_frontier.scatter(
            frontier[peak_exposure],
            frontier[cost],
            c=frontier[disruption],
            s=sizes,
            cmap="viridis",
            alpha=0.72,
            edgecolors=PALETTE["white"],
            linewidths=0.7,
        )
        chosen = frontier[frontier[selected].astype(bool)]
        if len(chosen) != 1:
            raise ValueError("Optimization frontier must contain exactly one selected operating point")
        ax_frontier.scatter(
            chosen[peak_exposure],
            chosen[cost],
            s=165,
            facecolors="none",
            edgecolors=PALETTE["vermillion"],
            linewidths=2.0,
            label="Selected point",
        )
        ax_frontier.set_title("Feasible operating frontier", loc="left", fontsize=10)
        ax_frontier.set_xlabel("Peak exposure")
        ax_frontier.set_ylabel("Expected cost")
        style_axis(ax_frontier)
        fig.colorbar(scatter, ax=ax_frontier, fraction=0.05, pad=0.03, label="Schedule disruption")
        ax_frontier.legend(loc="best")
        fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.115)
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 4",
            source=source,
            sample=sample,
        )


def plot_business_impact_governance(
    impact: pd.DataFrame,
    evidence: pd.DataFrame,
    output_path: Path,
    *,
    source: str,
    sample: str,
    component: str = "component",
    value: str = "value",
    evidence_stage: str = "stage",
    evidence_status: str = "status",
) -> Path:
    _require_frame(impact, [component, value], "business impact bridge")
    _require_frame(evidence, [evidence_stage, evidence_status], "evidence lineage")
    values = impact[value].astype(float).to_numpy()
    starts = np.r_[0.0, np.cumsum(values)[:-1]]
    totals = np.cumsum(values)
    colors = [PALETTE["teal"] if item >= 0 else PALETTE["vermillion"] for item in values]

    with publication_style():
        fig = plt.figure(figsize=(11, 6.4))
        grid = GridSpec(1, 2, figure=fig, width_ratios=[3.6, 2.1], wspace=0.38)
        ax_bridge = fig.add_subplot(grid[0, 0])
        ax_evidence = fig.add_subplot(grid[0, 1])
        add_figure_header(
            fig,
            "Assumption-bounded impact and evidence governance",
            "Measured outputs, derived scenario effects, and human authorization remain visibly separated.",
        )

        positions = np.arange(len(values))
        ax_bridge.bar(positions, values, bottom=starts, color=colors, width=0.68, alpha=0.9)
        ax_bridge.plot(positions, totals, color=PALETTE["ink"], linewidth=1.2, marker="o", markersize=3.5)
        ax_bridge.axhline(0, color=PALETTE["ink"], linewidth=0.8)
        ax_bridge.set_xticks(positions, labels=impact[component], rotation=25, ha="right")
        ax_bridge.set_ylabel("Illustrative value under stated assumptions")
        ax_bridge.set_title("Impact bridge", loc="left", fontsize=10)
        style_axis(ax_bridge)
        for x_pos, start, increment in zip(positions, starts, values, strict=True):
            ax_bridge.text(
                x_pos,
                start + increment / 2,
                _metric_text(increment, 1),
                ha="center",
                va="center",
                fontsize=7.5,
                color=PALETTE["white"] if abs(increment) > 0 else PALETTE["ink"],
            )

        status_order = {"passed": 2, "review": 1, "blocked": 0}
        status_colors = {"passed": PALETTE["teal"], "review": PALETTE["amber"], "blocked": PALETTE["vermillion"]}
        evidence_data = evidence.copy()
        evidence_data["_status"] = evidence_data[evidence_status].astype(str).str.lower()
        unknown = sorted(set(evidence_data["_status"]) - set(status_order))
        if unknown:
            raise ValueError(f"Unknown evidence statuses: {unknown}")
        y = np.arange(len(evidence_data))[::-1]
        ax_evidence.hlines(y, 0, 1, color=PALETTE["grid"], linewidth=1.2)
        for y_pos, (_, row) in zip(y, evidence_data.iterrows(), strict=True):
            status_value = row["_status"]
            ax_evidence.scatter(0.93, y_pos, s=90, color=status_colors[status_value])
            ax_evidence.text(0.02, y_pos, str(row[evidence_stage]), va="center", fontsize=8.5)
            ax_evidence.text(0.88, y_pos, status_value.upper(), va="center", ha="right", fontsize=7, color=PALETTE["muted"])
        ax_evidence.set_xlim(0, 1)
        ax_evidence.set_ylim(-0.7, len(evidence_data) - 0.3)
        ax_evidence.set_title("Evidence and approval state", loc="left", fontsize=10)
        ax_evidence.axis("off")
        fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.17)
        return save_publication_figure(
            fig,
            output_path,
            figure_id="Figure 5",
            source=source,
            sample=sample,
        )
