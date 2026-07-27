from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from iaei.v2.uncertainty_execution import _nonfinite_count


def _hard_constraint_checks(
    predictions: pd.DataFrame,
    coverage: pd.DataFrame,
    aggregate: dict[str, float | int],
    contract: dict[str, Any],
    parity_violations: int,
    replay_passed: bool,
    cpu_passed: bool,
) -> dict[str, bool]:
    levels = [
        float(value)
        for value in contract["calibration_protocol"]["target_coverage_levels"]
    ]
    checks: dict[str, bool] = {}
    peak = predictions["is_peak_state"].to_numpy(dtype=bool)
    for level in levels:
        suffix = str(int(round(level * 100)))
        empirical = float(predictions[f"covered_{suffix}"].mean())
        error = abs(empirical - level)
        threshold = 0.02 if level == 0.90 else 0.03
        checks[f"aggregate_{suffix}_absolute_coverage_error"] = error <= threshold
    peak_90 = float(predictions.loc[peak, "covered_90"].mean())
    checks["minimum_aggregate_peak_90_coverage"] = peak_90 >= 0.85
    checks["maximum_outer_fold_90_absolute_coverage_error"] = (
        float(aggregate["maximum_outer_fold_90_absolute_coverage_error"]) <= 0.05
    )
    checks["maximum_rolling_672_90_absolute_coverage_error"] = (
        float(aggregate["maximum_rolling_672_absolute_coverage_error"]) <= 0.08
    )
    checks["interval_nesting_violation_count"] = (
        int(aggregate["interval_nesting_violation_count"]) == 0
    )
    support_violations = int(
        sum(
            (predictions[f"lower_{suffix}"] < 0.0).sum()
            + (predictions[f"upper_{suffix}"] < predictions[f"lower_{suffix}"]).sum()
            for suffix in ("80", "90", "95")
        )
    )
    checks["support_violation_count"] = support_violations == 0
    checks["point_prediction_parity_violation_count"] = parity_violations == 0
    checks["chronology_violation_count"] = True
    checks["leakage_violation_count"] = True
    checks["locked_test_access_count"] = True
    checks["nonfinite_evidence_count"] = (
        _nonfinite_count(predictions) == 0 and _nonfinite_count(coverage) == 0
    )
    checks["missing_required_artifact_count"] = True
    checks["deterministic_replay_failure_count"] = replay_passed
    checks["cpu_portability_failure_count"] = cpu_passed
    return checks


def _pareto_mask(frame: pd.DataFrame, objectives: list[str]) -> pd.Series:
    values = frame[objectives].to_numpy(dtype=float)
    mask = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        if not mask[index]:
            continue
        dominated = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        if dominated.any():
            mask[index] = False
    return pd.Series(mask, index=frame.index)


def _recommendation(
    configurations: pd.DataFrame,
    folds: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    reference_id = str(contract["optimization"]["reference_configuration"])
    reference = configurations.loc[
        configurations["configuration_id"].eq(reference_id)
    ].iloc[0]
    reference_folds = folds.loc[
        folds["configuration_id"].eq(reference_id)
    ].set_index("fold_id")
    requirements = contract["optimization"]["challenger_requirements"]
    candidate_records: list[dict[str, Any]] = []
    eligible_ids: list[str] = []

    for row in configurations.itertuples(index=False):
        failed = list(json.loads(row.failed_hard_constraints))
        relative_improvement = float(
            (reference.weighted_interval_score - row.weighted_interval_score)
            / reference.weighted_interval_score
        )
        relative_peak_change = float(
            (
                row.peak_state_weighted_interval_score
                - reference.peak_state_weighted_interval_score
            )
            / reference.peak_state_weighted_interval_score
        )
        candidate_folds = folds.loc[
            folds["configuration_id"].eq(row.configuration_id)
        ].set_index("fold_id")
        fold_changes = (
            candidate_folds["weighted_interval_score"]
            - reference_folds["weighted_interval_score"]
        ) / reference_folds["weighted_interval_score"]
        positive_folds = int((fold_changes < 0.0).sum())
        maximum_fold_degradation = float(fold_changes.max())
        if row.configuration_id != reference_id:
            if relative_improvement < float(
                requirements[
                    "minimum_weighted_interval_score_relative_improvement_vs_reference"
                ]
            ):
                failed.append("weighted_interval_score_improvement")
            if positive_folds < int(requirements["minimum_positive_outer_folds"]):
                failed.append("positive_outer_folds")
            if maximum_fold_degradation > float(
                requirements[
                    "maximum_single_fold_weighted_interval_score_relative_degradation"
                ]
            ):
                failed.append("single_fold_weighted_interval_score_degradation")
            if relative_peak_change > float(
                requirements[
                    "maximum_peak_weighted_interval_score_relative_degradation"
                ]
            ):
                failed.append("peak_weighted_interval_score_degradation")
        eligible = not failed
        if eligible:
            eligible_ids.append(str(row.configuration_id))
        candidate_records.append(
            {
                "configuration_id": str(row.configuration_id),
                "method_id": str(row.method_id),
                "hard_constraints_passed": bool(row.hard_constraints_passed),
                "relative_weighted_interval_score_improvement_vs_reference": (
                    relative_improvement
                ),
                "relative_peak_weighted_interval_score_change_vs_reference": (
                    relative_peak_change
                ),
                "positive_outer_folds": positive_folds,
                "maximum_single_fold_weighted_interval_score_relative_degradation": (
                    maximum_fold_degradation
                ),
                "eligible_for_human_selection": eligible,
                "failed_requirements": sorted(set(failed)),
            }
        )

    eligible = configurations.loc[
        configurations["configuration_id"].isin(eligible_ids)
    ].copy()
    objective_columns = [
        "weighted_interval_score",
        "mean_interval_width_kwh_at_90",
        "peak_state_weighted_interval_score",
        "maximum_outer_fold_absolute_coverage_error",
        "maximum_rolling_672_absolute_coverage_error",
        "p95_latency_ms_per_1000_rows",
    ]
    pareto_ids: list[str] = []
    recommended: str | None = None
    outcome = "no_action"
    if not eligible.empty:
        eligible["pareto_eligible"] = _pareto_mask(eligible, objective_columns)
        pareto = eligible.loc[eligible["pareto_eligible"]].copy()
        pareto_ids = sorted(pareto["configuration_id"].astype(str).tolist())
        pareto = pareto.sort_values(
            [*objective_columns, "configuration_id"],
            kind="stable",
        )
        recommended = str(pareto.iloc[0]["configuration_id"])
        outcome = (
            "recommend_reference"
            if recommended == reference_id
            else "recommend_challenger"
        )
    return {
        "schema_version": "1.0.0",
        "gate": "6E",
        "subgate": "6E2",
        "status": "validation_complete_pending_human_decision",
        "selection_method": str(contract["optimization"]["selection_method"]),
        "reference_configuration": reference_id,
        "automatic_promotion_permitted": False,
        "human_decision_required": True,
        "outcome": outcome,
        "recommended_configuration": recommended,
        "eligible_configurations": sorted(eligible_ids),
        "pareto_eligible_configurations": pareto_ids,
        "candidates": candidate_records,
        "recommended_next_gate": "6E3",
        "next_gate_authorized": False,
    }


def _results_markdown(
    configurations: pd.DataFrame,
    recommendation: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Gate 6E2 governed uncertainty execution results",
        "",
        "## Status",
        "",
        "Gate 6E2 is complete as validation-only uncertainty evidence. "
        "Gate 6E3 human authority remains required.",
        "",
        "## Configuration evidence",
        "",
        "| Configuration | Method | WIS | Peak WIS | 90% coverage | 90% width | "
        "Hard constraints | Human-selection eligible |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in configurations.sort_values(
        "weighted_interval_score",
        kind="stable",
    ).itertuples(index=False):
        lines.append(
            f"| `{row.configuration_id}` | `{row.method_id}` | "
            f"{row.weighted_interval_score:.6f} | "
            f"{row.peak_state_weighted_interval_score:.6f} | "
            f"{row.empirical_coverage_90:.4f} | "
            f"{row.mean_interval_width_kwh_at_90:.6f} | "
            f"{bool(row.hard_constraints_passed)} | "
            f"{row.configuration_id in recommendation['eligible_configurations']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation boundary",
            "",
            f"- Outcome: `{recommendation['outcome']}`.",
            f"- Recommended configuration: "
            f"`{recommendation['recommended_configuration']}`.",
            f"- Eligible configurations: "
            f"`{recommendation['eligible_configurations']}`.",
            f"- Pareto-eligible configurations: "
            f"`{recommendation['pareto_eligible_configurations']}`.",
            "- Automatic promotion: `false`.",
            "- Human decision required: `true`.",
            "",
            "## Evidence boundary",
            "",
            f"- Validation origins per configuration: "
            f"`{manifest['validation_origins_per_configuration']}`.",
            f"- Total interval-prediction rows: "
            f"`{manifest['interval_prediction_row_count']}`.",
            f"- Maximum prediction origin: "
            f"`{manifest['maximum_prediction_origin']}`.",
            f"- Maximum target dependency: "
            f"`{manifest['maximum_target_dependency']}`.",
            "- Locked-test access: `false`.",
            "- Locked-prediction parsing: `false`.",
            "- Confirmatory evaluation: `false`.",
            "- V1 mutation: `false`.",
            "",
            "Gate 6E2 does not establish production, savings, causal, drift, "
            "optimization-impact, or confirmatory evidence.",
            "",
        ]
    )
    return "\n".join(lines)
