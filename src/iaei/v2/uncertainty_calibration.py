from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class UncertaintyCalibrationError(RuntimeError):
    """Raised when governed uncertainty calibration is invalid."""


@dataclass(frozen=True)
class UncertaintyConfiguration:
    configuration_id: str
    method_id: str
    role: str
    window_intervals: int | None
    adaptive_gamma: float | None
    adaptive_alpha_lower_bound: float | None
    adaptive_alpha_upper_bound: float | None


def configurations_from_contract(
    contract: dict[str, Any],
) -> list[UncertaintyConfiguration]:
    configurations: list[UncertaintyConfiguration] = []
    observed: set[str] = set()
    for family in contract["candidate_families"]:
        method_id = str(family["method_id"])
        role = str(family["role"])
        lower = family.get("adaptive_alpha_lower_bound")
        upper = family.get("adaptive_alpha_upper_bound")
        for raw in family["configurations"]:
            configuration_id = str(raw["configuration_id"])
            if configuration_id in observed:
                raise UncertaintyCalibrationError(
                    f"Duplicate uncertainty configuration: {configuration_id}"
                )
            observed.add(configuration_id)
            window = raw.get("window_intervals")
            gamma = raw.get("adaptive_gamma")
            configurations.append(
                UncertaintyConfiguration(
                    configuration_id=configuration_id,
                    method_id=method_id,
                    role=role,
                    window_intervals=(
                        None if window is None else int(window)
                    ),
                    adaptive_gamma=(
                        None if gamma is None else float(gamma)
                    ),
                    adaptive_alpha_lower_bound=(
                        None if lower is None else float(lower)
                    ),
                    adaptive_alpha_upper_bound=(
                        None if upper is None else float(upper)
                    ),
                )
            )
    expected = int(
        contract["execution_budget"]["unique_configuration_count"]
    )
    if len(configurations) != expected:
        raise UncertaintyCalibrationError(
            f"Expected {expected} configurations, "
            f"observed {len(configurations)}"
        )
    return configurations


def finite_sample_higher_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise UncertaintyCalibrationError(
            "Calibration scores must be a nonempty vector"
        )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise UncertaintyCalibrationError(
            "Calibration scores must be finite and nonnegative"
        )
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise UncertaintyCalibrationError(
            "Alpha must lie strictly between zero and one"
        )
    rank = int(np.ceil((values.size + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def interval_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise UncertaintyCalibrationError(
            "Interval-score inputs have inconsistent shapes"
        )
    if (
        not np.isfinite(y).all()
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
    ):
        raise UncertaintyCalibrationError(
            "Interval-score inputs must be finite"
        )
    if (hi < lo).any():
        raise UncertaintyCalibrationError(
            "Interval upper bound is below lower bound"
        )
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return (
        (hi - lo)
        + (2.0 / alpha) * below
        + (2.0 / alpha) * above
    )


def weighted_interval_score(
    actual: np.ndarray,
    point: np.ndarray,
    intervals: dict[float, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    median = np.asarray(point, dtype=float)
    if y.shape != median.shape:
        raise UncertaintyCalibrationError(
            "Point and actual vectors have inconsistent shapes"
        )
    weighted = 0.5 * np.abs(y - median)
    weight_sum = 0.5
    for coverage in sorted(intervals):
        alpha = 1.0 - float(coverage)
        lower, upper = intervals[coverage]
        weight = alpha / 2.0
        weighted = weighted + weight * interval_score(
            y,
            lower,
            upper,
            alpha,
        )
        weight_sum += weight
    return weighted / weight_sum


def longest_miss_run(covered: np.ndarray) -> int:
    values = np.asarray(covered, dtype=bool)
    longest = 0
    current = 0
    for hit in values:
        if hit:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def maximum_rolling_coverage_error(
    covered: np.ndarray,
    nominal_coverage: float,
    window: int,
) -> float:
    values = pd.Series(np.asarray(covered, dtype=float))
    if window < 1:
        raise UncertaintyCalibrationError(
            "Rolling coverage window must be positive"
        )
    if len(values) < window:
        return float(abs(values.mean() - nominal_coverage))
    rolling = values.rolling(
        window=window,
        min_periods=window,
    ).mean()
    return float((rolling - nominal_coverage).abs().max())


def evaluate_intervals(
    point_predictions: pd.DataFrame,
    initial_scores_by_fold: dict[int, np.ndarray],
    configuration: UncertaintyConfiguration,
    coverage_levels: list[float],
    *,
    lower_support_bound: float,
) -> pd.DataFrame:
    required = {
        "fold_id",
        "row_position",
        "prediction_origin",
        "actual",
        "prediction",
        "is_peak_state",
    }
    missing = sorted(required.difference(point_predictions.columns))
    if missing:
        raise UncertaintyCalibrationError(
            f"Point predictions are missing columns: {missing}"
        )
    if not coverage_levels:
        raise UncertaintyCalibrationError(
            "Coverage level collection is empty"
        )

    frames: list[pd.DataFrame] = []
    for fold_id, fold_frame in point_predictions.groupby(
        "fold_id",
        sort=True,
    ):
        fold = fold_frame.sort_values(
            "row_position",
            kind="stable",
        ).reset_index(drop=True)
        initial = np.asarray(
            initial_scores_by_fold[int(fold_id)],
            dtype=float,
        )
        if initial.size == 0:
            raise UncertaintyCalibrationError(
                f"Fold {fold_id} has no calibration scores"
            )
        pools: dict[float, list[float]] = {
            float(level): initial.astype(float).tolist()
            for level in coverage_levels
        }
        alpha_state = {
            float(level): 1.0 - float(level)
            for level in coverage_levels
        }
        records: list[dict[str, Any]] = []

        for row in fold.itertuples(index=False):
            point = float(row.prediction)
            actual = float(row.actual)
            absolute_residual = abs(actual - point)
            record: dict[str, Any] = {
                "configuration_id": configuration.configuration_id,
                "method_id": configuration.method_id,
                "fold_id": int(row.fold_id),
                "row_position": int(row.row_position),
                "prediction_origin": pd.Timestamp(
                    row.prediction_origin
                ),
                "actual": actual,
                "point_prediction": point,
                "absolute_residual": absolute_residual,
                "is_peak_state": bool(row.is_peak_state),
            }
            misses: dict[float, int] = {}
            for level in coverage_levels:
                coverage = float(level)
                nominal_alpha = 1.0 - coverage
                current_alpha = (
                    alpha_state[coverage]
                    if configuration.adaptive_gamma is not None
                    else nominal_alpha
                )
                pool = pools[coverage]
                if configuration.window_intervals is None:
                    selected_scores = np.asarray(
                        pool,
                        dtype=float,
                    )
                else:
                    selected_scores = np.asarray(
                        pool[-configuration.window_intervals :],
                        dtype=float,
                    )
                quantile = finite_sample_higher_quantile(
                    selected_scores,
                    current_alpha,
                )
                raw_lower = point - quantile
                raw_upper = point + quantile
                lower = max(raw_lower, lower_support_bound)
                upper = raw_upper
                covered = bool(lower <= actual <= upper)
                suffix = str(int(round(coverage * 100)))
                record[f"alpha_state_{suffix}"] = float(
                    current_alpha
                )
                record[f"calibration_count_{suffix}"] = int(
                    selected_scores.size
                )
                record[f"quantile_{suffix}"] = float(quantile)
                record[f"lower_raw_{suffix}"] = float(raw_lower)
                record[f"upper_raw_{suffix}"] = float(raw_upper)
                record[f"lower_{suffix}"] = float(lower)
                record[f"upper_{suffix}"] = float(upper)
                record[f"covered_{suffix}"] = covered
                record[
                    f"support_floor_activated_{suffix}"
                ] = bool(lower != raw_lower)
                misses[coverage] = int(not covered)

            records.append(record)
            for coverage in coverage_levels:
                coverage_value = float(coverage)
                pools[coverage_value].append(
                    float(absolute_residual)
                )
                if configuration.adaptive_gamma is not None:
                    lower_alpha = float(
                        configuration.adaptive_alpha_lower_bound
                    )
                    upper_alpha = float(
                        configuration.adaptive_alpha_upper_bound
                    )
                    nominal_alpha = 1.0 - coverage_value
                    updated = (
                        alpha_state[coverage_value]
                        + float(configuration.adaptive_gamma)
                        * (
                            nominal_alpha
                            - misses[coverage_value]
                        )
                    )
                    alpha_state[coverage_value] = float(
                        np.clip(
                            updated,
                            lower_alpha,
                            upper_alpha,
                        )
                    )
        frames.append(pd.DataFrame.from_records(records))

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_configuration(
    predictions: pd.DataFrame,
    coverage_levels: list[float],
    training_iqr_by_fold: dict[int, float],
    *,
    primary_coverage: float,
    rolling_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    coverage_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    primary_suffix = str(int(round(primary_coverage * 100)))

    for fold_id, fold in predictions.groupby("fold_id", sort=True):
        fold = fold.sort_values("row_position", kind="stable")
        actual = fold["actual"].to_numpy(dtype=float)
        point = fold["point_prediction"].to_numpy(dtype=float)
        peak = fold["is_peak_state"].to_numpy(dtype=bool)
        intervals: dict[
            float,
            tuple[np.ndarray, np.ndarray],
        ] = {}
        peak_intervals: dict[
            float,
            tuple[np.ndarray, np.ndarray],
        ] = {}
        for level in coverage_levels:
            coverage = float(level)
            suffix = str(int(round(coverage * 100)))
            lower = fold[f"lower_{suffix}"].to_numpy(
                dtype=float
            )
            upper = fold[f"upper_{suffix}"].to_numpy(
                dtype=float
            )
            covered = fold[f"covered_{suffix}"].to_numpy(
                dtype=bool
            )
            scores = interval_score(
                actual,
                lower,
                upper,
                1.0 - coverage,
            )
            intervals[coverage] = (lower, upper)
            peak_lower = lower[peak]
            peak_upper = upper[peak]
            peak_actual = actual[peak]
            peak_scores = interval_score(
                peak_actual,
                peak_lower,
                peak_upper,
                1.0 - coverage,
            )
            width = upper - lower
            iqr = float(training_iqr_by_fold[int(fold_id)])
            if not np.isfinite(iqr) or iqr <= 0.0:
                raise UncertaintyCalibrationError(
                    f"Fold {fold_id} has an invalid "
                    "training-target IQR"
                )
            coverage_rows.append(
                {
                    "configuration_id": str(
                        fold["configuration_id"].iloc[0]
                    ),
                    "method_id": str(
                        fold["method_id"].iloc[0]
                    ),
                    "fold_id": int(fold_id),
                    "coverage_level": coverage,
                    "empirical_coverage": float(
                        covered.mean()
                    ),
                    "absolute_coverage_error": float(
                        abs(covered.mean() - coverage)
                    ),
                    "mean_interval_width_kwh": float(
                        width.mean()
                    ),
                    "normalized_mean_interval_width": float(
                        width.mean() / iqr
                    ),
                    "interval_score": float(scores.mean()),
                    "peak_state_coverage": float(
                        covered[peak].mean()
                    ),
                    "peak_state_mean_interval_width_kwh": (
                        float(width[peak].mean())
                    ),
                    "peak_state_interval_score": float(
                        peak_scores.mean()
                    ),
                    "origin_count": int(len(fold)),
                    "peak_origin_count": int(peak.sum()),
                }
            )
            peak_intervals[coverage] = (
                peak_lower,
                peak_upper,
            )
        wis = weighted_interval_score(
            actual,
            point,
            intervals,
        )
        peak_wis = weighted_interval_score(
            actual[peak],
            point[peak],
            peak_intervals,
        )
        primary_covered = fold[
            f"covered_{primary_suffix}"
        ].to_numpy(dtype=bool)
        fold_rows.append(
            {
                "configuration_id": str(
                    fold["configuration_id"].iloc[0]
                ),
                "method_id": str(
                    fold["method_id"].iloc[0]
                ),
                "fold_id": int(fold_id),
                "weighted_interval_score": float(
                    wis.mean()
                ),
                "peak_state_weighted_interval_score": float(
                    peak_wis.mean()
                ),
                "primary_empirical_coverage": float(
                    primary_covered.mean()
                ),
                "primary_absolute_coverage_error": float(
                    abs(
                        primary_covered.mean()
                        - primary_coverage
                    )
                ),
                "primary_mean_interval_width_kwh": float(
                    (
                        fold[f"upper_{primary_suffix}"]
                        - fold[f"lower_{primary_suffix}"]
                    ).mean()
                ),
                "maximum_rolling_672_absolute_coverage_error": (
                    maximum_rolling_coverage_error(
                        primary_covered,
                        primary_coverage,
                        rolling_window,
                    )
                ),
                "longest_consecutive_miss_run": (
                    longest_miss_run(primary_covered)
                ),
                "origin_count": int(len(fold)),
                "peak_origin_count": int(peak.sum()),
            }
        )

    coverage_frame = pd.DataFrame(coverage_rows)
    fold_frame = pd.DataFrame(fold_rows)
    actual = predictions["actual"].to_numpy(dtype=float)
    point = predictions["point_prediction"].to_numpy(
        dtype=float
    )
    aggregate_intervals: dict[
        float,
        tuple[np.ndarray, np.ndarray],
    ] = {}
    aggregate_peak_intervals: dict[
        float,
        tuple[np.ndarray, np.ndarray],
    ] = {}
    peak = predictions["is_peak_state"].to_numpy(dtype=bool)
    for level in coverage_levels:
        suffix = str(int(round(float(level) * 100)))
        lower = predictions[f"lower_{suffix}"].to_numpy(
            dtype=float
        )
        upper = predictions[f"upper_{suffix}"].to_numpy(
            dtype=float
        )
        aggregate_intervals[float(level)] = (
            lower,
            upper,
        )
        aggregate_peak_intervals[float(level)] = (
            lower[peak],
            upper[peak],
        )
    aggregate_wis = weighted_interval_score(
        actual,
        point,
        aggregate_intervals,
    )
    aggregate_peak_wis = weighted_interval_score(
        actual[peak],
        point[peak],
        aggregate_peak_intervals,
    )
    nested = np.ones(len(predictions), dtype=bool)
    ordered_levels = sorted(
        float(level) for level in coverage_levels
    )
    for narrower, wider in zip(
        ordered_levels[:-1],
        ordered_levels[1:],
    ):
        narrow_suffix = str(int(round(narrower * 100)))
        wide_suffix = str(int(round(wider * 100)))
        nested &= (
            predictions[f"lower_{wide_suffix}"].to_numpy(
                dtype=float
            )
            <= predictions[f"lower_{narrow_suffix}"].to_numpy(
                dtype=float
            )
        ) & (
            predictions[f"upper_{wide_suffix}"].to_numpy(
                dtype=float
            )
            >= predictions[f"upper_{narrow_suffix}"].to_numpy(
                dtype=float
            )
        )
    support_activations = np.column_stack(
        [
            predictions[
                "support_floor_activated_"
                f"{int(round(float(level) * 100))}"
            ].to_numpy(dtype=bool)
            for level in coverage_levels
        ]
    )
    primary_coverage_rows = coverage_frame.loc[
        coverage_frame["coverage_level"].eq(
            primary_coverage
        )
    ]
    aggregate: dict[str, float | int] = {
        "weighted_interval_score": float(
            aggregate_wis.mean()
        ),
        "peak_state_weighted_interval_score": float(
            aggregate_peak_wis.mean()
        ),
        "maximum_outer_fold_absolute_coverage_error": float(
            coverage_frame["absolute_coverage_error"].max()
        ),
        "maximum_outer_fold_90_absolute_coverage_error": float(
            primary_coverage_rows[
                "absolute_coverage_error"
            ].max()
        ),
        "maximum_rolling_672_absolute_coverage_error": float(
            fold_frame[
                "maximum_rolling_672_absolute_coverage_error"
            ].max()
        ),
        "longest_consecutive_miss_run": int(
            fold_frame[
                "longest_consecutive_miss_run"
            ].max()
        ),
        "interval_nesting_violation_count": int(
            (~nested).sum()
        ),
        "support_floor_activation_rate": float(
            support_activations.mean()
        ),
        "origin_count": int(len(predictions)),
        "peak_origin_count": int(peak.sum()),
    }
    return coverage_frame, fold_frame, aggregate
