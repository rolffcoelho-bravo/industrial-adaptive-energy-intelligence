from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from iaei.contracts import validate_target_contract


class TargetConstructionError(RuntimeError):
    """Raised when supervised targets violate the locked temporal contract."""


@dataclass(frozen=True)
class TargetArtifacts:
    frame: pd.DataFrame
    peak_threshold_kwh: float


def _coerce_training_mask(
    training_mask: pd.Series,
    *,
    index: pd.Index,
) -> pd.Series:
    if not isinstance(training_mask, pd.Series):
        training_mask = pd.Series(training_mask, index=index)
    else:
        training_mask = training_mask.reindex(index)

    if training_mask.isna().any():
        raise TargetConstructionError("Training mask contains missing values")

    coerced = training_mask.astype(bool)
    if not coerced.any():
        raise TargetConstructionError("Training mask selects no observations")
    return coerced


def fit_peak_threshold(
    usage: pd.Series,
    training_mask: pd.Series,
    *,
    quantile: float = 0.90,
    interpolation: str = "linear",
) -> float:
    if not 0.0 < quantile < 1.0:
        raise TargetConstructionError("Peak quantile must be strictly between zero and one")

    numeric_usage = pd.to_numeric(usage, errors="raise")
    if numeric_usage.isna().any():
        raise TargetConstructionError("Usage contains missing values")
    if (numeric_usage < 0).any():
        raise TargetConstructionError("Usage contains negative values")

    mask = _coerce_training_mask(training_mask, index=numeric_usage.index)
    threshold = numeric_usage.loc[mask].quantile(
        quantile,
        interpolation=interpolation,
    )
    if pd.isna(threshold):
        raise TargetConstructionError("Training-only peak threshold could not be estimated")
    return float(threshold)


def training_mask_from_end(
    frame: pd.DataFrame,
    training_end: str | pd.Timestamp,
    *,
    timestamp_column: str = "effective_timestamp",
) -> pd.Series:
    if timestamp_column not in frame:
        raise TargetConstructionError(
            f"Timestamp column is missing from target frame: {timestamp_column}"
        )
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    cutoff = pd.Timestamp(training_end)
    return timestamps.le(cutoff)


def _validated_inputs(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    usage_column: str,
) -> tuple[pd.Series, pd.Series]:
    missing = [
        column
        for column in (timestamp_column, usage_column)
        if column not in frame.columns
    ]
    if missing:
        raise TargetConstructionError(f"Required target columns are missing: {missing}")

    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    usage = pd.to_numeric(frame[usage_column], errors="raise")

    if timestamps.isna().any():
        raise TargetConstructionError("Prediction-origin timestamps contain missing values")
    if timestamps.duplicated().any():
        raise TargetConstructionError("Prediction-origin timestamps contain duplicates")
    if not timestamps.is_monotonic_increasing:
        raise TargetConstructionError("Prediction origins must be chronologically ordered")
    if usage.isna().any():
        raise TargetConstructionError("Usage contains missing values")
    if (usage < 0).any():
        raise TargetConstructionError("Usage contains negative values")

    return timestamps, usage.astype(float)


def build_supervised_targets(
    frame: pd.DataFrame,
    training_mask: pd.Series,
    *,
    contract: dict[str, Any] | None = None,
) -> TargetArtifacts:
    locked_contract = contract or validate_target_contract()
    origin_contract = locked_contract["prediction_origin"]
    regression_contract = locked_contract["targets"]["regression"]
    peak_contract = locked_contract["targets"]["peak_risk"]

    timestamp_column = origin_contract["timestamp_column"]
    usage_column = regression_contract["source_column"]

    if peak_contract["source_column"] != usage_column:
        raise TargetConstructionError(
            "Regression and peak-risk targets must use the same usage source"
        )

    timestamps, usage = _validated_inputs(
        frame,
        timestamp_column=timestamp_column,
        usage_column=usage_column,
    )
    mask = _coerce_training_mask(training_mask, index=frame.index)

    threshold_contract = peak_contract["threshold"]
    threshold = fit_peak_threshold(
        usage,
        mask,
        quantile=float(threshold_contract["quantile"]),
        interpolation=str(threshold_contract["interpolation"]),
    )

    regression_steps = int(regression_contract["horizon_steps"])
    peak_steps = int(peak_contract["horizon_steps"])

    regression_name = str(regression_contract["name"])
    peak_name = str(peak_contract["name"])

    targets = pd.DataFrame(index=frame.index)
    targets["prediction_origin"] = timestamps
    targets[regression_name] = usage.shift(-regression_steps)
    targets[f"{regression_name}_timestamp"] = timestamps.shift(-regression_steps)

    future_usage = pd.concat(
        [usage.shift(-step) for step in range(1, peak_steps + 1)],
        axis=1,
    )
    complete_peak_window = future_usage.notna().all(axis=1)
    peak_label = future_usage.ge(threshold).any(axis=1).astype("Int64")
    peak_label = peak_label.where(complete_peak_window, pd.NA)

    targets[peak_name] = peak_label
    targets[f"{peak_name}_window_end"] = timestamps.shift(-peak_steps)
    targets["peak_threshold_kwh"] = threshold
    targets["target_contract_version"] = locked_contract["contract_version"]

    valid_regression = targets[f"{regression_name}_timestamp"].notna()
    if not (
        targets.loc[valid_regression, f"{regression_name}_timestamp"]
        > targets.loc[valid_regression, "prediction_origin"]
    ).all():
        raise TargetConstructionError(
            "Regression target timestamps must be strictly later than origins"
        )

    valid_peak = targets[f"{peak_name}_window_end"].notna()
    if not (
        targets.loc[valid_peak, f"{peak_name}_window_end"]
        > targets.loc[valid_peak, "prediction_origin"]
    ).all():
        raise TargetConstructionError(
            "Peak-window end timestamps must be strictly later than origins"
        )

    return TargetArtifacts(frame=targets, peak_threshold_kwh=threshold)
