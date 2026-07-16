from __future__ import annotations

import numpy as np
import pandas as pd


class MetricContractError(RuntimeError):
    """Raised when evaluation inputs violate the metric contract."""


def _aligned_binary_inputs(
    actual: pd.Series,
    probability: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    valid = actual.notna() & probability.notna()
    observed = actual.loc[valid].astype(int)
    forecast = probability.loc[valid].astype(float)

    if observed.empty:
        raise MetricContractError("No valid classification observations")

    if not observed.isin([0, 1]).all():
        raise MetricContractError("Classification outcomes must be binary")

    if not np.isfinite(forecast.to_numpy()).all():
        raise MetricContractError("Probabilities must be finite")

    if not forecast.between(0.0, 1.0).all():
        raise MetricContractError("Probabilities must lie within [0, 1]")

    return observed, forecast


def controlled_alert_metrics(
    actual: pd.Series,
    probability: pd.Series,
    *,
    alert_rate: float,
) -> dict[str, float]:
    """Compute exact-rate alert metrics with fractional cutoff ties."""
    if not 0.0 < alert_rate <= 1.0:
        raise MetricContractError("Alert rate must lie within (0, 1]")

    observed, forecast = _aligned_binary_inputs(actual, probability)
    positive_count = int(observed.sum())

    if positive_count == 0:
        raise MetricContractError("Alert recall requires positive outcomes")

    observation_count = len(observed)
    alert_mass = float(alert_rate * observation_count)

    if alert_mass >= observation_count:
        threshold = float(forecast.min())
        weights = pd.Series(1.0, index=forecast.index)
        tie_weight = 1.0
    else:
        ordered = np.sort(forecast.to_numpy())[::-1]
        cutoff_position = max(int(np.ceil(alert_mass)) - 1, 0)
        threshold = float(ordered[cutoff_position])

        above_cutoff = forecast.gt(threshold)
        at_cutoff = forecast.eq(threshold)
        remaining_mass = alert_mass - float(above_cutoff.sum())
        cutoff_count = int(at_cutoff.sum())

        if cutoff_count <= 0:
            raise MetricContractError("Alert cutoff tie set is empty")

        if remaining_mass < -1e-12 or remaining_mass > cutoff_count + 1e-12:
            raise MetricContractError("Alert cutoff allocation is invalid")

        tie_weight = float(
            min(max(remaining_mass / cutoff_count, 0.0), 1.0)
        )
        weights = pd.Series(0.0, index=forecast.index)
        weights.loc[above_cutoff] = 1.0
        weights.loc[at_cutoff] = tie_weight

    realized_mass = float(weights.sum())
    weighted_true_positives = float((weights * observed).sum())

    return {
        "controlled_alert_rate": float(alert_rate),
        "controlled_alert_rate_realized": float(
            realized_mass / observation_count
        ),
        "controlled_alert_count_equivalent": realized_mass,
        "controlled_alert_precision": float(
            weighted_true_positives / realized_mass
        ),
        "controlled_alert_recall": float(
            weighted_true_positives / positive_count
        ),
        "alert_probability_threshold": threshold,
        "alert_cutoff_tie_weight": tie_weight,
    }


def expected_calibration_error(
    actual: pd.Series,
    probability: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    if bins < 2:
        raise MetricContractError("Calibration requires at least two bins")

    observed, forecast = _aligned_binary_inputs(actual, probability)
    observed_array = observed.to_numpy(dtype=float)
    forecast_array = forecast.to_numpy(dtype=float)
    bin_index = np.minimum(
        (forecast_array * bins).astype(int),
        bins - 1,
    )

    error = 0.0
    observation_count = len(observed_array)

    for current_bin in range(bins):
        members = bin_index == current_bin

        if not members.any():
            continue

        weight = float(members.sum() / observation_count)
        observed_rate = float(observed_array[members].mean())
        predicted_rate = float(forecast_array[members].mean())
        error += weight * abs(observed_rate - predicted_rate)

    return float(error)
