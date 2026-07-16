from __future__ import annotations

import pandas as pd
import pytest

from iaei.modeling.metrics import (
    MetricContractError,
    controlled_alert_metrics,
    expected_calibration_error,
)


def test_constant_probability_uses_fractional_tie_allocation() -> None:
    actual = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    probability = pd.Series([0.25] * len(actual))

    metrics = controlled_alert_metrics(
        actual,
        probability,
        alert_rate=0.25,
    )

    assert metrics["controlled_alert_rate_realized"] == pytest.approx(0.25)
    assert metrics["controlled_alert_count_equivalent"] == pytest.approx(2.0)
    assert metrics["controlled_alert_precision"] == pytest.approx(0.5)
    assert metrics["controlled_alert_recall"] == pytest.approx(0.25)
    assert metrics["alert_cutoff_tie_weight"] == pytest.approx(0.25)


def test_ranked_probabilities_select_the_strongest_alerts() -> None:
    actual = pd.Series([0, 1, 1, 0])
    probability = pd.Series([0.1, 0.9, 0.8, 0.2])

    metrics = controlled_alert_metrics(
        actual,
        probability,
        alert_rate=0.5,
    )

    assert metrics["controlled_alert_precision"] == pytest.approx(1.0)
    assert metrics["controlled_alert_recall"] == pytest.approx(1.0)
    assert metrics["alert_probability_threshold"] == pytest.approx(0.8)


def test_perfect_probabilities_have_zero_calibration_error() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(actual, probability) == pytest.approx(0.0)


def test_constant_probability_calibration_gap_is_exact() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probability = pd.Series([0.25, 0.25, 0.25, 0.25])

    assert expected_calibration_error(actual, probability) == pytest.approx(0.25)


@pytest.mark.parametrize("alert_rate", [0.0, -0.1, 1.1])
def test_invalid_alert_rates_are_rejected(alert_rate: float) -> None:
    actual = pd.Series([0, 1])
    probability = pd.Series([0.2, 0.8])

    with pytest.raises(MetricContractError, match="Alert rate"):
        controlled_alert_metrics(
            actual,
            probability,
            alert_rate=alert_rate,
        )
