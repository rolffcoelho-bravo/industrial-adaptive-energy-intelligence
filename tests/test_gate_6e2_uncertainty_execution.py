from __future__ import annotations

import json

import numpy as np
import pandas as pd

from iaei.contracts import load_yaml
from iaei.paths import CONFIGS
from iaei.v2.uncertainty_calibration import (
    UncertaintyConfiguration,
    configurations_from_contract,
    evaluate_intervals,
    finite_sample_higher_quantile,
    interval_score,
    summarize_configuration,
    weighted_interval_score,
)
from iaei.v2.uncertainty_evidence import _recommendation


def _point_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": [1, 1, 1],
            "row_position": [10, 11, 12],
            "prediction_origin": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="15min",
            ),
            "actual": [10.0, 0.5, 1.0],
            "prediction": [0.0, 1.0, 1.0],
            "is_peak_state": [True, False, False],
        }
    )


def test_contract_freezes_nine_configurations() -> None:
    contract = load_yaml(CONFIGS / "uncertainty_contract.yml")
    configurations = configurations_from_contract(contract)
    assert len(configurations) == 9
    assert configurations[0].configuration_id == "expanding_all"
    assert configurations[-1].configuration_id == "aci_2688_0p02"


def test_finite_sample_higher_quantile_uses_locked_rank() -> None:
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    assert finite_sample_higher_quantile(scores, 0.20) == 4.0


def test_interval_score_penalizes_misses() -> None:
    actual = np.array([0.0, 3.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([2.0, 2.0])
    observed = interval_score(actual, lower, upper, 0.20)
    np.testing.assert_allclose(observed, np.array([2.0, 12.0]))


def test_weighted_interval_score_uses_standard_denominator() -> None:
    actual = np.array([3.0])
    point = np.array([2.0])
    lower = np.array([1.0])
    upper = np.array([2.5])
    observed = weighted_interval_score(
        actual,
        point,
        {0.80: (lower, upper)},
    )
    interval_component = interval_score(actual, lower, upper, 0.20)
    expected = (0.5 * np.abs(actual - point) + 0.10 * interval_component) / 1.5
    np.testing.assert_allclose(observed, expected)


def test_prequential_expanding_update_and_support_floor() -> None:
    configuration = UncertaintyConfiguration(
        configuration_id="expanding_all",
        method_id="expanding_absolute_conformal",
        role="reference",
        window_intervals=None,
        adaptive_gamma=None,
        adaptive_alpha_lower_bound=None,
        adaptive_alpha_upper_bound=None,
    )
    evaluated = evaluate_intervals(
        _point_predictions(),
        {1: np.ones(4, dtype=float)},
        configuration,
        [0.80],
        lower_support_bound=0.0,
    )
    assert evaluated.loc[0, "quantile_80"] == 1.0
    assert evaluated.loc[1, "quantile_80"] == 10.0
    assert evaluated["lower_80"].ge(0.0).all()
    np.testing.assert_array_equal(
        evaluated["point_prediction"].to_numpy(),
        _point_predictions()["prediction"].to_numpy(),
    )


def test_adaptive_configuration_is_exactly_replayable() -> None:
    configuration = UncertaintyConfiguration(
        configuration_id="aci_672_0p01",
        method_id="adaptive_absolute_conformal",
        role="challenger",
        window_intervals=672,
        adaptive_gamma=0.01,
        adaptive_alpha_lower_bound=0.001,
        adaptive_alpha_upper_bound=0.5,
    )
    first = evaluate_intervals(
        _point_predictions(),
        {1: np.array([0.5, 1.0, 1.5, 2.0])},
        configuration,
        [0.80, 0.90, 0.95],
        lower_support_bound=0.0,
    )
    second = evaluate_intervals(
        _point_predictions(),
        {1: np.array([0.5, 1.0, 1.5, 2.0])},
        configuration,
        [0.80, 0.90, 0.95],
        lower_support_bound=0.0,
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_summary_records_nested_intervals_and_weighted_scores() -> None:
    configuration = UncertaintyConfiguration(
        configuration_id="expanding_all",
        method_id="expanding_absolute_conformal",
        role="reference",
        window_intervals=None,
        adaptive_gamma=None,
        adaptive_alpha_lower_bound=None,
        adaptive_alpha_upper_bound=None,
    )
    evaluated = evaluate_intervals(
        _point_predictions(),
        {1: np.array([0.5, 1.0, 2.0, 4.0])},
        configuration,
        [0.80, 0.90, 0.95],
        lower_support_bound=0.0,
    )
    coverage, folds, aggregate = summarize_configuration(
        evaluated,
        [0.80, 0.90, 0.95],
        {1: 2.0},
        primary_coverage=0.90,
        rolling_window=2,
    )
    assert len(coverage) == 3
    assert len(folds) == 1
    assert aggregate["interval_nesting_violation_count"] == 0
    assert float(aggregate["weighted_interval_score"]) > 0.0


def test_recommendation_allows_governed_no_action() -> None:
    configurations = pd.DataFrame(
        {
            "configuration_id": ["expanding_all", "rolling_672"],
            "method_id": [
                "expanding_absolute_conformal",
                "rolling_absolute_conformal",
            ],
            "weighted_interval_score": [1.0, 0.9],
            "peak_state_weighted_interval_score": [1.0, 0.9],
            "mean_interval_width_kwh_at_90": [2.0, 1.8],
            "maximum_outer_fold_absolute_coverage_error": [0.2, 0.2],
            "maximum_rolling_672_absolute_coverage_error": [0.2, 0.2],
            "p95_latency_ms_per_1000_rows": [1.0, 1.0],
            "hard_constraints_passed": [False, False],
            "failed_hard_constraints": [
                json.dumps(["aggregate_90_absolute_coverage_error"]),
                json.dumps(["aggregate_90_absolute_coverage_error"]),
            ],
        }
    )
    folds = pd.DataFrame(
        {
            "configuration_id": [
                "expanding_all",
                "expanding_all",
                "rolling_672",
                "rolling_672",
            ],
            "fold_id": [1, 2, 1, 2],
            "weighted_interval_score": [1.0, 1.0, 0.9, 0.9],
        }
    )
    contract = load_yaml(CONFIGS / "uncertainty_contract.yml")
    recommendation = _recommendation(configurations, folds, contract)
    assert recommendation["outcome"] == "no_action"
    assert recommendation["recommended_configuration"] is None
    assert recommendation["next_gate_authorized"] is False
