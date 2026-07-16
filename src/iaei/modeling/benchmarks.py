from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from iaei.modeling.metrics import (
    controlled_alert_metrics,
    expected_calibration_error,
)
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class BenchmarkError(RuntimeError):
    """Raised when benchmark evidence violates the model contract."""


@dataclass(frozen=True)
class BenchmarkEvaluation:
    regression_results: pd.DataFrame
    classification_results: pd.DataFrame
    predictions: pd.DataFrame


def _regression_metrics(
    actual: pd.Series,
    prediction: pd.Series,
    *,
    peak_mask: pd.Series,
    mase_denominator: float,
    rolling_window: int,
) -> dict[str, float | int]:
    valid = actual.notna() & prediction.notna()
    observed = actual.loc[valid].astype(float)
    forecast = prediction.loc[valid].astype(float)

    if observed.empty:
        raise BenchmarkError("Regression benchmark has no valid observations")

    absolute_error = observed.sub(forecast).abs()
    squared_error = observed.sub(forecast).pow(2)
    valid_peak = peak_mask.reindex(observed.index).fillna(False).astype(bool)

    peak_error = absolute_error.loc[valid_peak]
    rolling_mae = absolute_error.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).mean()

    if not np.isfinite(mase_denominator) or mase_denominator <= 0.0:
        raise BenchmarkError("MASE denominator must be finite and positive")

    return {
        "validation_rows": int(len(observed)),
        "peak_rows": int(valid_peak.sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "mase": float(absolute_error.mean() / mase_denominator),
        "peak_mae": float(peak_error.mean()) if not peak_error.empty else float("nan"),
        "maximum_rolling_96_mae": float(rolling_mae.max()),
    }


def _classification_metrics(
    actual: pd.Series,
    probability: pd.Series,
    *,
    alert_rate: float,
    calibration_bins: int,
) -> dict[str, float | int]:
    valid = actual.notna() & probability.notna()
    observed = actual.loc[valid].astype(int)
    forecast = (
        probability.loc[valid]
        .astype(float)
        .clip(1e-12, 1.0 - 1e-12)
    )

    if observed.empty:
        raise BenchmarkError(
            "Classification benchmark has no valid observations"
        )

    if observed.nunique() != 2:
        raise BenchmarkError(
            "Classification validation fold requires both classes"
        )

    alert_metrics = controlled_alert_metrics(
        observed,
        forecast,
        alert_rate=alert_rate,
    )
    calibration_error = expected_calibration_error(
        observed,
        forecast,
        bins=calibration_bins,
    )

    return {
        "validation_rows": int(len(observed)),
        "validation_prevalence": float(observed.mean()),
        "pr_auc": float(average_precision_score(observed, forecast)),
        "roc_auc": float(roc_auc_score(observed, forecast)),
        "brier_score": float(brier_score_loss(observed, forecast)),
        "log_loss": float(log_loss(observed, forecast, labels=[0, 1])),
        "recall_at_controlled_alert_rate": alert_metrics[
            "controlled_alert_recall"
        ],
        "precision_at_controlled_alert_rate": alert_metrics[
            "controlled_alert_precision"
        ],
        "controlled_alert_rate_realized": alert_metrics[
            "controlled_alert_rate_realized"
        ],
        "controlled_alert_count_equivalent": alert_metrics[
            "controlled_alert_count_equivalent"
        ],
        "alert_probability_threshold": alert_metrics[
            "alert_probability_threshold"
        ],
        "alert_cutoff_tie_weight": alert_metrics[
            "alert_cutoff_tie_weight"
        ],
        "expected_calibration_error": calibration_error,
    }


def _mase_denominator(
    usage: pd.Series,
    fold: ChronologicalFold,
    seasonal_lag: int,
) -> float:
    training = usage.iloc[fold.train_start : fold.train_stop].astype(float)
    differences = training.sub(training.shift(seasonal_lag)).abs().dropna()

    if differences.empty:
        raise BenchmarkError("Training history is insufficient for MASE")

    return float(differences.mean())


def evaluate_benchmarks(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict[str, Any],
    *,
    target_contract: dict[str, Any] | None = None,
) -> BenchmarkEvaluation:
    required = {"effective_timestamp", "usage_kwh"}
    missing = sorted(required.difference(silver.columns))

    if missing:
        raise BenchmarkError(f"Silver fields are missing: {missing}")

    if not folds:
        raise BenchmarkError("No chronological folds were supplied")

    usage = pd.to_numeric(silver["usage_kwh"], errors="raise").astype(float)
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    regression_name = str(model_contract["objectives"]["regression_target"])
    classification_name = str(
        model_contract["objectives"]["classification_target"]
    )
    daily_lag = int(model_contract["regression_ladder"]["seasonal_daily_lag_steps"])
    weekly_lag = int(
        model_contract["regression_ladder"]["seasonal_weekly_lag_steps"]
    )
    horizon_steps = int(model_contract["objectives"]["regression_horizon_minutes"] // 15)
    rolling_window = 96
    alert_rate = float(
        model_contract["operating_threshold"]["controlled_alert_rate"]
    )
    calibration_bins = int(
        model_contract["operating_threshold"]["calibration_bins"]
    )

    if horizon_steps < 1:
        raise BenchmarkError("Regression horizon must be positive")

    regression_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        if fold.validation_stop > fold.test_start:
            raise BenchmarkError("Validation rows enter the locked test block")

        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[fold.train_start : fold.train_stop] = True

        target_artifacts = build_supervised_targets(
            silver,
            training_mask,
            contract=target_contract,
        )
        targets = target_artifacts.frame
        threshold = float(target_artifacts.peak_threshold_kwh)
        validation_slice = slice(fold.validation_start, fold.validation_stop)
        validation_index = silver.index[validation_slice]
        actual_regression = targets.loc[validation_index, regression_name].astype(float)
        actual_classification = targets.loc[
            validation_index, classification_name
        ].astype("Int64")
        peak_state = actual_regression.ge(threshold)
        mase_scale = _mase_denominator(usage, fold, daily_lag)

        regression_predictions = {
            "persistence": usage,
            "seasonal_naive_daily": usage.shift(daily_lag - horizon_steps),
            "seasonal_naive_weekly": usage.shift(weekly_lag - horizon_steps),
        }

        for benchmark, full_prediction in regression_predictions.items():
            prediction = full_prediction.loc[validation_index].astype(float)
            metrics = _regression_metrics(
                actual_regression,
                prediction,
                peak_mask=peak_state,
                mase_denominator=mase_scale,
                rolling_window=rolling_window,
            )
            regression_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "benchmark": benchmark,
                    "peak_threshold_kwh": threshold,
                    "mase_denominator": mase_scale,
                    **metrics,
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "fold_id": fold.fold_id,
                        "row_position": validation_index.to_numpy(),
                        "prediction_origin": timestamps.loc[validation_index].to_numpy(),
                        "task": "regression",
                        "benchmark": benchmark,
                        "actual": actual_regression.to_numpy(),
                        "prediction": prediction.to_numpy(),
                        "peak_threshold_kwh": threshold,
                        "is_peak_state": peak_state.to_numpy(),
                    }
                )
            )

        training_classification = targets.iloc[
            fold.train_start : fold.train_stop
        ][classification_name].dropna().astype(int)

        if training_classification.empty:
            raise BenchmarkError("Training peak labels are unavailable")

        prevalence = float(training_classification.mean())
        probability = pd.Series(
            prevalence,
            index=validation_index,
            dtype=float,
        )
        classification_metrics = _classification_metrics(
            actual_classification,
            probability,
            alert_rate=alert_rate,
            calibration_bins=calibration_bins,
        )
        classification_rows.append(
            {
                "fold_id": fold.fold_id,
                "benchmark": "training_prevalence",
                "training_prevalence": prevalence,
                "peak_threshold_kwh": threshold,
                **classification_metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold_id": fold.fold_id,
                    "row_position": validation_index.to_numpy(),
                    "prediction_origin": timestamps.loc[validation_index].to_numpy(),
                    "task": "classification",
                    "benchmark": "training_prevalence",
                    "actual": actual_classification.astype(float).to_numpy(),
                    "prediction": probability.to_numpy(),
                    "peak_threshold_kwh": threshold,
                    "is_peak_state": actual_classification.astype(bool).to_numpy(),
                }
            )
        )

    return BenchmarkEvaluation(
        regression_results=pd.DataFrame(regression_rows),
        classification_results=pd.DataFrame(classification_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )
