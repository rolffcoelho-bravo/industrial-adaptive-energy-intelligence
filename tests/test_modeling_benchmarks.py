from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.modeling.benchmarks import (
    BenchmarkError,
    BenchmarkEvaluation,
    evaluate_benchmarks,
)
from iaei.modeling.splits import build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT
    / "data"
    / "processed"
    / "steel_energy_silver.parquet"
)
MODEL_CONTRACT_PATH = ROOT / "configs" / "model_contract.yml"
TARGET_CONTRACT_PATH = ROOT / "configs" / "target_contract.yml"


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    return pd.read_parquet(SILVER_PATH)


@pytest.fixture(scope="module")
def model_contract() -> dict:
    return load_yaml(MODEL_CONTRACT_PATH)


@pytest.fixture(scope="module")
def target_contract() -> dict:
    return load_yaml(TARGET_CONTRACT_PATH)


@pytest.fixture(scope="module")
def benchmark_evaluation(
    silver: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> BenchmarkEvaluation:
    folds = build_expanding_window_folds(
        silver["effective_timestamp"],
        model_contract,
    )
    return evaluate_benchmarks(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )


def test_benchmark_evidence_has_expected_shape(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    assert len(benchmark_evaluation.regression_results) == 12
    assert len(benchmark_evaluation.classification_results) == 4
    assert len(benchmark_evaluation.predictions) == 28_016


def test_prediction_origins_never_enter_locked_boundary(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    predictions = benchmark_evaluation.predictions

    assert predictions["row_position"].nunique() == 7_004
    assert int(predictions["row_position"].min()) == 21_024
    assert int(predictions["row_position"].max()) == 28_027
    assert not predictions["row_position"].between(28_028, 35_039).any()


def test_target_dependencies_stop_before_locked_test(
    benchmark_evaluation: BenchmarkEvaluation,
    model_contract: dict,
) -> None:
    maximum_origin = int(
        benchmark_evaluation.predictions["row_position"].max()
    )
    classification_horizon = int(
        model_contract["objectives"]["classification_horizon_minutes"]
        // 15
    )

    assert maximum_origin + classification_horizon == 28_031
    assert maximum_origin + classification_horizon < 28_032


def test_each_benchmark_covers_all_validation_origins(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    counts = benchmark_evaluation.predictions.groupby(
        ["task", "benchmark"]
    ).size()

    assert counts.to_dict() == {
        ("classification", "training_prevalence"): 7_004,
        ("regression", "persistence"): 7_004,
        ("regression", "seasonal_naive_daily"): 7_004,
        ("regression", "seasonal_naive_weekly"): 7_004,
    }


def test_persistence_is_the_strongest_naive_reference(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    results = benchmark_evaluation.regression_results
    mean_mae = results.groupby("benchmark")["mae"].mean()

    assert mean_mae["persistence"] < mean_mae["seasonal_naive_daily"]
    assert mean_mae["persistence"] < mean_mae["seasonal_naive_weekly"]

    persistence = results.loc[
        results["benchmark"].eq("persistence")
    ]
    assert persistence["mase"].lt(1.0).all()


def test_regression_metrics_are_finite(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    results = benchmark_evaluation.regression_results
    metric_columns = [
        "mae",
        "rmse",
        "mase",
        "peak_mae",
        "maximum_rolling_96_mae",
    ]

    assert np.isfinite(results[metric_columns].to_numpy()).all()
    assert results["validation_rows"].eq(1_751).all()


def test_prevalence_reference_matches_classification_contract(
    benchmark_evaluation: BenchmarkEvaluation,
) -> None:
    results = benchmark_evaluation.classification_results

    assert np.allclose(results["roc_auc"], 0.5)
    assert np.allclose(
        results["pr_auc"],
        results["validation_prevalence"],
    )
    assert np.allclose(
        results["controlled_alert_rate_realized"],
        0.10,
    )
    assert np.allclose(
        results["recall_at_controlled_alert_rate"],
        0.10,
    )
    assert np.allclose(
        results["precision_at_controlled_alert_rate"],
        results["validation_prevalence"],
    )

    expected_gap = (
        results["validation_prevalence"]
        - results["training_prevalence"]
    ).abs()
    assert np.allclose(
        results["expected_calibration_error"],
        expected_gap,
    )


def test_missing_required_silver_field_is_rejected(
    silver: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = silver.drop(columns=["usage_kwh"])
    folds = build_expanding_window_folds(
        altered["effective_timestamp"],
        model_contract,
    )

    with pytest.raises(BenchmarkError, match="Silver fields are missing"):
        evaluate_benchmarks(
            altered,
            folds,
            model_contract,
            target_contract=target_contract,
        )


def test_empty_fold_collection_is_rejected(
    silver: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> None:
    with pytest.raises(BenchmarkError, match="No chronological folds"):
        evaluate_benchmarks(
            silver,
            [],
            model_contract,
            target_contract=target_contract,
        )
