from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.modeling.candidates import (
    CandidateModelError,
    RidgeCandidateEvaluation,
    build_feature_preprocessor,
    build_inner_time_series_split,
    evaluate_ridge_candidate,
)
from iaei.modeling.splits import ChronologicalFold, build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'


@pytest.fixture(scope='module')
def silver() -> pd.DataFrame:
    return pd.read_parquet(SILVER_PATH)


@pytest.fixture(scope='module')
def model_contract() -> dict:
    return load_yaml(MODEL_CONTRACT_PATH)


@pytest.fixture(scope='module')
def target_contract() -> dict:
    return load_yaml(TARGET_CONTRACT_PATH)


@pytest.fixture(scope='module')
def folds(
    silver: pd.DataFrame,
    model_contract: dict,
) -> list[ChronologicalFold]:
    return build_expanding_window_folds(
        silver['effective_timestamp'],
        model_contract,
    )


@pytest.fixture(scope='module')
def ridge_evaluation(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
    target_contract: dict,
) -> RidgeCandidateEvaluation:
    return evaluate_ridge_candidate(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )


def test_ridge_evidence_has_expected_shape(
    ridge_evaluation: RidgeCandidateEvaluation,
) -> None:
    assert len(ridge_evaluation.regression_results) == 4
    assert len(ridge_evaluation.predictions) == 7_004


def test_ridge_predictions_preserve_validation_boundary(
    ridge_evaluation: RidgeCandidateEvaluation,
) -> None:
    predictions = ridge_evaluation.predictions

    assert predictions['row_position'].nunique() == 7_004
    assert int(predictions['row_position'].min()) == 21_024
    assert int(predictions['row_position'].max()) == 28_027
    assert not predictions['row_position'].between(28_028, 35_039).any()


def test_ridge_target_dependencies_stop_before_locked_test(
    ridge_evaluation: RidgeCandidateEvaluation,
    model_contract: dict,
) -> None:
    maximum_origin = int(ridge_evaluation.predictions['row_position'].max())
    regression_steps = int(
        model_contract['objectives']['regression_horizon_minutes'] // 15
    )
    peak_steps = int(
        model_contract['objectives']['classification_horizon_minutes'] // 15
    )

    assert maximum_origin + regression_steps == 28_028
    assert maximum_origin + peak_steps == 28_031
    assert maximum_origin + peak_steps < 28_032


def test_ridge_metrics_and_selected_parameters_are_valid(
    ridge_evaluation: RidgeCandidateEvaluation,
    model_contract: dict,
) -> None:
    results = ridge_evaluation.regression_results
    alpha_grid = set(
        float(value)
        for value in model_contract['candidate_selection']['ridge_alpha_grid']
    )
    metric_columns = [
        'inner_validation_mae',
        'mae',
        'rmse',
        'mase',
        'peak_mae',
        'maximum_rolling_96_mae',
    ]

    assert results['candidate'].eq('ridge').all()
    assert results['selected_alpha'].isin(alpha_grid).all()
    assert np.isfinite(results[metric_columns].to_numpy()).all()
    assert results['validation_rows'].eq(1_751).all()
    assert results['peak_rows'].gt(0).all()


def test_inner_time_series_split_preserves_four_step_gap(
    folds: list[ChronologicalFold],
    model_contract: dict,
) -> None:
    splitter = build_inner_time_series_split(model_contract)
    positions = np.arange(folds[0].train_stop)

    for training, validation in splitter.split(positions):
        assert int(validation[0] - training[-1] - 1) == 4
        assert int(training[0]) == 0


def test_numeric_imputer_is_fitted_from_outer_training_only(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
) -> None:
    fold = folds[0]
    numeric_features = model_contract['feature_policy']['numeric_features']
    categorical_features = model_contract['feature_policy']['categorical_features']
    requested = numeric_features + categorical_features
    training = silver.iloc[fold.train_start : fold.train_stop][requested]
    preprocessor = build_feature_preprocessor(model_contract)
    preprocessor.fit(training)

    statistics = preprocessor.named_transformers_['numeric'].named_steps[
        'imputer'
    ].statistics_
    column_position = numeric_features.index('usage_lag_96')
    expected_median = float(training['usage_lag_96'].median())

    assert np.isclose(statistics[column_position], expected_median)


def test_missing_candidate_feature_is_rejected(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = silver.drop(columns=['usage_lag_96'])

    with pytest.raises(
        CandidateModelError,
        match='Silver candidate fields are missing',
    ):
        evaluate_ridge_candidate(
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
    with pytest.raises(CandidateModelError, match='No chronological folds'):
        evaluate_ridge_candidate(
            silver,
            [],
            model_contract,
            target_contract=target_contract,
        )
