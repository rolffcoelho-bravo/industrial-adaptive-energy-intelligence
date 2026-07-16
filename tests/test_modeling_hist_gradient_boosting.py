from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.modeling.hist_gradient_boosting import (
    HistGradientBoostingCandidateError,
    HistGradientBoostingCandidateEvaluation,
    build_hist_gradient_boosting_estimator,
    evaluate_hist_gradient_boosting_candidate,
)
from iaei.modeling.splits import (
    ChronologicalFold,
    build_expanding_window_folds,
)


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
)
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'
RESULTS_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'hist_gradient_boosting_validation_results.csv'
)
MANIFEST_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'hist_gradient_boosting_candidate_manifest.json'
)


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
def smoke_evaluation(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
    target_contract: dict,
) -> HistGradientBoostingCandidateEvaluation:
    reduced_contract = copy.deepcopy(model_contract)
    selection = reduced_contract['candidate_selection']
    selection['hist_gradient_boosting_learning_rate_grid'] = [0.10]
    selection['hist_gradient_boosting_max_iter_grid'] = [150]
    selection['hist_gradient_boosting_max_leaf_nodes_grid'] = [15]
    selection[
        'hist_gradient_boosting_l2_regularization_grid'
    ] = [1.0]

    return evaluate_hist_gradient_boosting_candidate(
        silver,
        folds[:1],
        reduced_contract,
        target_contract=target_contract,
    )


def test_hist_gradient_boosting_artifacts_exist() -> None:
    assert RESULTS_PATH.is_file()
    assert MANIFEST_PATH.is_file()


def test_hist_gradient_boosting_manifest_preserves_boundary() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

    assert manifest['governance_gate'] == '4C3'
    assert manifest['status'] == 'validated'
    assert manifest['locked_test_evaluated'] is False
    assert manifest['validation_origin_count'] == 7_004
    assert manifest['prediction_row_count'] == 7_004
    assert manifest['maximum_prediction_origin'] == 28_027
    assert manifest['maximum_target_dependency'] == 28_031
    assert manifest['locked_test_start'] == 28_032
    assert manifest['search_governance'][
        'internal_early_stopping'
    ] is False


def test_hist_gradient_boosting_results_are_complete() -> None:
    results = pd.read_csv(RESULTS_PATH)
    metric_columns = [
        'inner_validation_mae',
        'mae',
        'rmse',
        'mase',
        'peak_mae',
        'maximum_rolling_96_mae',
    ]

    assert len(results) == 4
    assert results['candidate'].eq('hist_gradient_boosting').all()
    assert results['inner_parameter_count'].eq(16).all()
    assert results['inner_split_count'].eq(3).all()
    assert results['inner_total_fit_count'].eq(48).all()
    assert not results['internal_early_stopping'].any()
    assert results['fitted_iteration_count'].equals(
        results['selected_max_iter']
    )
    assert results['feature_count'].gt(0).all()
    assert results['validation_rows'].eq(1_751).all()
    assert results['peak_rows'].gt(0).all()
    assert np.isfinite(results[metric_columns].to_numpy()).all()


def test_hist_gradient_boosting_parameters_match_contract(
    model_contract: dict,
) -> None:
    results = pd.read_csv(RESULTS_PATH)
    selection = model_contract['candidate_selection']

    assert results['selected_learning_rate'].isin(
        selection['hist_gradient_boosting_learning_rate_grid']
    ).all()
    assert results['selected_max_iter'].isin(
        selection['hist_gradient_boosting_max_iter_grid']
    ).all()
    assert results['selected_max_leaf_nodes'].isin(
        selection['hist_gradient_boosting_max_leaf_nodes_grid']
    ).all()
    assert results['selected_l2_regularization'].isin(
        selection[
            'hist_gradient_boosting_l2_regularization_grid'
        ]
    ).all()


def test_hist_gradient_boosting_estimator_controls(
    model_contract: dict,
) -> None:
    estimator = build_hist_gradient_boosting_estimator(model_contract)

    assert estimator.loss == 'absolute_error'
    assert estimator.min_samples_leaf == 20
    assert estimator.max_bins == 255
    assert np.isclose(estimator.max_features, 1.0)
    assert estimator.categorical_features is None
    assert estimator.early_stopping is False
    assert estimator.random_state == 20_260_716


def test_hist_gradient_boosting_promotion_is_consistent() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    checks = manifest['promotion_checks']
    expected = (
        'promoted'
        if all(bool(value) for value in checks.values())
        else 'rejected'
    )

    assert manifest['promotion_decision'] == expected


def test_hist_gradient_boosting_real_data_smoke_path(
    smoke_evaluation: HistGradientBoostingCandidateEvaluation,
) -> None:
    results = smoke_evaluation.regression_results
    predictions = smoke_evaluation.predictions

    assert len(results) == 1
    assert len(predictions) == 1_751
    assert int(predictions['row_position'].min()) == 21_024
    assert int(predictions['row_position'].max()) == 22_774
    assert int(results.loc[0, 'inner_parameter_count']) == 1
    assert int(results.loc[0, 'inner_total_fit_count']) == 3
    assert not bool(results.loc[0, 'internal_early_stopping'])
    assert int(results.loc[0, 'fitted_iteration_count']) == 150


def test_missing_hist_gradient_boosting_feature_is_rejected(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = silver.drop(columns=['usage_lag_96'])

    with pytest.raises(
        HistGradientBoostingCandidateError,
        match='Silver candidate fields are missing',
    ):
        evaluate_hist_gradient_boosting_candidate(
            altered,
            folds[:1],
            model_contract,
            target_contract=target_contract,
        )


def test_empty_hist_gradient_boosting_folds_are_rejected(
    silver: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> None:
    with pytest.raises(
        HistGradientBoostingCandidateError,
        match='No chronological folds',
    ):
        evaluate_hist_gradient_boosting_candidate(
            silver,
            [],
            model_contract,
            target_contract=target_contract,
        )
