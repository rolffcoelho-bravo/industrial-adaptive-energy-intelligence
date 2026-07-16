from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.modeling.elastic_net import (
    ElasticNetCandidateError,
    ElasticNetCandidateEvaluation,
    evaluate_elastic_net_candidate,
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
    ROOT / 'outputs' / 'modeling' / 'elastic_net_validation_results.csv'
)
MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'elastic_net_candidate_manifest.json'
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
) -> ElasticNetCandidateEvaluation:
    reduced_contract = copy.deepcopy(model_contract)
    selection = reduced_contract['candidate_selection']
    selection['elastic_net_alpha_grid'] = [1.0]
    selection['elastic_net_l1_ratio_grid'] = [0.50]

    return evaluate_elastic_net_candidate(
        silver,
        folds[:1],
        reduced_contract,
        target_contract=target_contract,
    )


def test_elastic_net_artifacts_exist() -> None:
    assert RESULTS_PATH.is_file()
    assert MANIFEST_PATH.is_file()


def test_elastic_net_manifest_preserves_locked_boundary() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

    assert manifest['governance_gate'] == '4C2'
    assert manifest['status'] == 'validated'
    assert manifest['locked_test_evaluated'] is False
    assert manifest['validation_origin_count'] == 7_004
    assert manifest['prediction_row_count'] == 7_004
    assert manifest['maximum_prediction_origin'] == 28_027
    assert manifest['maximum_target_dependency'] == 28_031
    assert manifest['locked_test_start'] == 28_032


def test_elastic_net_results_record_convergence() -> None:
    results = pd.read_csv(RESULTS_PATH)
    metric_columns = [
        'inner_validation_mae',
        'mae',
        'rmse',
        'mase',
        'peak_mae',
        'maximum_rolling_96_mae',
        'outer_fit_dual_gap',
    ]

    assert len(results) == 4
    assert results['candidate'].eq('elastic_net').all()
    assert results['outer_fit_converged'].all()
    assert results['inner_parameter_count'].eq(30).all()
    assert results['inner_converged_parameter_count'].gt(0).all()
    assert results['inner_total_fit_count'].eq(90).all()
    assert np.isfinite(results[metric_columns].to_numpy()).all()
    assert results['validation_rows'].eq(1_751).all()


def test_elastic_net_promotion_decision_is_consistent() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    checks = manifest['promotion_checks']
    expected = (
        'promoted'
        if all(bool(value) for value in checks.values())
        else 'rejected'
    )

    assert manifest['promotion_decision'] == expected


def test_elastic_net_real_data_smoke_path(
    smoke_evaluation: ElasticNetCandidateEvaluation,
) -> None:
    results = smoke_evaluation.regression_results
    predictions = smoke_evaluation.predictions

    assert len(results) == 1
    assert len(predictions) == 1_751
    assert int(predictions['row_position'].min()) == 21_024
    assert int(predictions['row_position'].max()) == 22_774
    assert bool(results.loc[0, 'outer_fit_converged'])
    assert int(results.loc[0, 'inner_parameter_count']) == 1
    assert int(results.loc[0, 'inner_converged_parameter_count']) == 1


def test_missing_elastic_net_feature_is_rejected(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = silver.drop(columns=['usage_lag_96'])

    with pytest.raises(
        ElasticNetCandidateError,
        match='Silver candidate fields are missing',
    ):
        evaluate_elastic_net_candidate(
            altered,
            folds[:1],
            model_contract,
            target_contract=target_contract,
        )


def test_empty_elastic_net_fold_collection_is_rejected(
    silver: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> None:
    with pytest.raises(
        ElasticNetCandidateError,
        match='No chronological folds',
    ):
        evaluate_elastic_net_candidate(
            silver,
            [],
            model_contract,
            target_contract=target_contract,
        )
