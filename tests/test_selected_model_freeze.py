from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.modeling.selection import (
    SelectedModelFreeze,
    SelectedModelFreezeError,
    SelectionBoundary,
    freeze_hist_gradient_boosting_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
)
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'
PROCESSING_MANIFEST_PATH = (
    ROOT
    / 'data'
    / 'processed'
    / 'steel_energy_processing_manifest.json'
)
FOLD_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'chronological_folds.json'
)
SELECTED_MANIFEST_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'selected_model_manifest.json'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)

    return digest.hexdigest()


@pytest.fixture(scope='module')
def fold_manifest() -> dict:
    return json.loads(
        FOLD_MANIFEST_PATH.read_text(encoding='utf-8')
    )


@pytest.fixture(scope='module')
def boundary(fold_manifest: dict) -> SelectionBoundary:
    return SelectionBoundary(
        test_purge_start=int(fold_manifest['test_purge_start']),
        test_purge_stop=int(fold_manifest['test_purge_stop']),
        locked_test_start=int(fold_manifest['locked_test_start']),
        locked_test_stop=int(fold_manifest['locked_test_stop']),
        row_count=int(fold_manifest['row_count']),
    )


@pytest.fixture(scope='module')
def selection_frame(
    boundary: SelectionBoundary,
) -> pd.DataFrame:
    query = (
        'SELECT * FROM read_parquet(?) '
        f'LIMIT {boundary.test_purge_stop}'
    )
    connection = duckdb.connect()

    try:
        frame = connection.execute(
            query,
            [str(SILVER_PATH)],
        ).fetch_df()
    finally:
        connection.close()

    return frame.reset_index(drop=True)


@pytest.fixture(scope='module')
def model_contract() -> dict:
    return load_yaml(MODEL_CONTRACT_PATH)


@pytest.fixture(scope='module')
def target_contract() -> dict:
    return load_yaml(TARGET_CONTRACT_PATH)


@pytest.fixture(scope='module')
def manifest() -> dict:
    return json.loads(
        SELECTED_MANIFEST_PATH.read_text(encoding='utf-8')
    )


@pytest.fixture(scope='module')
def smoke_freeze(
    selection_frame: pd.DataFrame,
    boundary: SelectionBoundary,
    model_contract: dict,
    target_contract: dict,
) -> SelectedModelFreeze:
    reduced = copy.deepcopy(model_contract)
    selection = reduced['candidate_selection']
    selection[
        'hist_gradient_boosting_learning_rate_grid'
    ] = [0.10]
    selection[
        'hist_gradient_boosting_max_iter_grid'
    ] = [150]
    selection[
        'hist_gradient_boosting_max_leaf_nodes_grid'
    ] = [15]
    selection[
        'hist_gradient_boosting_l2_regularization_grid'
    ] = [1.0]

    return freeze_hist_gradient_boosting_selection(
        selection_frame,
        boundary,
        reduced,
        target_contract=target_contract,
    )


def test_selected_model_manifest_exists() -> None:
    assert SELECTED_MANIFEST_PATH.is_file()


def test_selected_model_manifest_is_pretest_only(
    manifest: dict,
) -> None:
    assert manifest['governance_gate'] == '4D'
    assert manifest['status'] == 'frozen'
    assert manifest['selection_basis'] == 'validation_only'
    assert manifest['selected_model'] == (
        'hist_gradient_boosting'
    )
    assert manifest['locked_test_evaluated'] is False
    assert manifest['locked_test_prediction_rows_produced'] == 0
    assert manifest['authorized_next_gate'] == '4E'


def test_selection_input_stops_before_locked_test(
    manifest: dict,
) -> None:
    selection = manifest['selection_input_boundary']

    assert selection['start'] == 0
    assert selection['stop_exclusive'] == 28_032
    assert selection['row_count'] == 28_032
    assert selection['training_origin_stop_exclusive'] == 28_028
    assert selection['purge_dependency_start'] == 28_028
    assert selection['purge_dependency_stop_exclusive'] == 28_032
    assert selection['purge_rows_used_as_prediction_origins'] is False
    assert (
        selection[
            'purge_rows_used_only_for_future_target_dependencies'
        ]
        is True
    )
    assert selection['locked_test_rows_materialized'] is False
    assert (
        selection['physical_storage_byte_exclusion_claimed']
        is False
    )


def test_selected_model_boundary_is_locked(
    manifest: dict,
) -> None:
    boundary = manifest['training_boundary']

    assert boundary['training_start'] == 0
    assert boundary['training_stop_exclusive'] == 28_028
    assert boundary['training_origin_count'] == 28_028
    assert boundary['training_label_count'] == 28_028
    assert boundary['maximum_training_origin'] == 28_027
    assert boundary['maximum_target_dependency'] == 28_031
    assert boundary['test_purge_start'] == 28_028
    assert boundary['test_purge_stop'] == 28_032
    assert boundary['locked_test_start'] == 28_032
    assert boundary['locked_test_stop'] == 35_040


def test_candidate_ladder_decisions_are_preserved(
    manifest: dict,
) -> None:
    ladder = manifest['candidate_ladder']

    assert ladder['persistence']['role'] == 'formal_reference'
    assert ladder['ridge']['decision'] == 'rejected'
    assert ladder['elastic_net']['decision'] == 'rejected'
    assert ladder['hist_gradient_boosting']['decision'] == (
        'promoted'
    )


def test_final_search_contract_is_exact(
    manifest: dict,
    model_contract: dict,
) -> None:
    search = manifest['search_governance']
    parameters = manifest['selected_parameters']
    selection = model_contract['candidate_selection']

    assert search['strategy'] == 'chronological_grid_search'
    assert search['scoring'] == 'neg_mean_absolute_error'
    assert search['parameter_combinations'] == 16
    assert search['inner_splits'] == 3
    assert search['inner_gap_steps'] == 4
    assert search['total_inner_fit_count'] == 48
    assert search['shuffle'] is False
    assert search['internal_early_stopping'] is False
    assert search['refit_on_all_pretest_training_origins'] is True

    assert parameters['learning_rate'] in selection[
        'hist_gradient_boosting_learning_rate_grid'
    ]
    assert parameters['max_iter'] in selection[
        'hist_gradient_boosting_max_iter_grid'
    ]
    assert parameters['max_leaf_nodes'] in selection[
        'hist_gradient_boosting_max_leaf_nodes_grid'
    ]
    assert parameters['l2_regularization'] in selection[
        'hist_gradient_boosting_l2_regularization_grid'
    ]


def test_fitted_model_evidence_matches_selection(
    manifest: dict,
) -> None:
    fitted = manifest['fitted_model_evidence']
    parameters = manifest['selected_parameters']

    assert fitted['feature_count'] > 0
    assert fitted['fitted_iteration_count'] == (
        parameters['max_iter']
    )
    assert fitted['internal_early_stopping'] is False
    assert fitted['model_binary_committed'] is False


def test_test_access_controls_are_closed(
    manifest: dict,
) -> None:
    controls = manifest['test_access_controls']

    assert controls['locked_test_rows_materialized'] is False
    assert (
        controls['locked_test_features_used_for_prediction']
        is False
    )
    assert controls['locked_test_targets_used_for_scoring'] is False
    assert controls['locked_test_metrics_computed'] is False
    assert controls['locked_test_predictions_written'] is False
    assert controls['physical_storage_byte_exclusion_claimed'] is False
    assert controls['single_evaluation_authorized_gate'] == '4E'


def test_static_source_hashes_match_repository(
    manifest: dict,
) -> None:
    for record in manifest['source_evidence'].values():
        path = ROOT / record['path']

        assert path.is_file()
        assert _sha256(path) == record['sha256']


def test_silver_identity_matches_reconstructed_layer(
    manifest: dict,
) -> None:
    processing = json.loads(
        PROCESSING_MANIFEST_PATH.read_text(encoding='utf-8')
    )
    identity = manifest['data_identity']

    assert identity['silver_parquet_sha256'] == (
        processing['output']['parquet_sha256']
    )
    assert identity['silver_row_count'] == 35_040
    assert identity['silver_column_count'] == 57
    assert identity['quality_flag_dq_any'] == 0


def test_selection_frame_contains_only_pretest_and_purge_rows(
    selection_frame: pd.DataFrame,
    boundary: SelectionBoundary,
) -> None:
    assert len(selection_frame) == boundary.test_purge_stop
    assert selection_frame['source_row_number'].iloc[0] == 0
    assert selection_frame['source_row_number'].iloc[-1] == 28_031
    assert pd.Timestamp(
        selection_frame['effective_timestamp'].iloc[-1]
    ) < pd.Timestamp('2018-10-20T00:15:00')


def test_real_data_pretest_freeze_smoke_path(
    smoke_freeze: SelectedModelFreeze,
) -> None:
    assert smoke_freeze.parameter_count == 1
    assert smoke_freeze.inner_split_count == 3
    assert smoke_freeze.total_inner_fit_count == 3
    assert smoke_freeze.training_start == 0
    assert smoke_freeze.training_stop == 28_028
    assert smoke_freeze.training_origin_count == 28_028
    assert smoke_freeze.training_label_count == 28_028
    assert smoke_freeze.maximum_training_origin == 28_027
    assert smoke_freeze.maximum_target_dependency == 28_031
    assert smoke_freeze.selection_input_stop == 28_032
    assert smoke_freeze.locked_test_start == 28_032
    assert smoke_freeze.fitted_iteration_count == 150
    assert smoke_freeze.internal_early_stopping is False


def test_missing_feature_is_rejected_before_selection(
    selection_frame: pd.DataFrame,
    boundary: SelectionBoundary,
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = selection_frame.drop(columns=['usage_lag_96'])

    with pytest.raises(
        SelectedModelFreezeError,
        match='Silver candidate fields are missing',
    ):
        freeze_hist_gradient_boosting_selection(
            altered,
            boundary,
            model_contract,
            target_contract=target_contract,
        )


def test_frame_entering_locked_test_is_rejected(
    selection_frame: pd.DataFrame,
    boundary: SelectionBoundary,
    model_contract: dict,
    target_contract: dict,
) -> None:
    altered = pd.concat(
        [selection_frame, selection_frame.tail(1)],
        ignore_index=True,
    )

    with pytest.raises(
        SelectedModelFreezeError,
        match='Selection frame must stop',
    ):
        freeze_hist_gradient_boosting_selection(
            altered,
            boundary,
            model_contract,
            target_contract=target_contract,
        )


def test_inconsistent_boundary_is_rejected(
    selection_frame: pd.DataFrame,
    model_contract: dict,
    target_contract: dict,
) -> None:
    invalid = SelectionBoundary(
        test_purge_start=28_028,
        test_purge_stop=28_032,
        locked_test_start=28_033,
        locked_test_stop=35_040,
        row_count=35_040,
    )

    with pytest.raises(
        SelectedModelFreezeError,
        match='boundaries are inconsistent',
    ):
        freeze_hist_gradient_boosting_selection(
            selection_frame,
            invalid,
            model_contract,
            target_contract=target_contract,
        )
