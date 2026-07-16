from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.selection import (
    SelectionBoundary,
    freeze_hist_gradient_boosting_selection,
)


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
)
SILVER_PROCESSING_MANIFEST_PATH = (
    ROOT
    / 'data'
    / 'processed'
    / 'steel_energy_processing_manifest.json'
)
RAW_MANIFEST_PATH = (
    ROOT
    / 'data'
    / 'manifests'
    / 'uci_steel_energy_manifest.json'
)
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'
SILVER_CONTRACT_PATH = ROOT / 'configs' / 'silver_contract.yml'
FOLD_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'chronological_folds.json'
)
BENCHMARK_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'benchmark_manifest.json'
)
RIDGE_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'ridge_candidate_manifest.json'
)
ELASTIC_NET_MANIFEST_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'elastic_net_candidate_manifest.json'
)
HISTOGRAM_MANIFEST_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'hist_gradient_boosting_candidate_manifest.json'
)
OUTPUT_DIRECTORY = ROOT / 'outputs' / 'modeling'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)

    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f'Required evidence file is missing: {path}')

    return json.loads(path.read_text(encoding='utf-8'))


def _source_record(path: Path) -> dict[str, str]:
    return {
        'path': path.relative_to(ROOT).as_posix(),
        'sha256': _sha256(path),
    }


def _boundary_from_fold_manifest(
    payload: dict[str, Any],
) -> SelectionBoundary:
    boundary = SelectionBoundary(
        test_purge_start=int(payload['test_purge_start']),
        test_purge_stop=int(payload['test_purge_stop']),
        locked_test_start=int(payload['locked_test_start']),
        locked_test_stop=int(payload['locked_test_stop']),
        row_count=int(payload['row_count']),
    )

    if payload.get('split_type') != 'expanding_window':
        raise RuntimeError('Unexpected chronological split type')

    if int(payload.get('validation_fold_count', 0)) != 4:
        raise RuntimeError('Unexpected validation-fold count')

    if boundary.test_purge_stop != boundary.locked_test_start:
        raise RuntimeError('Purge does not end at the locked test')

    fold_boundaries = {
        (
            int(fold['test_purge_start']),
            int(fold['test_purge_stop']),
            int(fold['test_start']),
            int(fold['test_stop']),
        )
        for fold in payload['folds']
    }

    expected = {
        (
            boundary.test_purge_start,
            boundary.test_purge_stop,
            boundary.locked_test_start,
            boundary.locked_test_stop,
        )
    }

    if fold_boundaries != expected:
        raise RuntimeError(
            'Fold-level locked boundaries are inconsistent'
        )

    return boundary


def _validate_benchmark(
    payload: dict[str, Any],
    boundary: SelectionBoundary,
) -> None:
    if payload.get('governance_gate') != '4B':
        raise RuntimeError('Unexpected benchmark governance gate')

    if payload.get('status') != 'validated':
        raise RuntimeError('Benchmark evidence is not validated')

    expected = {
        'test_purge_start': boundary.test_purge_start,
        'test_purge_stop': boundary.test_purge_stop,
        'locked_test_start': boundary.locked_test_start,
        'locked_test_stop': boundary.locked_test_stop,
        'silver_row_count': boundary.row_count,
    }

    for key, value in expected.items():
        if int(payload.get(key, -1)) != value:
            raise RuntimeError(
                f'Benchmark boundary mismatch for {key}'
            )

    maximum_origin = int(payload['maximum_prediction_origin'])
    maximum_dependency = int(payload['maximum_target_dependency'])

    if maximum_origin >= boundary.test_purge_start:
        raise RuntimeError(
            'Benchmark origins enter the test-boundary purge'
        )

    if maximum_dependency >= boundary.locked_test_start:
        raise RuntimeError(
            'Benchmark targets consume locked-test observations'
        )


def _validate_candidate(
    path: Path,
    boundary: SelectionBoundary,
    *,
    candidate: str,
    decision: str,
) -> dict[str, Any]:
    payload = _load_json(path)

    if payload.get('candidate') != candidate:
        raise RuntimeError(
            f'Unexpected candidate identity in {path}'
        )

    if payload.get('promotion_decision') != decision:
        raise RuntimeError(
            f'Unexpected promotion decision in {path}'
        )

    if payload.get('locked_test_evaluated') is not False:
        raise RuntimeError(
            f'Candidate evidence evaluated the locked test: {path}'
        )

    expected = {
        'test_purge_start': boundary.test_purge_start,
        'test_purge_stop': boundary.test_purge_stop,
        'locked_test_start': boundary.locked_test_start,
        'locked_test_stop': boundary.locked_test_stop,
    }

    for key, value in expected.items():
        if int(payload.get(key, -1)) != value:
            raise RuntimeError(
                f'Candidate boundary mismatch for {key}: {path}'
            )

    if int(payload['maximum_prediction_origin']) >= (
        boundary.test_purge_start
    ):
        raise RuntimeError(
            f'Candidate origins enter the purge: {path}'
        )

    if int(payload['maximum_target_dependency']) >= (
        boundary.locked_test_start
    ):
        raise RuntimeError(
            f'Candidate targets enter the locked test: {path}'
        )

    return payload


def _read_selection_frame(
    path: Path,
    boundary: SelectionBoundary,
) -> pd.DataFrame:
    query = (
        'SELECT * FROM read_parquet(?) '
        f'LIMIT {boundary.test_purge_stop}'
    )
    connection = duckdb.connect()

    try:
        frame = connection.execute(query, [str(path)]).fetch_df()
    finally:
        connection.close()

    frame = frame.reset_index(drop=True)

    if len(frame) != boundary.test_purge_stop:
        raise RuntimeError(
            'Pre-test selection frame has an unexpected row count'
        )

    if 'source_row_number' not in frame.columns:
        raise RuntimeError('Silver source-row identity is missing')

    source_rows = pd.to_numeric(
        frame['source_row_number'],
        errors='raise',
    ).astype(int)

    if source_rows.iloc[0] != 0:
        raise RuntimeError('Selection frame does not start at row zero')

    if source_rows.iloc[-1] != boundary.test_purge_stop - 1:
        raise RuntimeError(
            'Selection frame enters or truncates the purge boundary'
        )

    if not source_rows.equals(
        pd.Series(range(boundary.test_purge_stop))
    ):
        raise RuntimeError(
            'Selection frame source rows are not contiguous'
        )

    return frame


def main() -> None:
    model_contract = load_yaml(MODEL_CONTRACT_PATH)
    target_contract = load_yaml(TARGET_CONTRACT_PATH)
    silver_contract = load_yaml(SILVER_CONTRACT_PATH)
    silver_processing = _load_json(
        SILVER_PROCESSING_MANIFEST_PATH
    )
    raw_manifest = _load_json(RAW_MANIFEST_PATH)
    fold_manifest = _load_json(FOLD_MANIFEST_PATH)
    benchmark_manifest = _load_json(BENCHMARK_MANIFEST_PATH)

    boundary = _boundary_from_fold_manifest(fold_manifest)
    _validate_benchmark(benchmark_manifest, boundary)

    ridge_manifest = _validate_candidate(
        RIDGE_MANIFEST_PATH,
        boundary,
        candidate='ridge',
        decision='rejected',
    )
    elastic_net_manifest = _validate_candidate(
        ELASTIC_NET_MANIFEST_PATH,
        boundary,
        candidate='elastic_net',
        decision='rejected',
    )
    histogram_manifest = _validate_candidate(
        HISTOGRAM_MANIFEST_PATH,
        boundary,
        candidate='hist_gradient_boosting',
        decision='promoted',
    )

    if int(silver_processing['output']['row_count']) != (
        boundary.row_count
    ):
        raise RuntimeError(
            'Silver processing manifest row count is inconsistent'
        )

    if int(
        silver_contract['chronology']['frequency_minutes']
    ) != 15:
        raise RuntimeError('Unexpected Silver frequency')

    selection_frame = _read_selection_frame(SILVER_PATH, boundary)
    freeze = freeze_hist_gradient_boosting_selection(
        selection_frame,
        boundary,
        model_contract,
        target_contract=target_contract,
    )

    if freeze.internal_early_stopping:
        raise RuntimeError(
            'Frozen model activated internal early stopping'
        )

    if freeze.maximum_target_dependency >= freeze.locked_test_start:
        raise RuntimeError(
            'Frozen-model labels consume locked-test observations'
        )

    output_name = str(
        model_contract['outputs']['selected_model_manifest']
    )
    output_path = OUTPUT_DIRECTORY / output_name

    source_evidence = {
        'model_contract': _source_record(MODEL_CONTRACT_PATH),
        'target_contract': _source_record(TARGET_CONTRACT_PATH),
        'silver_contract': _source_record(SILVER_CONTRACT_PATH),
        'raw_manifest': _source_record(RAW_MANIFEST_PATH),
        'chronological_folds': _source_record(
            FOLD_MANIFEST_PATH
        ),
        'benchmark_manifest': _source_record(
            BENCHMARK_MANIFEST_PATH
        ),
        'ridge_candidate_manifest': _source_record(
            RIDGE_MANIFEST_PATH
        ),
        'elastic_net_candidate_manifest': _source_record(
            ELASTIC_NET_MANIFEST_PATH
        ),
        'hist_gradient_boosting_candidate_manifest': (
            _source_record(HISTOGRAM_MANIFEST_PATH)
        ),
    }

    manifest = {
        'contract_version': str(model_contract['contract_version']),
        'governance_gate': '4D',
        'status': 'frozen',
        'selection_basis': 'validation_only',
        'selected_model': 'hist_gradient_boosting',
        'locked_test_evaluated': False,
        'locked_test_prediction_rows_produced': 0,
        'authorized_next_gate': '4E',
        'candidate_ladder': {
            'persistence': {
                'role': 'formal_reference',
                'manifest': source_evidence['benchmark_manifest'],
            },
            'ridge': {
                'decision': ridge_manifest['promotion_decision'],
                'manifest': source_evidence[
                    'ridge_candidate_manifest'
                ],
            },
            'elastic_net': {
                'decision': (
                    elastic_net_manifest['promotion_decision']
                ),
                'manifest': source_evidence[
                    'elastic_net_candidate_manifest'
                ],
            },
            'hist_gradient_boosting': {
                'decision': (
                    histogram_manifest['promotion_decision']
                ),
                'manifest': source_evidence[
                    'hist_gradient_boosting_candidate_manifest'
                ],
            },
        },
        'data_identity': {
            'raw_csv_sha256': raw_manifest['csv_sha256'],
            'silver_parquet_sha256': silver_processing[
                'output'
            ]['parquet_sha256'],
            'silver_row_count': int(
                silver_processing['output']['row_count']
            ),
            'silver_column_count': int(
                silver_processing['output']['column_count']
            ),
            'quality_flag_dq_any': int(
                silver_processing['quality'][
                    'quality_flag_counts'
                ]['dq_any']
            ),
        },
        'selection_input_boundary': {
            'start': freeze.selection_input_start,
            'stop_exclusive': freeze.selection_input_stop,
            'row_count': freeze.selection_input_row_count,
            'training_origin_stop_exclusive': (
                freeze.training_stop
            ),
            'purge_dependency_start': freeze.test_purge_start,
            'purge_dependency_stop_exclusive': (
                freeze.test_purge_stop
            ),
            'purge_rows_used_as_prediction_origins': False,
            'purge_rows_used_only_for_future_target_dependencies': (
                True
            ),
            'locked_test_rows_materialized': False,
            'physical_storage_byte_exclusion_claimed': False,
            'selection_input_timestamp_end': (
                freeze.selection_input_timestamp_end
            ),
        },
        'training_boundary': {
            'training_start': freeze.training_start,
            'training_stop_exclusive': freeze.training_stop,
            'training_origin_count': freeze.training_origin_count,
            'training_label_count': freeze.training_label_count,
            'maximum_training_origin': (
                freeze.maximum_training_origin
            ),
            'maximum_target_dependency': (
                freeze.maximum_target_dependency
            ),
            'test_purge_start': freeze.test_purge_start,
            'test_purge_stop': freeze.test_purge_stop,
            'locked_test_start': freeze.locked_test_start,
            'locked_test_stop': freeze.locked_test_stop,
            'training_timestamp_start': (
                freeze.training_timestamp_start
            ),
            'training_timestamp_end': (
                freeze.training_timestamp_end
            ),
        },
        'search_governance': {
            'strategy': 'chronological_grid_search',
            'scoring': 'neg_mean_absolute_error',
            'parameter_combinations': freeze.parameter_count,
            'inner_splits': freeze.inner_split_count,
            'inner_gap_steps': int(
                model_contract['candidate_selection'][
                    'inner_gap_steps'
                ]
            ),
            'total_inner_fit_count': (
                freeze.total_inner_fit_count
            ),
            'shuffle': False,
            'internal_early_stopping': False,
            'refit_on_all_pretest_training_origins': True,
            'inner_validation_mae': (
                freeze.inner_validation_mae
            ),
        },
        'selected_parameters': freeze.selected_parameters,
        'fitted_model_evidence': {
            'feature_count': freeze.feature_count,
            'fitted_iteration_count': (
                freeze.fitted_iteration_count
            ),
            'internal_early_stopping': (
                freeze.internal_early_stopping
            ),
            'model_binary_committed': False,
        },
        'test_access_controls': {
            'locked_test_rows_materialized': False,
            'locked_test_features_used_for_prediction': False,
            'locked_test_targets_used_for_scoring': False,
            'locked_test_metrics_computed': False,
            'locked_test_predictions_written': False,
            'physical_storage_byte_exclusion_claimed': False,
            'single_evaluation_authorized_gate': '4E',
        },
        'reconstruction_contract': {
            'selection_script': (
                'scripts/build_selected_model_manifest.py'
            ),
            'selection_module': 'src/iaei/modeling/selection.py',
            'estimator_module': (
                'src/iaei/modeling/hist_gradient_boosting.py'
            ),
            'refit_required_before_test_evaluation': True,
            'selected_parameters_are_immutable': True,
        },
        'source_evidence': source_evidence,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print(
        'Selected model freeze: PASS | candidate={} | '
        'training_origins={} | inner_mae={:.6f} | '
        'selection_rows={} | locked_test_evaluated={}'.format(
            manifest['selected_model'],
            manifest['training_boundary'][
                'training_origin_count'
            ],
            manifest['search_governance'][
                'inner_validation_mae'
            ],
            manifest['selection_input_boundary']['row_count'],
            manifest['locked_test_evaluated'],
        )
    )


if __name__ == '__main__':
    main()
