from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.hist_gradient_boosting import (
    evaluate_hist_gradient_boosting_candidate,
)
from iaei.modeling.splits import build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = (
    ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
)
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'
BENCHMARK_RESULTS_PATH = (
    ROOT / 'outputs' / 'modeling' / 'regression_validation_results.csv'
)
BENCHMARK_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'benchmark_manifest.json'
)
RIDGE_RESULTS_PATH = (
    ROOT / 'outputs' / 'modeling' / 'ridge_validation_results.csv'
)
RIDGE_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'ridge_candidate_manifest.json'
)
ELASTIC_NET_RESULTS_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'elastic_net_validation_results.csv'
)
ELASTIC_NET_MANIFEST_PATH = (
    ROOT
    / 'outputs'
    / 'modeling'
    / 'elastic_net_candidate_manifest.json'
)
OUTPUT_DIRECTORY = ROOT / 'outputs' / 'modeling'


def _relative_change(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise RuntimeError('Reference metric must be positive')
    return float((candidate - reference) / reference)


def _relative_improvement(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise RuntimeError('Reference metric must be positive')
    return float((reference - candidate) / reference)


def _artifact_path(
    model_contract: dict[str, Any],
    output_name: str,
) -> Path:
    filename = str(model_contract['outputs'][output_name])
    return OUTPUT_DIRECTORY / filename


def main() -> None:
    silver = pd.read_parquet(SILVER_PATH)
    model_contract = load_yaml(MODEL_CONTRACT_PATH)
    target_contract = load_yaml(TARGET_CONTRACT_PATH)

    folds = build_expanding_window_folds(
        silver['effective_timestamp'],
        model_contract,
    )
    evaluation = evaluate_hist_gradient_boosting_candidate(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )

    candidate = evaluation.regression_results.sort_values(
        'fold_id',
        kind='stable',
    ).reset_index(drop=True)
    predictions = evaluation.predictions.sort_values(
        ['row_position', 'fold_id'],
        kind='stable',
    ).reset_index(drop=True)

    benchmark_results = pd.read_csv(BENCHMARK_RESULTS_PATH)
    persistence = benchmark_results.loc[
        benchmark_results['benchmark'].eq('persistence')
    ].sort_values(
        'fold_id',
        kind='stable',
    ).reset_index(drop=True)
    ridge = pd.read_csv(RIDGE_RESULTS_PATH).sort_values(
        'fold_id',
        kind='stable',
    ).reset_index(drop=True)
    elastic_net = pd.read_csv(ELASTIC_NET_RESULTS_PATH).sort_values(
        'fold_id',
        kind='stable',
    ).reset_index(drop=True)

    compared = {
        'hist_gradient_boosting': candidate,
        'persistence': persistence,
        'ridge': ridge,
        'elastic_net': elastic_net,
    }

    for frame_name, frame in compared.items():
        if len(frame) != len(folds):
            raise RuntimeError(
                f'{frame_name} evidence does not cover every fold'
            )

    for frame_name, frame in compared.items():
        if not candidate['fold_id'].equals(frame['fold_id']):
            raise RuntimeError(
                f'Candidate and {frame_name} fold identifiers differ'
            )

    if candidate['internal_early_stopping'].any():
        raise RuntimeError(
            'Histogram boosting activated internal early stopping'
        )

    comparison = candidate.copy()
    comparison['reference'] = 'persistence'
    comparison['reference_mae'] = persistence['mae']
    comparison['reference_peak_mae'] = persistence['peak_mae']
    comparison['reference_maximum_rolling_96_mae'] = persistence[
        'maximum_rolling_96_mae'
    ]
    comparison['ridge_mae'] = ridge['mae']
    comparison['elastic_net_mae'] = elastic_net['mae']
    comparison['mae_difference'] = (
        comparison['mae'] - comparison['reference_mae']
    )
    comparison['peak_mae_difference'] = (
        comparison['peak_mae'] - comparison['reference_peak_mae']
    )

    maximum_origin = int(predictions['row_position'].max())
    peak_horizon_steps = int(
        model_contract['objectives']['classification_horizon_minutes']
        // 15
    )
    maximum_target_dependency = maximum_origin + peak_horizon_steps

    if maximum_origin >= folds[0].test_purge_start:
        raise RuntimeError(
            'Histogram boosting predictions enter the test purge'
        )

    if maximum_target_dependency >= folds[0].test_start:
        raise RuntimeError(
            'Histogram boosting labels consume locked-test rows'
        )

    candidate_mean_mae = float(candidate['mae'].mean())
    reference_mean_mae = float(persistence['mae'].mean())
    candidate_mean_peak_mae = float(candidate['peak_mae'].mean())
    reference_mean_peak_mae = float(persistence['peak_mae'].mean())
    candidate_worst_fold_mae = float(candidate['mae'].max())
    reference_worst_fold_mae = float(persistence['mae'].max())
    ridge_mean_mae = float(ridge['mae'].mean())
    elastic_net_mean_mae = float(elastic_net['mae'].mean())

    relative_mae_improvement = _relative_improvement(
        candidate_mean_mae,
        reference_mean_mae,
    )
    relative_peak_mae_change = _relative_change(
        candidate_mean_peak_mae,
        reference_mean_peak_mae,
    )
    relative_worst_fold_mae_change = _relative_change(
        candidate_worst_fold_mae,
        reference_worst_fold_mae,
    )
    relative_mae_improvement_over_ridge = _relative_improvement(
        candidate_mean_mae,
        ridge_mean_mae,
    )
    relative_mae_improvement_over_elastic_net = _relative_improvement(
        candidate_mean_mae,
        elastic_net_mean_mae,
    )

    promotion_contract = model_contract['promotion']
    aggregate_mae_pass = relative_mae_improvement > float(
        promotion_contract['minimum_relative_mae_improvement']
    )
    peak_mae_pass = relative_peak_mae_change <= float(
        promotion_contract['maximum_relative_peak_mae_degradation']
    )
    worst_fold_mae_pass = relative_worst_fold_mae_change <= float(
        promotion_contract[
            'maximum_relative_worst_fold_mae_degradation'
        ]
    )
    promotion_pass = bool(
        aggregate_mae_pass
        and peak_mae_pass
        and worst_fold_mae_pass
    )

    checks = {
        'aggregate_mae_improvement': aggregate_mae_pass,
        'peak_mae_non_degradation': peak_mae_pass,
        'worst_fold_mae_non_degradation': worst_fold_mae_pass,
    }
    failed_criteria = [
        name for name, passed in checks.items() if not passed
    ]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results_path = _artifact_path(
        model_contract,
        'hist_gradient_boosting_validation_results',
    )
    manifest_path = _artifact_path(
        model_contract,
        'hist_gradient_boosting_candidate_manifest',
    )
    predictions_path = _artifact_path(
        model_contract,
        'hist_gradient_boosting_out_of_fold_predictions',
    )

    comparison.to_csv(results_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    manifest = {
        'contract_version': str(model_contract['contract_version']),
        'governance_gate': '4C3',
        'status': 'validated',
        'candidate': 'hist_gradient_boosting',
        'reference': 'persistence',
        'locked_test_evaluated': False,
        'validation_fold_count': int(len(folds)),
        'validation_origin_count': int(
            predictions['row_position'].nunique()
        ),
        'prediction_row_count': int(len(predictions)),
        'maximum_prediction_origin': maximum_origin,
        'maximum_target_dependency': maximum_target_dependency,
        'test_purge_start': int(folds[0].test_purge_start),
        'test_purge_stop': int(folds[0].test_purge_stop),
        'locked_test_start': int(folds[0].test_start),
        'locked_test_stop': int(folds[0].test_stop),
        'search_governance': {
            'strategy': 'chronological_grid_search',
            'inner_splits': int(
                model_contract['candidate_selection']['inner_splits']
            ),
            'inner_gap_steps': int(
                model_contract['candidate_selection']['inner_gap_steps']
            ),
            'parameter_combinations': int(
                candidate['inner_parameter_count'].iloc[0]
            ),
            'internal_early_stopping': False,
            'scoring': 'neg_mean_absolute_error',
        },
        'selected_parameters_by_fold': {
            str(int(row.fold_id)): {
                'learning_rate': float(row.selected_learning_rate),
                'max_iter': int(row.selected_max_iter),
                'max_leaf_nodes': int(row.selected_max_leaf_nodes),
                'l2_regularization': float(
                    row.selected_l2_regularization
                ),
                'inner_validation_mae': float(
                    row.inner_validation_mae
                ),
                'fitted_iteration_count': int(
                    row.fitted_iteration_count
                ),
                'feature_count': int(row.feature_count),
            }
            for row in candidate.itertuples(index=False)
        },
        'candidate_metrics': {
            'mean_mae': candidate_mean_mae,
            'mean_peak_mae': candidate_mean_peak_mae,
            'worst_fold_mae': candidate_worst_fold_mae,
        },
        'reference_metrics': {
            'mean_mae': reference_mean_mae,
            'mean_peak_mae': reference_mean_peak_mae,
            'worst_fold_mae': reference_worst_fold_mae,
        },
        'relative_evidence': {
            'mae_improvement': relative_mae_improvement,
            'peak_mae_change': relative_peak_mae_change,
            'worst_fold_mae_change': relative_worst_fold_mae_change,
        },
        'linear_candidate_diagnostics': {
            'ridge_mean_mae': ridge_mean_mae,
            'elastic_net_mean_mae': elastic_net_mean_mae,
            'mae_improvement_over_ridge': (
                relative_mae_improvement_over_ridge
            ),
            'mae_improvement_over_elastic_net': (
                relative_mae_improvement_over_elastic_net
            ),
        },
        'promotion_checks': checks,
        'promotion_decision': (
            'promoted' if promotion_pass else 'rejected'
        ),
        'failed_promotion_criteria': failed_criteria,
        'artifacts': {
            'hist_gradient_boosting_validation_results': str(
                results_path.relative_to(ROOT).as_posix()
            ),
            'hist_gradient_boosting_out_of_fold_predictions': str(
                predictions_path.relative_to(ROOT).as_posix()
            ),
            'benchmark_manifest': str(
                BENCHMARK_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
            'ridge_candidate_manifest': str(
                RIDGE_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
            'elastic_net_candidate_manifest': str(
                ELASTIC_NET_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print(
        'Histogram boosting evidence: PASS | origins={} | '
        'mean_mae={:.6f} | reference_mae={:.6f} | decision={}'.format(
            manifest['validation_origin_count'],
            candidate_mean_mae,
            reference_mean_mae,
            manifest['promotion_decision'],
        )
    )


if __name__ == '__main__':
    main()
