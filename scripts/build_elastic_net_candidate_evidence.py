from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.elastic_net import evaluate_elastic_net_candidate
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
OUTPUT_DIRECTORY = ROOT / 'outputs' / 'modeling'


def _relative_change(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        raise RuntimeError('Reference metric must be positive')
    return float((candidate - reference) / reference)


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
    evaluation = evaluate_elastic_net_candidate(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )

    elastic_net = evaluation.regression_results.sort_values(
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

    for frame_name, frame in {
        'elastic_net': elastic_net,
        'persistence': persistence,
        'ridge': ridge,
    }.items():
        if len(frame) != len(folds):
            raise RuntimeError(
                f'{frame_name} evidence does not cover every fold'
            )

    if not elastic_net['fold_id'].equals(persistence['fold_id']):
        raise RuntimeError(
            'Elastic Net and persistence fold identifiers differ'
        )

    if not elastic_net['fold_id'].equals(ridge['fold_id']):
        raise RuntimeError(
            'Elastic Net and Ridge fold identifiers differ'
        )

    if not elastic_net['outer_fit_converged'].all():
        raise RuntimeError(
            'An Elastic Net outer-fold fit did not converge'
        )

    comparison = elastic_net.copy()
    comparison['reference'] = 'persistence'
    comparison['reference_mae'] = persistence['mae']
    comparison['reference_peak_mae'] = persistence['peak_mae']
    comparison['reference_maximum_rolling_96_mae'] = persistence[
        'maximum_rolling_96_mae'
    ]
    comparison['ridge_mae'] = ridge['mae']
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
            'Elastic Net predictions enter the locked-test purge'
        )

    if maximum_target_dependency >= folds[0].test_start:
        raise RuntimeError(
            'Elastic Net labels consume locked-test observations'
        )

    candidate_mean_mae = float(elastic_net['mae'].mean())
    reference_mean_mae = float(persistence['mae'].mean())
    candidate_mean_peak_mae = float(
        elastic_net['peak_mae'].mean()
    )
    reference_mean_peak_mae = float(persistence['peak_mae'].mean())
    candidate_worst_fold_mae = float(elastic_net['mae'].max())
    reference_worst_fold_mae = float(persistence['mae'].max())
    ridge_mean_mae = float(ridge['mae'].mean())

    relative_mae_improvement = float(
        (reference_mean_mae - candidate_mean_mae)
        / reference_mean_mae
    )
    relative_peak_mae_change = _relative_change(
        candidate_mean_peak_mae,
        reference_mean_peak_mae,
    )
    relative_worst_fold_mae_change = _relative_change(
        candidate_worst_fold_mae,
        reference_worst_fold_mae,
    )
    relative_mae_improvement_over_ridge = float(
        (ridge_mean_mae - candidate_mean_mae) / ridge_mean_mae
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
        'elastic_net_validation_results',
    )
    manifest_path = _artifact_path(
        model_contract,
        'elastic_net_candidate_manifest',
    )
    predictions_path = _artifact_path(
        model_contract,
        'elastic_net_out_of_fold_predictions',
    )

    comparison.to_csv(results_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    manifest = {
        'contract_version': str(model_contract['contract_version']),
        'governance_gate': '4C2',
        'status': 'validated',
        'candidate': 'elastic_net',
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
            'strategy': 'chronological_warm_start_parameter_path',
            'inner_splits': int(
                model_contract['candidate_selection']['inner_splits']
            ),
            'inner_gap_steps': int(
                model_contract['candidate_selection']['inner_gap_steps']
            ),
            'alpha_path_order': 'descending',
            'nonconverged_parameters_eligible': False,
            'tie_break': 'stronger_regularization',
        },
        'selected_parameters_by_fold': {
            str(int(row.fold_id)): {
                'alpha': float(row.selected_alpha),
                'l1_ratio': float(row.selected_l1_ratio),
                'inner_parameter_count': int(
                    row.inner_parameter_count
                ),
                'inner_converged_parameter_count': int(
                    row.inner_converged_parameter_count
                ),
                'inner_nonconverged_fit_count': int(
                    row.inner_nonconverged_fit_count
                ),
                'outer_fit_converged': bool(
                    row.outer_fit_converged
                ),
                'outer_fit_iterations': int(
                    row.outer_fit_iterations
                ),
                'outer_fit_dual_gap': float(
                    row.outer_fit_dual_gap
                ),
                'coefficient_nonzero_count': int(
                    row.coefficient_nonzero_count
                ),
                'coefficient_count': int(row.coefficient_count),
                'coefficient_sparsity_ratio': float(
                    row.coefficient_sparsity_ratio
                ),
            }
            for row in elastic_net.itertuples(index=False)
        },
        'candidate_metrics': {
            'mean_mae': candidate_mean_mae,
            'mean_peak_mae': candidate_mean_peak_mae,
            'worst_fold_mae': candidate_worst_fold_mae,
            'mean_coefficient_sparsity_ratio': float(
                elastic_net['coefficient_sparsity_ratio'].mean()
            ),
            'total_nonconverged_inner_fits': int(
                elastic_net['inner_nonconverged_fit_count'].sum()
            ),
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
        'ridge_diagnostic': {
            'ridge_mean_mae': ridge_mean_mae,
            'mae_improvement_over_ridge': (
                relative_mae_improvement_over_ridge
            ),
        },
        'promotion_checks': checks,
        'promotion_decision': (
            'promoted' if promotion_pass else 'rejected'
        ),
        'failed_promotion_criteria': failed_criteria,
        'artifacts': {
            'elastic_net_validation_results': str(
                results_path.relative_to(ROOT).as_posix()
            ),
            'elastic_net_out_of_fold_predictions': str(
                predictions_path.relative_to(ROOT).as_posix()
            ),
            'benchmark_manifest': str(
                BENCHMARK_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
            'ridge_candidate_manifest': str(
                RIDGE_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print(
        'Elastic Net evidence: PASS | origins={} | mean_mae={:.6f} | '
        'reference_mae={:.6f} | nonconverged_inner_fits={} | '
        'decision={}'.format(
            manifest['validation_origin_count'],
            candidate_mean_mae,
            reference_mean_mae,
            manifest['candidate_metrics'][
                'total_nonconverged_inner_fits'
            ],
            manifest['promotion_decision'],
        )
    )


if __name__ == '__main__':
    main()
