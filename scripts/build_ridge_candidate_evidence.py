from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iaei.contracts import load_yaml
from iaei.modeling.candidates import evaluate_ridge_candidate
from iaei.modeling.splits import build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
SILVER_PATH = ROOT / 'data' / 'processed' / 'steel_energy_silver.parquet'
MODEL_CONTRACT_PATH = ROOT / 'configs' / 'model_contract.yml'
TARGET_CONTRACT_PATH = ROOT / 'configs' / 'target_contract.yml'
BENCHMARK_RESULTS_PATH = (
    ROOT / 'outputs' / 'modeling' / 'regression_validation_results.csv'
)
BENCHMARK_MANIFEST_PATH = (
    ROOT / 'outputs' / 'modeling' / 'benchmark_manifest.json'
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
    evaluation = evaluate_ridge_candidate(
        silver,
        folds,
        model_contract,
        target_contract=target_contract,
    )

    ridge = evaluation.regression_results.sort_values(
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

    if len(ridge) != len(folds):
        raise RuntimeError('Ridge evidence does not cover every fold')

    if len(persistence) != len(folds):
        raise RuntimeError('Persistence evidence does not cover every fold')

    if not ridge['fold_id'].equals(persistence['fold_id']):
        raise RuntimeError('Ridge and persistence fold identifiers differ')

    comparison = ridge.copy()
    comparison['reference'] = 'persistence'
    comparison['reference_mae'] = persistence['mae']
    comparison['reference_peak_mae'] = persistence['peak_mae']
    comparison['reference_maximum_rolling_96_mae'] = persistence[
        'maximum_rolling_96_mae'
    ]
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
        raise RuntimeError('Ridge predictions enter the locked-test purge')

    if maximum_target_dependency >= folds[0].test_start:
        raise RuntimeError(
            'Ridge validation labels consume locked-test observations'
        )

    ridge_mean_mae = float(ridge['mae'].mean())
    reference_mean_mae = float(persistence['mae'].mean())
    ridge_mean_peak_mae = float(ridge['peak_mae'].mean())
    reference_mean_peak_mae = float(persistence['peak_mae'].mean())
    ridge_worst_fold_mae = float(ridge['mae'].max())
    reference_worst_fold_mae = float(persistence['mae'].max())

    relative_mae_improvement = float(
        (reference_mean_mae - ridge_mean_mae) / reference_mean_mae
    )
    relative_peak_mae_change = _relative_change(
        ridge_mean_peak_mae,
        reference_mean_peak_mae,
    )
    relative_worst_fold_mae_change = _relative_change(
        ridge_worst_fold_mae,
        reference_worst_fold_mae,
    )

    promotion_contract = model_contract['promotion']
    minimum_mae_improvement = float(
        promotion_contract['minimum_relative_mae_improvement']
    )
    maximum_peak_degradation = float(
        promotion_contract['maximum_relative_peak_mae_degradation']
    )
    maximum_worst_fold_degradation = float(
        promotion_contract[
            'maximum_relative_worst_fold_mae_degradation'
        ]
    )

    aggregate_mae_pass = (
        relative_mae_improvement > minimum_mae_improvement
    )
    peak_mae_pass = (
        relative_peak_mae_change <= maximum_peak_degradation
    )
    worst_fold_mae_pass = (
        relative_worst_fold_mae_change
        <= maximum_worst_fold_degradation
    )

    promotion_pass = bool(
        aggregate_mae_pass
        and peak_mae_pass
        and worst_fold_mae_pass
    )

    failed_criteria = [
        name
        for name, passed in {
            'aggregate_mae_improvement': aggregate_mae_pass,
            'peak_mae_non_degradation': peak_mae_pass,
            'worst_fold_mae_non_degradation': worst_fold_mae_pass,
        }.items()
        if not passed
    ]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results_path = _artifact_path(
        model_contract,
        'ridge_validation_results',
    )
    manifest_path = _artifact_path(
        model_contract,
        'ridge_candidate_manifest',
    )
    predictions_path = _artifact_path(
        model_contract,
        'ridge_out_of_fold_predictions',
    )

    comparison.to_csv(results_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    manifest = {
        'contract_version': str(model_contract['contract_version']),
        'governance_gate': '4C1',
        'status': 'validated',
        'candidate': 'ridge',
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
        'selected_alpha_by_fold': {
            str(int(row.fold_id)): float(row.selected_alpha)
            for row in ridge.itertuples(index=False)
        },
        'candidate_metrics': {
            'mean_mae': ridge_mean_mae,
            'mean_peak_mae': ridge_mean_peak_mae,
            'worst_fold_mae': ridge_worst_fold_mae,
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
        'promotion_checks': {
            'aggregate_mae_improvement': aggregate_mae_pass,
            'peak_mae_non_degradation': peak_mae_pass,
            'worst_fold_mae_non_degradation': worst_fold_mae_pass,
        },
        'promotion_decision': (
            'promoted' if promotion_pass else 'rejected'
        ),
        'failed_promotion_criteria': failed_criteria,
        'artifacts': {
            'ridge_validation_results': str(
                results_path.relative_to(ROOT).as_posix()
            ),
            'ridge_out_of_fold_predictions': str(
                predictions_path.relative_to(ROOT).as_posix()
            ),
            'benchmark_manifest': str(
                BENCHMARK_MANIFEST_PATH.relative_to(ROOT).as_posix()
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print(
        'Ridge evidence: PASS | origins={} | mean_mae={:.6f} | '
        'reference_mae={:.6f} | decision={}'.format(
            manifest['validation_origin_count'],
            ridge_mean_mae,
            reference_mean_mae,
            manifest['promotion_decision'],
        )
    )


if __name__ == '__main__':
    main()
