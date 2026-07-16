from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

from iaei.modeling.benchmarks import _mase_denominator, _regression_metrics
from iaei.modeling.candidates import (
    build_feature_preprocessor,
    build_inner_time_series_split,
)
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class ElasticNetCandidateError(RuntimeError):
    '''Raised when Elastic Net evaluation violates the model contract.'''


@dataclass(frozen=True)
class ElasticNetCandidateEvaluation:
    regression_results: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ElasticNetFit:
    model: ElasticNet
    converged: bool
    iterations: int
    dual_gap: float


def _feature_frame(
    silver: pd.DataFrame,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    feature_policy = model_contract['feature_policy']
    numeric_features = [
        str(value) for value in feature_policy['numeric_features']
    ]
    categorical_features = [
        str(value) for value in feature_policy['categorical_features']
    ]
    requested = numeric_features + categorical_features
    missing = sorted(set(requested).difference(silver.columns))

    if missing:
        raise ElasticNetCandidateError(
            f'Silver candidate fields are missing: {missing}'
        )

    features = silver.loc[:, requested].copy()

    for column in numeric_features:
        features[column] = pd.to_numeric(
            features[column],
            errors='raise',
        )

    for column in categorical_features:
        values = features[column].astype('object')
        features[column] = values.where(pd.notna(values), np.nan)

    return features


def _positive_grid(
    values: list[Any],
    *,
    name: str,
) -> list[float]:
    numeric = [float(value) for value in values]

    if not numeric:
        raise ElasticNetCandidateError(f'{name} is empty')

    if not all(np.isfinite(value) and value > 0.0 for value in numeric):
        raise ElasticNetCandidateError(
            f'{name} values must be finite and positive'
        )

    if len(numeric) != len(set(numeric)):
        raise ElasticNetCandidateError(f'{name} contains duplicates')

    return numeric


def _elastic_net_grid(
    model_contract: dict[str, Any],
) -> tuple[list[float], list[float]]:
    selection = model_contract['candidate_selection']
    alphas = _positive_grid(
        selection['elastic_net_alpha_grid'],
        name='Elastic Net alpha grid',
    )
    l1_ratios = _positive_grid(
        selection['elastic_net_l1_ratio_grid'],
        name='Elastic Net l1-ratio grid',
    )

    if not all(0.0 < value < 1.0 for value in l1_ratios):
        raise ElasticNetCandidateError(
            'Elastic Net l1 ratios must lie strictly within (0, 1)'
        )

    if str(selection['elastic_net_alpha_path_order']) != 'descending':
        raise ElasticNetCandidateError(
            'Elastic Net alpha path must be descending'
        )

    return sorted(alphas, reverse=True), l1_ratios


def _estimator_controls(
    model_contract: dict[str, Any],
) -> dict[str, Any]:
    selection = model_contract['candidate_selection']
    max_iter = int(selection['elastic_net_max_iter'])
    tolerance = float(selection['elastic_net_tolerance'])
    coordinate_selection = str(selection['elastic_net_selection'])
    warm_start = bool(selection['elastic_net_warm_start'])
    require_convergence = bool(
        selection['elastic_net_require_convergence']
    )
    precompute = bool(selection['elastic_net_precompute'])

    if max_iter < 1:
        raise ElasticNetCandidateError(
            'Elastic Net max_iter must be positive'
        )

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ElasticNetCandidateError(
            'Elastic Net tolerance must be finite and positive'
        )

    if coordinate_selection != 'cyclic':
        raise ElasticNetCandidateError(
            'Elastic Net coordinate selection must be cyclic'
        )

    if not warm_start:
        raise ElasticNetCandidateError(
            'Elastic Net warm-start path must be enabled'
        )

    if not require_convergence:
        raise ElasticNetCandidateError(
            'Elastic Net convergence must be required'
        )

    return {
        'max_iter': max_iter,
        'tol': tolerance,
        'selection': coordinate_selection,
        'warm_start': warm_start,
        'precompute': precompute,
    }


def _fit_model(
    model: ElasticNet,
    features: np.ndarray,
    target: np.ndarray,
) -> ElasticNetFit:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(features, target)

    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in caught
    )
    iterations = int(np.max(np.asarray(model.n_iter_)))
    dual_gap = float(np.max(np.asarray(model.dual_gap_)))
    converged = bool(
        not convergence_warning
        and iterations < int(model.max_iter)
        and np.isfinite(dual_gap)
    )

    return ElasticNetFit(
        model=model,
        converged=converged,
        iterations=iterations,
        dual_gap=dual_gap,
    )


def _inner_parameter_evidence(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    alphas, l1_ratios = _elastic_net_grid(model_contract)
    controls = _estimator_controls(model_contract)
    splitter = build_inner_time_series_split(model_contract)
    evidence_rows: list[dict[str, Any]] = []

    for inner_fold_id, (train_positions, validation_positions) in enumerate(
        splitter.split(training_features),
        start=1,
    ):
        inner_train_features = training_features.iloc[train_positions]
        inner_validation_features = training_features.iloc[
            validation_positions
        ]
        inner_train_target = training_target.iloc[
            train_positions
        ].to_numpy(dtype=float)
        inner_validation_target = training_target.iloc[
            validation_positions
        ].to_numpy(dtype=float)

        preprocessor = build_feature_preprocessor(model_contract)
        transformed_train = preprocessor.fit_transform(
            inner_train_features
        )
        transformed_validation = preprocessor.transform(
            inner_validation_features
        )

        for l1_ratio in l1_ratios:
            model = ElasticNet(
                l1_ratio=l1_ratio,
                **controls,
            )

            for alpha in alphas:
                model.set_params(alpha=alpha)
                fit = _fit_model(
                    model,
                    np.asarray(transformed_train),
                    inner_train_target,
                )
                validation_prediction = fit.model.predict(
                    transformed_validation
                )
                validation_mae = float(
                    mean_absolute_error(
                        inner_validation_target,
                        validation_prediction,
                    )
                )

                evidence_rows.append(
                    {
                        'inner_fold_id': inner_fold_id,
                        'alpha': float(alpha),
                        'l1_ratio': float(l1_ratio),
                        'mae': validation_mae,
                        'converged': fit.converged,
                        'iterations': fit.iterations,
                        'dual_gap': fit.dual_gap,
                    }
                )

    return pd.DataFrame(evidence_rows)


def _select_parameters(
    evidence: pd.DataFrame,
    model_contract: dict[str, Any],
) -> dict[str, float | int]:
    expected_folds = int(
        model_contract['candidate_selection']['inner_splits']
    )
    grouped = evidence.groupby(
        ['alpha', 'l1_ratio'],
        as_index=False,
        sort=False,
    ).agg(
        inner_fold_count=('inner_fold_id', 'nunique'),
        converged_fold_count=('converged', 'sum'),
        mean_mae=('mae', 'mean'),
        mean_iterations=('iterations', 'mean'),
        maximum_dual_gap=('dual_gap', 'max'),
    )

    eligible = grouped.loc[
        grouped['inner_fold_count'].eq(expected_folds)
        & grouped['converged_fold_count'].eq(expected_folds)
    ].copy()

    if eligible.empty:
        raise ElasticNetCandidateError(
            'No Elastic Net parameter combination converged in every inner fold'
        )

    tie_break = str(
        model_contract['candidate_selection']['elastic_net_tie_break']
    )

    if tie_break != 'stronger_regularization':
        raise ElasticNetCandidateError(
            'Elastic Net tie break must favor stronger regularization'
        )

    eligible = eligible.sort_values(
        ['mean_mae', 'alpha', 'l1_ratio'],
        ascending=[True, False, False],
        kind='stable',
    ).reset_index(drop=True)
    selected = eligible.iloc[0]

    return {
        'alpha': float(selected['alpha']),
        'l1_ratio': float(selected['l1_ratio']),
        'inner_validation_mae': float(selected['mean_mae']),
        'selected_inner_mean_iterations': float(
            selected['mean_iterations']
        ),
        'selected_inner_maximum_dual_gap': float(
            selected['maximum_dual_gap']
        ),
        'inner_parameter_count': int(len(grouped)),
        'inner_converged_parameter_count': int(len(eligible)),
        'inner_total_fit_count': int(len(evidence)),
        'inner_nonconverged_fit_count': int(
            (~evidence['converged']).sum()
        ),
    }


def _fit_outer_path(
    transformed_features: np.ndarray,
    target: np.ndarray,
    *,
    selected_alpha: float,
    selected_l1_ratio: float,
    model_contract: dict[str, Any],
) -> ElasticNetFit:
    alphas, _ = _elastic_net_grid(model_contract)
    controls = _estimator_controls(model_contract)
    model = ElasticNet(
        l1_ratio=selected_l1_ratio,
        **controls,
    )
    final_fit: ElasticNetFit | None = None

    for alpha in alphas:
        if alpha < selected_alpha:
            continue

        model.set_params(alpha=alpha)
        final_fit = _fit_model(
            model,
            transformed_features,
            target,
        )

    if final_fit is None:
        raise ElasticNetCandidateError(
            'Selected Elastic Net alpha was not present in the path'
        )

    if not final_fit.converged:
        raise ElasticNetCandidateError(
            'Selected Elastic Net outer-fold fit did not converge'
        )

    return final_fit


def evaluate_elastic_net_candidate(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict[str, Any],
    *,
    target_contract: dict[str, Any] | None = None,
) -> ElasticNetCandidateEvaluation:
    required = {'effective_timestamp', 'usage_kwh'}
    missing_required = sorted(required.difference(silver.columns))

    if missing_required:
        raise ElasticNetCandidateError(
            f'Silver candidate fields are missing: {missing_required}'
        )

    if not folds:
        raise ElasticNetCandidateError(
            'No chronological folds were supplied'
        )

    features = _feature_frame(silver, model_contract)
    timestamps = pd.to_datetime(
        silver['effective_timestamp'],
        errors='raise',
    )
    usage = pd.to_numeric(
        silver['usage_kwh'],
        errors='raise',
    ).astype(float)
    regression_name = str(
        model_contract['objectives']['regression_target']
    )
    daily_lag = int(
        model_contract['regression_ladder']['seasonal_daily_lag_steps']
    )

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        if fold.validation_stop > fold.test_purge_start:
            raise ElasticNetCandidateError(
                'Validation rows enter the locked-test purge'
            )

        training_mask = pd.Series(False, index=silver.index)
        training_mask.iloc[fold.train_start : fold.train_stop] = True
        target_artifacts = build_supervised_targets(
            silver,
            training_mask,
            contract=target_contract,
        )
        targets = target_artifacts.frame
        peak_threshold = float(target_artifacts.peak_threshold_kwh)
        training_index = silver.index[
            fold.train_start : fold.train_stop
        ]
        validation_index = silver.index[
            fold.validation_start : fold.validation_stop
        ]

        training_target = targets.loc[
            training_index,
            regression_name,
        ].astype(float)
        valid_training = training_target.notna()

        if not valid_training.any():
            raise ElasticNetCandidateError(
                f'Fold {fold.fold_id} has no valid training labels'
            )

        outer_training_features = (
            features.loc[training_index]
            .loc[valid_training]
            .reset_index(drop=True)
        )
        outer_training_target = (
            training_target.loc[valid_training]
            .reset_index(drop=True)
        )
        validation_target = targets.loc[
            validation_index,
            regression_name,
        ].astype(float)

        if validation_target.isna().any():
            raise ElasticNetCandidateError(
                f'Fold {fold.fold_id} has missing validation labels'
            )

        inner_evidence = _inner_parameter_evidence(
            outer_training_features,
            outer_training_target,
            model_contract,
        )
        selected = _select_parameters(
            inner_evidence,
            model_contract,
        )

        outer_preprocessor = build_feature_preprocessor(model_contract)
        transformed_outer_training = outer_preprocessor.fit_transform(
            outer_training_features
        )
        transformed_validation = outer_preprocessor.transform(
            features.loc[validation_index]
        )
        outer_fit = _fit_outer_path(
            np.asarray(transformed_outer_training),
            outer_training_target.to_numpy(dtype=float),
            selected_alpha=float(selected['alpha']),
            selected_l1_ratio=float(selected['l1_ratio']),
            model_contract=model_contract,
        )

        coefficients = np.asarray(
            outer_fit.model.coef_,
            dtype=float,
        ).ravel()
        coefficient_count = int(coefficients.size)

        if coefficient_count < 1:
            raise ElasticNetCandidateError(
                f'Fold {fold.fold_id} produced no coefficients'
            )

        nonzero_count = int(
            np.count_nonzero(np.abs(coefficients) > 1e-12)
        )
        zero_count = coefficient_count - nonzero_count
        sparsity_ratio = float(zero_count / coefficient_count)
        prediction = pd.Series(
            outer_fit.model.predict(transformed_validation),
            index=validation_index,
            dtype=float,
        )
        peak_state = validation_target.ge(peak_threshold)
        mase_scale = _mase_denominator(usage, fold, daily_lag)
        metrics = _regression_metrics(
            validation_target,
            prediction,
            peak_mask=peak_state,
            mase_denominator=mase_scale,
            rolling_window=daily_lag,
        )

        result_rows.append(
            {
                'fold_id': fold.fold_id,
                'candidate': 'elastic_net',
                'selected_alpha': float(selected['alpha']),
                'selected_l1_ratio': float(selected['l1_ratio']),
                'inner_validation_mae': float(
                    selected['inner_validation_mae']
                ),
                'inner_parameter_count': int(
                    selected['inner_parameter_count']
                ),
                'inner_converged_parameter_count': int(
                    selected['inner_converged_parameter_count']
                ),
                'inner_total_fit_count': int(
                    selected['inner_total_fit_count']
                ),
                'inner_nonconverged_fit_count': int(
                    selected['inner_nonconverged_fit_count']
                ),
                'selected_inner_mean_iterations': float(
                    selected['selected_inner_mean_iterations']
                ),
                'selected_inner_maximum_dual_gap': float(
                    selected['selected_inner_maximum_dual_gap']
                ),
                'outer_fit_converged': outer_fit.converged,
                'outer_fit_iterations': outer_fit.iterations,
                'outer_fit_dual_gap': outer_fit.dual_gap,
                'coefficient_count': coefficient_count,
                'coefficient_nonzero_count': nonzero_count,
                'coefficient_zero_count': zero_count,
                'coefficient_sparsity_ratio': sparsity_ratio,
                'peak_threshold_kwh': peak_threshold,
                'mase_denominator': mase_scale,
                **metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    'fold_id': fold.fold_id,
                    'row_position': validation_index.to_numpy(),
                    'prediction_origin': timestamps.loc[
                        validation_index
                    ].to_numpy(),
                    'task': 'regression',
                    'candidate': 'elastic_net',
                    'actual': validation_target.to_numpy(),
                    'prediction': prediction.to_numpy(),
                    'peak_threshold_kwh': peak_threshold,
                    'is_peak_state': peak_state.to_numpy(),
                    'selected_alpha': float(selected['alpha']),
                    'selected_l1_ratio': float(selected['l1_ratio']),
                    'coefficient_sparsity_ratio': sparsity_ratio,
                }
            )
        )

    return ElasticNetCandidateEvaluation(
        regression_results=pd.DataFrame(result_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )
