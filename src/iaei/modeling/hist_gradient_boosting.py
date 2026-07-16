from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from iaei.modeling.benchmarks import _mase_denominator, _regression_metrics
from iaei.modeling.candidates import (
    build_feature_preprocessor,
    build_inner_time_series_split,
)
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class HistGradientBoostingCandidateError(RuntimeError):
    '''Raised when histogram boosting violates the model contract.'''


@dataclass(frozen=True)
class HistGradientBoostingCandidateEvaluation:
    regression_results: pd.DataFrame
    predictions: pd.DataFrame


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
        raise HistGradientBoostingCandidateError(
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


def _finite_float_grid(
    values: list[Any],
    *,
    name: str,
    allow_zero: bool,
) -> list[float]:
    numeric = [float(value) for value in values]

    if not numeric:
        raise HistGradientBoostingCandidateError(f'{name} is empty')

    if allow_zero:
        valid = all(
            np.isfinite(value) and value >= 0.0
            for value in numeric
        )
    else:
        valid = all(
            np.isfinite(value) and value > 0.0
            for value in numeric
        )

    if not valid:
        raise HistGradientBoostingCandidateError(
            f'{name} contains invalid values'
        )

    if len(numeric) != len(set(numeric)):
        raise HistGradientBoostingCandidateError(
            f'{name} contains duplicates'
        )

    return numeric


def _positive_integer_grid(
    values: list[Any],
    *,
    name: str,
    minimum: int,
) -> list[int]:
    numeric = [int(value) for value in values]

    if not numeric:
        raise HistGradientBoostingCandidateError(f'{name} is empty')

    if not all(value >= minimum for value in numeric):
        raise HistGradientBoostingCandidateError(
            f'{name} values must be at least {minimum}'
        )

    if len(numeric) != len(set(numeric)):
        raise HistGradientBoostingCandidateError(
            f'{name} contains duplicates'
        )

    return numeric


def _parameter_grid(
    model_contract: dict[str, Any],
) -> dict[str, list[float | int]]:
    selection = model_contract['candidate_selection']

    learning_rates = _finite_float_grid(
        selection['hist_gradient_boosting_learning_rate_grid'],
        name='Histogram boosting learning-rate grid',
        allow_zero=False,
    )
    max_iterations = _positive_integer_grid(
        selection['hist_gradient_boosting_max_iter_grid'],
        name='Histogram boosting max-iteration grid',
        minimum=1,
    )
    leaf_nodes = _positive_integer_grid(
        selection['hist_gradient_boosting_max_leaf_nodes_grid'],
        name='Histogram boosting leaf-node grid',
        minimum=2,
    )
    l2_values = _finite_float_grid(
        selection[
            'hist_gradient_boosting_l2_regularization_grid'
        ],
        name='Histogram boosting L2 grid',
        allow_zero=True,
    )

    return {
        'model__learning_rate': learning_rates,
        'model__max_iter': max_iterations,
        'model__max_leaf_nodes': leaf_nodes,
        'model__l2_regularization': l2_values,
    }


def build_hist_gradient_boosting_estimator(
    model_contract: dict[str, Any],
) -> HistGradientBoostingRegressor:
    selection = model_contract['candidate_selection']
    loss = str(selection['hist_gradient_boosting_loss'])
    min_samples_leaf = int(
        selection['hist_gradient_boosting_min_samples_leaf']
    )
    max_bins = int(selection['hist_gradient_boosting_max_bins'])
    max_features = float(
        selection['hist_gradient_boosting_max_features']
    )
    early_stopping = bool(
        selection['hist_gradient_boosting_early_stopping']
    )
    random_state = int(
        selection['hist_gradient_boosting_random_state']
    )

    if loss != 'absolute_error':
        raise HistGradientBoostingCandidateError(
            'Histogram boosting loss must be absolute_error'
        )

    if min_samples_leaf < 1:
        raise HistGradientBoostingCandidateError(
            'Histogram boosting min_samples_leaf must be positive'
        )

    if max_bins < 2 or max_bins > 255:
        raise HistGradientBoostingCandidateError(
            'Histogram boosting max_bins must lie within [2, 255]'
        )

    if not np.isfinite(max_features) or not 0.0 < max_features <= 1.0:
        raise HistGradientBoostingCandidateError(
            'Histogram boosting max_features must lie within (0, 1]'
        )

    if early_stopping:
        raise HistGradientBoostingCandidateError(
            'Histogram boosting internal early stopping is forbidden'
        )

    return HistGradientBoostingRegressor(
        loss=loss,
        min_samples_leaf=min_samples_leaf,
        max_bins=max_bins,
        max_features=max_features,
        categorical_features=None,
        early_stopping=False,
        random_state=random_state,
    )


def evaluate_hist_gradient_boosting_candidate(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict[str, Any],
    *,
    target_contract: dict[str, Any] | None = None,
) -> HistGradientBoostingCandidateEvaluation:
    required = {'effective_timestamp', 'usage_kwh'}
    missing_required = sorted(required.difference(silver.columns))

    if missing_required:
        raise HistGradientBoostingCandidateError(
            f'Silver candidate fields are missing: {missing_required}'
        )

    if not folds:
        raise HistGradientBoostingCandidateError(
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
    scoring = str(model_contract['candidate_selection']['scoring'])
    parameter_grid = _parameter_grid(model_contract)
    parameter_count = int(
        np.prod([len(values) for values in parameter_grid.values()])
    )
    inner_split_count = int(
        model_contract['candidate_selection']['inner_splits']
    )

    if scoring != 'neg_mean_absolute_error':
        raise HistGradientBoostingCandidateError(
            'Histogram boosting scoring must be neg_mean_absolute_error'
        )

    if not bool(
        model_contract['candidate_selection']['refit_on_outer_training']
    ):
        raise HistGradientBoostingCandidateError(
            'Histogram boosting must refit on the outer training fold'
        )

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        if fold.validation_stop > fold.test_purge_start:
            raise HistGradientBoostingCandidateError(
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
            raise HistGradientBoostingCandidateError(
                f'Fold {fold.fold_id} has no valid training labels'
            )

        validation_target = targets.loc[
            validation_index,
            regression_name,
        ].astype(float)

        if validation_target.isna().any():
            raise HistGradientBoostingCandidateError(
                f'Fold {fold.fold_id} has missing validation labels'
            )

        pipeline = Pipeline(
            steps=[
                (
                    'preprocessor',
                    build_feature_preprocessor(model_contract),
                ),
                (
                    'model',
                    build_hist_gradient_boosting_estimator(
                        model_contract
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            cv=build_inner_time_series_split(model_contract),
            refit=True,
            n_jobs=1,
            error_score='raise',
            return_train_score=False,
        )
        search.fit(
            features.loc[training_index].loc[valid_training],
            training_target.loc[valid_training],
        )

        best_pipeline = search.best_estimator_
        fitted_model = best_pipeline.named_steps['model']

        if bool(fitted_model.do_early_stopping_):
            raise HistGradientBoostingCandidateError(
                f'Fold {fold.fold_id} activated internal early stopping'
            )

        selected_learning_rate = float(
            search.best_params_['model__learning_rate']
        )
        selected_max_iter = int(
            search.best_params_['model__max_iter']
        )
        selected_max_leaf_nodes = int(
            search.best_params_['model__max_leaf_nodes']
        )
        selected_l2_regularization = float(
            search.best_params_['model__l2_regularization']
        )
        fitted_iteration_count = int(fitted_model.n_iter_)

        if fitted_iteration_count != selected_max_iter:
            raise HistGradientBoostingCandidateError(
                f'Fold {fold.fold_id} did not fit every selected iteration'
            )

        prediction = pd.Series(
            search.predict(features.loc[validation_index]),
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
                'candidate': 'hist_gradient_boosting',
                'selected_learning_rate': selected_learning_rate,
                'selected_max_iter': selected_max_iter,
                'selected_max_leaf_nodes': selected_max_leaf_nodes,
                'selected_l2_regularization': (
                    selected_l2_regularization
                ),
                'inner_validation_mae': float(-search.best_score_),
                'inner_parameter_count': parameter_count,
                'inner_split_count': inner_split_count,
                'inner_total_fit_count': (
                    parameter_count * inner_split_count
                ),
                'internal_early_stopping': bool(
                    fitted_model.do_early_stopping_
                ),
                'fitted_iteration_count': fitted_iteration_count,
                'feature_count': int(fitted_model.n_features_in_),
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
                    'candidate': 'hist_gradient_boosting',
                    'actual': validation_target.to_numpy(),
                    'prediction': prediction.to_numpy(),
                    'peak_threshold_kwh': peak_threshold,
                    'is_peak_state': peak_state.to_numpy(),
                    'selected_learning_rate': selected_learning_rate,
                    'selected_max_iter': selected_max_iter,
                    'selected_max_leaf_nodes': (
                        selected_max_leaf_nodes
                    ),
                    'selected_l2_regularization': (
                        selected_l2_regularization
                    ),
                }
            )
        )

    return HistGradientBoostingCandidateEvaluation(
        regression_results=pd.DataFrame(result_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )
