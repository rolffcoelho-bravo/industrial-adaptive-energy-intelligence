from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from iaei.modeling.benchmarks import _mase_denominator, _regression_metrics
from iaei.modeling.splits import ChronologicalFold
from iaei.targets import build_supervised_targets


class CandidateModelError(RuntimeError):
    '''Raised when candidate-model evaluation violates the model contract.'''


@dataclass(frozen=True)
class RidgeCandidateEvaluation:
    regression_results: pd.DataFrame
    predictions: pd.DataFrame


def _feature_names(
    model_contract: dict[str, Any],
) -> tuple[list[str], list[str]]:
    feature_policy = model_contract['feature_policy']
    numeric_features = [str(value) for value in feature_policy['numeric_features']]
    categorical_features = [
        str(value) for value in feature_policy['categorical_features']
    ]

    if not numeric_features:
        raise CandidateModelError('Numeric feature collection is empty')

    if not categorical_features:
        raise CandidateModelError('Categorical feature collection is empty')

    overlap = sorted(set(numeric_features).intersection(categorical_features))
    if overlap:
        raise CandidateModelError(f'Features have conflicting types: {overlap}')

    return numeric_features, categorical_features


def build_feature_preprocessor(
    model_contract: dict[str, Any],
) -> ColumnTransformer:
    numeric_features, categorical_features = _feature_names(model_contract)

    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            (
                'encoder',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features),
        ],
        remainder='drop',
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_inner_time_series_split(
    model_contract: dict[str, Any],
) -> TimeSeriesSplit:
    selection = model_contract['candidate_selection']

    if str(selection['inner_split_type']) != 'time_series':
        raise CandidateModelError('Inner split type must be time_series')

    split_count = int(selection['inner_splits'])
    gap_steps = int(selection['inner_gap_steps'])

    if split_count < 2:
        raise CandidateModelError('Inner validation requires at least two splits')

    if gap_steps < 0:
        raise CandidateModelError('Inner validation gap cannot be negative')

    return TimeSeriesSplit(
        n_splits=split_count,
        gap=gap_steps,
    )


def _candidate_feature_frame(
    silver: pd.DataFrame,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    numeric_features, categorical_features = _feature_names(model_contract)
    requested = numeric_features + categorical_features
    missing = sorted(set(requested).difference(silver.columns))

    if missing:
        raise CandidateModelError(f'Silver candidate fields are missing: {missing}')

    features = silver.loc[:, requested].copy()

    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors='raise')

    for column in categorical_features:
        values = features[column].astype('object')
        features[column] = values.where(pd.notna(values), np.nan)

    return features


def _ridge_alpha_grid(model_contract: dict[str, Any]) -> list[float]:
    values = [
        float(value)
        for value in model_contract['candidate_selection']['ridge_alpha_grid']
    ]

    if not values:
        raise CandidateModelError('Ridge alpha grid is empty')

    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise CandidateModelError('Ridge alpha values must be finite and positive')

    if len(values) != len(set(values)):
        raise CandidateModelError('Ridge alpha grid contains duplicates')

    return values


def evaluate_ridge_candidate(
    silver: pd.DataFrame,
    folds: list[ChronologicalFold],
    model_contract: dict[str, Any],
    *,
    target_contract: dict[str, Any] | None = None,
) -> RidgeCandidateEvaluation:
    required = {'effective_timestamp', 'usage_kwh'}
    missing_required = sorted(required.difference(silver.columns))

    if missing_required:
        raise CandidateModelError(
            f'Silver candidate fields are missing: {missing_required}'
        )

    if not folds:
        raise CandidateModelError('No chronological folds were supplied')

    features = _candidate_feature_frame(silver, model_contract)
    timestamps = pd.to_datetime(silver['effective_timestamp'], errors='raise')
    usage = pd.to_numeric(silver['usage_kwh'], errors='raise').astype(float)
    regression_name = str(model_contract['objectives']['regression_target'])
    daily_lag = int(
        model_contract['regression_ladder']['seasonal_daily_lag_steps']
    )
    rolling_window = daily_lag
    alpha_grid = _ridge_alpha_grid(model_contract)
    scoring = str(model_contract['candidate_selection']['scoring'])

    if scoring != 'neg_mean_absolute_error':
        raise CandidateModelError(
            'Ridge selection scoring must be neg_mean_absolute_error'
        )

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        if fold.validation_stop > fold.test_purge_start:
            raise CandidateModelError(
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
        training_index = silver.index[fold.train_start : fold.train_stop]
        validation_index = silver.index[
            fold.validation_start : fold.validation_stop
        ]

        training_target = targets.loc[training_index, regression_name].astype(float)
        valid_training = training_target.notna()

        if not valid_training.any():
            raise CandidateModelError(
                f'Fold {fold.fold_id} has no valid Ridge training labels'
            )

        validation_target = targets.loc[
            validation_index, regression_name
        ].astype(float)

        if validation_target.isna().any():
            raise CandidateModelError(
                f'Fold {fold.fold_id} has missing Ridge validation labels'
            )

        pipeline = Pipeline(
            steps=[
                ('preprocessor', build_feature_preprocessor(model_contract)),
                ('model', Ridge()),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={'model__alpha': alpha_grid},
            scoring=scoring,
            cv=build_inner_time_series_split(model_contract),
            refit=bool(
                model_contract['candidate_selection']['refit_on_outer_training']
            ),
            n_jobs=1,
            error_score='raise',
            return_train_score=False,
        )
        search.fit(
            features.loc[training_index].loc[valid_training],
            training_target.loc[valid_training],
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
            rolling_window=rolling_window,
        )
        selected_alpha = float(search.best_params_['model__alpha'])
        inner_validation_mae = float(-search.best_score_)

        result_rows.append(
            {
                'fold_id': fold.fold_id,
                'candidate': 'ridge',
                'selected_alpha': selected_alpha,
                'inner_validation_mae': inner_validation_mae,
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
                    'candidate': 'ridge',
                    'actual': validation_target.to_numpy(),
                    'prediction': prediction.to_numpy(),
                    'peak_threshold_kwh': peak_threshold,
                    'is_peak_state': peak_state.to_numpy(),
                    'selected_alpha': selected_alpha,
                }
            )
        )

    return RidgeCandidateEvaluation(
        regression_results=pd.DataFrame(result_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )
