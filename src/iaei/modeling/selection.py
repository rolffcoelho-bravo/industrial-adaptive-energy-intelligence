from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from iaei.contracts import validate_target_contract
from iaei.modeling.candidates import (
    build_feature_preprocessor,
    build_inner_time_series_split,
)
from iaei.modeling.hist_gradient_boosting import (
    build_hist_gradient_boosting_estimator,
)
from iaei.targets import build_supervised_targets


class SelectedModelFreezeError(RuntimeError):
    '''Raised when the pre-test model freeze violates governance.'''


@dataclass(frozen=True)
class SelectionBoundary:
    test_purge_start: int
    test_purge_stop: int
    locked_test_start: int
    locked_test_stop: int
    row_count: int


@dataclass(frozen=True)
class SelectedModelFreeze:
    selected_parameters: dict[str, float | int | str]
    inner_validation_mae: float
    parameter_count: int
    inner_split_count: int
    total_inner_fit_count: int
    training_start: int
    training_stop: int
    training_origin_count: int
    training_label_count: int
    maximum_training_origin: int
    maximum_target_dependency: int
    selection_input_start: int
    selection_input_stop: int
    selection_input_row_count: int
    test_purge_start: int
    test_purge_stop: int
    locked_test_start: int
    locked_test_stop: int
    training_timestamp_start: str
    training_timestamp_end: str
    selection_input_timestamp_end: str
    feature_count: int
    fitted_iteration_count: int
    internal_early_stopping: bool


def _validate_boundary(boundary: SelectionBoundary) -> None:
    if boundary.row_count <= 0:
        raise SelectedModelFreezeError('Silver row count must be positive')

    if not (
        0
        < boundary.test_purge_start
        < boundary.test_purge_stop
        == boundary.locked_test_start
        < boundary.locked_test_stop
        == boundary.row_count
    ):
        raise SelectedModelFreezeError(
            'Selection and locked-test boundaries are inconsistent'
        )


def _feature_frame(
    frame: pd.DataFrame,
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
    missing = sorted(set(requested).difference(frame.columns))

    if missing:
        raise SelectedModelFreezeError(
            f'Silver candidate fields are missing: {missing}'
        )

    features = frame.loc[:, requested].copy()

    for column in numeric_features:
        features[column] = pd.to_numeric(
            features[column],
            errors='raise',
        )

    for column in categorical_features:
        values = features[column].astype('object')
        features[column] = values.where(pd.notna(values), np.nan)

    return features


def _float_grid(
    values: list[Any],
    *,
    name: str,
    allow_zero: bool,
) -> list[float]:
    numeric = [float(value) for value in values]

    if not numeric:
        raise SelectedModelFreezeError(f'{name} is empty')

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
        raise SelectedModelFreezeError(f'{name} is invalid')

    if len(numeric) != len(set(numeric)):
        raise SelectedModelFreezeError(f'{name} contains duplicates')

    return numeric


def _integer_grid(
    values: list[Any],
    *,
    name: str,
    minimum: int,
) -> list[int]:
    numeric = [int(value) for value in values]

    if not numeric:
        raise SelectedModelFreezeError(f'{name} is empty')

    if not all(value >= minimum for value in numeric):
        raise SelectedModelFreezeError(
            f'{name} values must be at least {minimum}'
        )

    if len(numeric) != len(set(numeric)):
        raise SelectedModelFreezeError(f'{name} contains duplicates')

    return numeric


def _parameter_grid(
    model_contract: dict[str, Any],
) -> dict[str, list[float | int]]:
    selection = model_contract['candidate_selection']

    return {
        'model__learning_rate': _float_grid(
            selection[
                'hist_gradient_boosting_learning_rate_grid'
            ],
            name='Histogram boosting learning-rate grid',
            allow_zero=False,
        ),
        'model__max_iter': _integer_grid(
            selection['hist_gradient_boosting_max_iter_grid'],
            name='Histogram boosting max-iteration grid',
            minimum=1,
        ),
        'model__max_leaf_nodes': _integer_grid(
            selection[
                'hist_gradient_boosting_max_leaf_nodes_grid'
            ],
            name='Histogram boosting leaf-node grid',
            minimum=2,
        ),
        'model__l2_regularization': _float_grid(
            selection[
                'hist_gradient_boosting_l2_regularization_grid'
            ],
            name='Histogram boosting L2 grid',
            allow_zero=True,
        ),
    }


def freeze_hist_gradient_boosting_selection(
    selection_frame: pd.DataFrame,
    boundary: SelectionBoundary,
    model_contract: dict[str, Any],
    *,
    target_contract: dict[str, Any] | None = None,
) -> SelectedModelFreeze:
    _validate_boundary(boundary)

    required = {'effective_timestamp', 'usage_kwh'}
    missing_required = sorted(
        required.difference(selection_frame.columns)
    )

    if missing_required:
        raise SelectedModelFreezeError(
            f'Silver candidate fields are missing: {missing_required}'
        )

    if len(selection_frame) != boundary.test_purge_stop:
        raise SelectedModelFreezeError(
            'Selection frame must stop at the locked-test boundary'
        )

    if not isinstance(selection_frame.index, pd.RangeIndex):
        raise SelectedModelFreezeError(
            'Selection frame must use a zero-based RangeIndex'
        )

    if selection_frame.index.start != 0:
        raise SelectedModelFreezeError(
            'Selection frame index must start at zero'
        )

    timestamps = pd.to_datetime(
        selection_frame['effective_timestamp'],
        errors='raise',
    )

    if timestamps.isna().any():
        raise SelectedModelFreezeError(
            'Selection timestamps contain missing values'
        )

    if not timestamps.is_monotonic_increasing:
        raise SelectedModelFreezeError(
            'Selection timestamps are not chronological'
        )

    if timestamps.duplicated().any():
        raise SelectedModelFreezeError(
            'Selection timestamps contain duplicates'
        )

    features = _feature_frame(selection_frame, model_contract)
    locked_target_contract = (
        target_contract or validate_target_contract()
    )
    regression_contract = locked_target_contract['targets'][
        'regression'
    ]
    peak_contract = locked_target_contract['targets']['peak_risk']
    regression_name = str(regression_contract['name'])

    dependency_steps = max(
        int(regression_contract['horizon_steps']),
        int(peak_contract['horizon_steps']),
    )

    training_start = 0
    training_stop = boundary.test_purge_start
    training_index = selection_frame.index[
        training_start:training_stop
    ]
    training_mask = pd.Series(False, index=selection_frame.index)
    training_mask.iloc[training_start:training_stop] = True

    target_artifacts = build_supervised_targets(
        selection_frame,
        training_mask,
        contract=locked_target_contract,
    )
    training_target = target_artifacts.frame.loc[
        training_index,
        regression_name,
    ].astype(float)
    valid_training = training_target.notna()

    if not valid_training.all():
        raise SelectedModelFreezeError(
            'Purged dependency rows are insufficient for all origins'
        )

    valid_index = training_index[valid_training.to_numpy()]
    maximum_training_origin = int(valid_index.max())
    maximum_target_dependency = (
        maximum_training_origin + dependency_steps
    )

    if maximum_training_origin >= boundary.test_purge_start:
        raise SelectedModelFreezeError(
            'Training origins enter the test-boundary purge'
        )

    if maximum_target_dependency >= boundary.locked_test_start:
        raise SelectedModelFreezeError(
            'Training targets consume locked-test observations'
        )

    if maximum_target_dependency >= len(selection_frame):
        raise SelectedModelFreezeError(
            'Selection frame does not contain required purge rows'
        )

    scoring = str(model_contract['candidate_selection']['scoring'])

    if scoring != 'neg_mean_absolute_error':
        raise SelectedModelFreezeError(
            'Final selection scoring must be neg_mean_absolute_error'
        )

    if not bool(
        model_contract['candidate_selection'][
            'refit_on_outer_training'
        ]
    ):
        raise SelectedModelFreezeError(
            'Final selection must refit on all training origins'
        )

    parameter_grid = _parameter_grid(model_contract)
    parameter_count = int(
        np.prod([len(values) for values in parameter_grid.values()])
    )
    inner_split_count = int(
        model_contract['candidate_selection']['inner_splits']
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
        features.loc[valid_index],
        training_target.loc[valid_index],
    )

    fitted_model = search.best_estimator_.named_steps['model']

    if bool(fitted_model.do_early_stopping_):
        raise SelectedModelFreezeError(
            'Final model activated internal early stopping'
        )

    selected_max_iter = int(
        search.best_params_['model__max_iter']
    )
    fitted_iteration_count = int(fitted_model.n_iter_)

    if fitted_iteration_count != selected_max_iter:
        raise SelectedModelFreezeError(
            'Final model did not fit every selected iteration'
        )

    selected_parameters: dict[str, float | int | str] = {
        'loss': str(fitted_model.loss),
        'learning_rate': float(
            search.best_params_['model__learning_rate']
        ),
        'max_iter': selected_max_iter,
        'max_leaf_nodes': int(
            search.best_params_['model__max_leaf_nodes']
        ),
        'l2_regularization': float(
            search.best_params_['model__l2_regularization']
        ),
        'min_samples_leaf': int(fitted_model.min_samples_leaf),
        'max_bins': int(fitted_model.max_bins),
        'max_features': float(fitted_model.max_features),
        'random_state': int(fitted_model.random_state),
    }

    return SelectedModelFreeze(
        selected_parameters=selected_parameters,
        inner_validation_mae=float(-search.best_score_),
        parameter_count=parameter_count,
        inner_split_count=inner_split_count,
        total_inner_fit_count=parameter_count * inner_split_count,
        training_start=training_start,
        training_stop=training_stop,
        training_origin_count=int(len(training_index)),
        training_label_count=int(valid_training.sum()),
        maximum_training_origin=maximum_training_origin,
        maximum_target_dependency=maximum_target_dependency,
        selection_input_start=0,
        selection_input_stop=len(selection_frame),
        selection_input_row_count=len(selection_frame),
        test_purge_start=boundary.test_purge_start,
        test_purge_stop=boundary.test_purge_stop,
        locked_test_start=boundary.locked_test_start,
        locked_test_stop=boundary.locked_test_stop,
        training_timestamp_start=timestamps.iloc[0].isoformat(),
        training_timestamp_end=(
            timestamps.iloc[maximum_training_origin].isoformat()
        ),
        selection_input_timestamp_end=(
            timestamps.iloc[-1].isoformat()
        ),
        feature_count=int(fitted_model.n_features_in_),
        fitted_iteration_count=fitted_iteration_count,
        internal_early_stopping=bool(
            fitted_model.do_early_stopping_
        ),
    )
