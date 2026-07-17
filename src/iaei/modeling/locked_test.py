from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from iaei.modeling.candidates import build_feature_preprocessor
from iaei.modeling.hist_gradient_boosting import (
    build_hist_gradient_boosting_estimator,
)
from iaei.targets import build_supervised_targets


class LockedTestEvaluationError(RuntimeError):
    """Raised when a locked-test evaluation violates its contract."""


@dataclass(frozen=True)
class LockedTestEvaluation:
    predictions: pd.DataFrame
    results: dict[str, Any]


_PARAMETER_KEYS = {
    "l2_regularization",
    "learning_rate",
    "loss",
    "max_bins",
    "max_features",
    "max_iter",
    "max_leaf_nodes",
    "min_samples_leaf",
    "random_state",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LockedTestEvaluationError(message)


def _parameter_equal(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)

    try:
        return bool(np.isclose(float(left), float(right)))
    except (TypeError, ValueError):
        return left == right


def _validate_selected_parameters(
    model_contract: dict[str, Any],
    selected_manifest: dict[str, Any],
) -> dict[str, float | int | str]:
    raw_parameters = selected_manifest.get("selected_parameters", {})
    parameters = dict(raw_parameters)

    _require(
        set(parameters) == _PARAMETER_KEYS,
        "Frozen parameter collection is incomplete or unexpected",
    )

    selection = model_contract["candidate_selection"]
    expected_static = {
        "loss": str(selection["hist_gradient_boosting_loss"]),
        "min_samples_leaf": int(
            selection["hist_gradient_boosting_min_samples_leaf"]
        ),
        "max_bins": int(
            selection["hist_gradient_boosting_max_bins"]
        ),
        "max_features": float(
            selection["hist_gradient_boosting_max_features"]
        ),
        "random_state": int(
            selection["hist_gradient_boosting_random_state"]
        ),
    }

    for name, expected in expected_static.items():
        _require(
            _parameter_equal(parameters[name], expected),
            f"Frozen parameter differs from the model contract: {name}",
        )

    grid_checks = {
        "learning_rate": [
            float(value)
            for value in selection[
                "hist_gradient_boosting_learning_rate_grid"
            ]
        ],
        "max_iter": [
            int(value)
            for value in selection[
                "hist_gradient_boosting_max_iter_grid"
            ]
        ],
        "max_leaf_nodes": [
            int(value)
            for value in selection[
                "hist_gradient_boosting_max_leaf_nodes_grid"
            ]
        ],
        "l2_regularization": [
            float(value)
            for value in selection[
                "hist_gradient_boosting_l2_regularization_grid"
            ]
        ],
    }

    for name, allowed in grid_checks.items():
        _require(
            any(
                _parameter_equal(parameters[name], value)
                for value in allowed
            ),
            f"Frozen parameter is outside its approved grid: {name}",
        )

    _require(
        selection["hist_gradient_boosting_early_stopping"] is False,
        "Internal early stopping must remain disabled",
    )

    return parameters


def _validate_governance(
    frame: pd.DataFrame,
    locked_contract: dict[str, Any],
    selected_manifest: dict[str, Any],
) -> None:
    governance = locked_contract["governance"]
    selected_source = locked_contract["selected_model_source"]
    refit = locked_contract["refit_boundary"]
    boundary = locked_contract["locked_test_boundary"]

    _require(
        governance["decision_gate"] == "4E",
        "Unexpected locked-test governance gate",
    )
    _require(
        governance["status"] == "locked_pending_execution",
        "Locked-test contract is not pending execution",
    )
    _require(
        int(governance["maximum_evaluation_count"]) == 1,
        "Locked-test contract must authorize exactly one evaluation",
    )
    _require(
        governance["repeated_test_evaluation_prohibited"] is True,
        "Repeated evaluation must be prohibited",
    )
    _require(
        selected_source["parameters_immutable"] is True,
        "Selected parameters are not immutable",
    )

    manifest_checks = {
        "governance_gate": selected_source["required_governance_gate"],
        "status": selected_source["required_status"],
        "selection_basis": selected_source["required_selection_basis"],
        "selected_model": selected_source["required_selected_model"],
        "locked_test_evaluated": selected_source[
            "required_locked_test_evaluated"
        ],
    }

    for name, expected in manifest_checks.items():
        _require(
            selected_manifest.get(name) == expected,
            f"Selected-model manifest mismatch: {name}",
        )

    fitted_evidence = selected_manifest["fitted_model_evidence"]
    _require(
        fitted_evidence["internal_early_stopping"] is False,
        "Frozen model used internal early stopping",
    )

    test_controls = selected_manifest.get("test_access_controls", {})
    for name in (
        "locked_test_features_used_for_prediction",
        "locked_test_targets_used_for_scoring",
        "locked_test_metrics_computed",
        "locked_test_predictions_written",
    ):
        _require(
            test_controls.get(name) is False,
            f"Gate 4D test-access control is not clean: {name}",
        )

    train_start = int(refit["training_origin_start"])
    train_stop = int(refit["training_origin_stop_exclusive"])
    purge_start = int(refit["purge_dependency_start"])
    purge_stop = int(refit["purge_dependency_stop_exclusive"])
    locked_start = int(boundary["locked_test_start"])
    locked_stop = int(boundary["locked_test_stop_exclusive"])
    evaluation_start = int(boundary["evaluation_origin_start"])
    evaluation_stop = int(boundary["evaluation_origin_stop_exclusive"])
    horizon = int(boundary["maximum_target_horizon_steps"])

    _require(train_start == 0, "Refit must begin at the first origin")
    _require(train_stop == purge_start, "Refit and purge do not align")
    _require(purge_stop == locked_start, "Purge and test do not align")
    _require(
        evaluation_start == locked_start,
        "Evaluation must begin at the locked-test boundary",
    )
    _require(
        evaluation_stop + horizon == locked_stop,
        "Evaluation stop does not preserve the target horizon",
    )
    _require(
        len(frame) == locked_stop,
        "Input frame does not match the locked row boundary",
    )
    _require(
        evaluation_stop - evaluation_start
        == int(boundary["evaluation_origin_count"]),
        "Evaluation-origin count is inconsistent",
    )
    _require(
        int(boundary["maximum_evaluation_origin"])
        == evaluation_stop - 1,
        "Maximum evaluation origin is inconsistent",
    )
    _require(
        int(boundary["maximum_target_dependency"])
        == evaluation_stop - 1 + horizon,
        "Maximum evaluation dependency is inconsistent",
    )

    frozen_boundary = selected_manifest["training_boundary"]
    expected_frozen = {
        "training_start": train_start,
        "training_stop_exclusive": train_stop,
        "test_purge_start": purge_start,
        "test_purge_stop": purge_stop,
        "locked_test_start": locked_start,
        "locked_test_stop": locked_stop,
        "maximum_training_origin": train_stop - 1,
        "maximum_target_dependency": train_stop - 1 + horizon,
    }

    for name, expected in expected_frozen.items():
        _require(
            int(frozen_boundary[name]) == expected,
            f"Frozen training boundary mismatch: {name}",
        )

    temporal = locked_contract["metrics"]["temporal_stability"]
    blocks = temporal["blocks"]
    _require(
        len(blocks) == int(temporal["block_count"]),
        "Temporal block count is inconsistent",
    )
    _require(
        int(blocks[0]["origin_start"]) == evaluation_start,
        "First temporal block begins at the wrong origin",
    )
    _require(
        int(blocks[-1]["origin_stop_exclusive"]) == evaluation_stop,
        "Last temporal block ends at the wrong origin",
    )

    expected_block_count = int(
        temporal["equal_origin_count_per_block"]
    )

    for index, block in enumerate(blocks):
        start = int(block["origin_start"])
        stop = int(block["origin_stop_exclusive"])
        _require(
            stop - start == expected_block_count,
            "Temporal block has an unexpected origin count",
        )

        if index > 0:
            previous_stop = int(
                blocks[index - 1]["origin_stop_exclusive"]
            )
            _require(
                start == previous_stop,
                "Temporal blocks are not contiguous",
            )


def _feature_frame(
    frame: pd.DataFrame,
    model_contract: dict[str, Any],
) -> pd.DataFrame:
    policy = model_contract["feature_policy"]
    numeric = [str(value) for value in policy["numeric_features"]]
    categorical = [
        str(value) for value in policy["categorical_features"]
    ]
    requested = numeric + categorical
    missing = sorted(set(requested).difference(frame.columns))

    _require(not missing, f"Locked-test features are missing: {missing}")
    features = frame.loc[:, requested].copy()

    for column in numeric:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    for column in categorical:
        values = features[column].astype("object")
        features[column] = values.where(pd.notna(values), np.nan)

    return features


def _build_frozen_pipeline(
    model_contract: dict[str, Any],
    parameters: dict[str, float | int | str],
) -> Pipeline:
    estimator = build_hist_gradient_boosting_estimator(model_contract)
    estimator.set_params(**parameters, early_stopping=False)

    return Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor(model_contract)),
            ("model", estimator),
        ]
    )


def _mae(actual: pd.Series, prediction: pd.Series) -> float:
    error = actual.astype(float).sub(prediction.astype(float)).abs()
    _require(not error.empty, "Metric input is empty")
    _require(
        bool(np.isfinite(error.to_numpy(dtype=float)).all()),
        "Metric input contains non-finite values",
    )
    return float(error.mean())


def _relative_improvement(candidate: float, reference: float) -> float:
    _require(
        np.isfinite(reference) and reference > 0.0,
        "Reference MAE must be finite and positive",
    )
    return float((reference - candidate) / reference)


def evaluate_locked_test_frame(
    frame: pd.DataFrame,
    model_contract: dict[str, Any],
    target_contract: dict[str, Any],
    locked_contract: dict[str, Any],
    selected_manifest: dict[str, Any],
) -> LockedTestEvaluation:
    """Evaluate one explicitly supplied frame under the frozen contract."""
    _require(
        isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.step == 1,
        "Input frame must use a zero-based contiguous RangeIndex",
    )

    required = {
        "source_row_number",
        "effective_timestamp",
        "usage_kwh",
    }
    missing = sorted(required.difference(frame.columns))
    _require(not missing, f"Required evaluation fields are missing: {missing}")

    timestamps = pd.to_datetime(
        frame["effective_timestamp"],
        errors="raise",
    )
    _require(
        timestamps.is_monotonic_increasing,
        "Evaluation timestamps are not chronological",
    )
    _require(
        not timestamps.duplicated().any(),
        "Evaluation timestamps contain duplicates",
    )

    _validate_governance(frame, locked_contract, selected_manifest)
    parameters = _validate_selected_parameters(
        model_contract,
        selected_manifest,
    )

    refit = locked_contract["refit_boundary"]
    boundary = locked_contract["locked_test_boundary"]
    targets_contract = locked_contract["targets"]
    temporal = locked_contract["metrics"]["temporal_stability"]

    train_start = int(refit["training_origin_start"])
    train_stop = int(refit["training_origin_stop_exclusive"])
    evaluation_start = int(boundary["evaluation_origin_start"])
    evaluation_stop = int(boundary["evaluation_origin_stop_exclusive"])

    training_mask = pd.Series(False, index=frame.index)
    training_mask.iloc[train_start:train_stop] = True
    target_artifacts = build_supervised_targets(
        frame,
        training_mask,
        contract=target_contract,
    )
    targets = target_artifacts.frame

    regression_name = str(targets_contract["regression_target"])
    peak_name = str(targets_contract["peak_state_target"])
    training_index = frame.index[train_start:train_stop]
    evaluation_index = frame.index[evaluation_start:evaluation_stop]

    training_target = targets.loc[
        training_index,
        regression_name,
    ].astype(float)
    evaluation_target = targets.loc[
        evaluation_index,
        regression_name,
    ].astype(float)
    peak_state = targets.loc[evaluation_index, peak_name].astype("Int64")

    _require(
        training_target.notna().all(),
        "Refit labels are incomplete",
    )
    _require(
        evaluation_target.notna().all(),
        "Locked-test regression labels are incomplete",
    )
    _require(
        peak_state.notna().all(),
        "Locked-test peak labels are incomplete",
    )

    features = _feature_frame(frame, model_contract)
    pipeline = _build_frozen_pipeline(model_contract, parameters)
    pipeline.fit(
        features.loc[training_index],
        training_target,
    )

    fitted_model = pipeline.named_steps["model"]
    _require(
        bool(fitted_model.do_early_stopping_) is False,
        "Frozen evaluator activated internal early stopping",
    )
    _require(
        int(fitted_model.n_iter_) == int(parameters["max_iter"]),
        "Frozen evaluator did not fit every approved iteration",
    )

    candidate_prediction = pd.Series(
        pipeline.predict(features.loc[evaluation_index]),
        index=evaluation_index,
        dtype=float,
    )
    persistence_prediction = pd.to_numeric(
        frame.loc[evaluation_index, "usage_kwh"],
        errors="raise",
    ).astype(float)

    candidate_error = evaluation_target.sub(
        candidate_prediction
    ).abs()
    persistence_error = evaluation_target.sub(
        persistence_prediction
    ).abs()

    positions = evaluation_index.to_numpy(dtype=int)
    block_ids = np.full(len(evaluation_index), -1, dtype=int)

    for block in temporal["blocks"]:
        block_id = int(block["block_id"])
        start = int(block["origin_start"])
        stop = int(block["origin_stop_exclusive"])
        mask = (positions >= start) & (positions < stop)
        block_ids[mask] = block_id

    _require(
        bool((block_ids >= 0).all()),
        "At least one evaluation origin has no temporal block",
    )

    predictions = pd.DataFrame(
        {
            "source_row_number": frame.loc[
                evaluation_index,
                "source_row_number",
            ].to_numpy(),
            "effective_timestamp": timestamps.loc[
                evaluation_index
            ].to_numpy(),
            "actual_usage_kwh_t_plus_1": evaluation_target.to_numpy(),
            "candidate_prediction": candidate_prediction.to_numpy(),
            "persistence_prediction": persistence_prediction.to_numpy(),
            "candidate_absolute_error": candidate_error.to_numpy(),
            "persistence_absolute_error": persistence_error.to_numpy(),
            "peak_within_next_60_minutes": peak_state.astype(
                int
            ).to_numpy(),
            "temporal_block_id": block_ids,
        }
    )

    required_columns = [
        str(value)
        for value in locked_contract["outputs"][
            "predictions_required_columns"
        ]
    ]
    _require(
        list(predictions.columns) == required_columns,
        "Prediction schema differs from the locked output contract",
    )

    candidate_mae = _mae(
        evaluation_target,
        candidate_prediction,
    )
    persistence_mae = _mae(
        evaluation_target,
        persistence_prediction,
    )
    peak_mask = peak_state.eq(1)
    _require(
        bool(peak_mask.any()),
        "Locked-test evaluation contains no peak-state rows",
    )

    peak_candidate_mae = _mae(
        evaluation_target.loc[peak_mask],
        candidate_prediction.loc[peak_mask],
    )
    peak_persistence_mae = _mae(
        evaluation_target.loc[peak_mask],
        persistence_prediction.loc[peak_mask],
    )

    temporal_rows: list[dict[str, float | int]] = []

    for block in temporal["blocks"]:
        block_id = int(block["block_id"])
        block_mask = predictions["temporal_block_id"].eq(block_id)
        block_candidate = float(
            predictions.loc[
                block_mask,
                "candidate_absolute_error",
            ].mean()
        )
        block_persistence = float(
            predictions.loc[
                block_mask,
                "persistence_absolute_error",
            ].mean()
        )

        temporal_rows.append(
            {
                "block_id": block_id,
                "origin_start": int(block["origin_start"]),
                "origin_stop_exclusive": int(
                    block["origin_stop_exclusive"]
                ),
                "origin_count": int(block_mask.sum()),
                "candidate_mae": block_candidate,
                "persistence_mae": block_persistence,
                "relative_mae_improvement": _relative_improvement(
                    block_candidate,
                    block_persistence,
                ),
            }
        )

    results: dict[str, Any] = {
        "governance_gate": "4E",
        "status": "evaluation_complete_in_memory",
        "selected_model": selected_manifest["selected_model"],
        "locked_test_evaluated": True,
        "prediction_row_count": int(len(predictions)),
        "peak_state_row_count": int(peak_mask.sum()),
        "peak_threshold_kwh": float(
            target_artifacts.peak_threshold_kwh
        ),
        "boundaries": {
            "training_origin_start": train_start,
            "training_origin_stop_exclusive": train_stop,
            "evaluation_origin_start": evaluation_start,
            "evaluation_origin_stop_exclusive": evaluation_stop,
            "maximum_target_dependency": int(
                boundary["maximum_target_dependency"]
            ),
        },
        "selected_parameters": parameters,
        "metrics": {
            "aggregate": {
                "candidate_mae": candidate_mae,
                "persistence_mae": persistence_mae,
                "relative_mae_improvement": _relative_improvement(
                    candidate_mae,
                    persistence_mae,
                ),
            },
            "peak_state": {
                "row_count": int(peak_mask.sum()),
                "candidate_mae": peak_candidate_mae,
                "persistence_mae": peak_persistence_mae,
                "relative_mae_improvement": _relative_improvement(
                    peak_candidate_mae,
                    peak_persistence_mae,
                ),
            },
            "temporal_blocks": {
                "block_count": int(len(temporal_rows)),
                "blocks": temporal_rows,
            },
        },
    }

    return LockedTestEvaluation(
        predictions=predictions,
        results=results,
    )
