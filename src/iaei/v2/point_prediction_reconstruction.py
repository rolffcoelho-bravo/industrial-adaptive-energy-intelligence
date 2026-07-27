from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn

from iaei.modeling.hist_gradient_boosting import _feature_frame
from iaei.targets import build_supervised_targets
from iaei.v2.uncertainty_execution import (
    Gate6E2ExecutionError,
    _fixed_pipeline,
)


def rebuild_frozen_point_predictions(
    silver: pd.DataFrame,
    folds: list[Any],
    model_contract: dict[str, Any],
    target_contract: dict[str, Any],
    point_manifest: dict[str, Any],
    committed_fold_results: pd.DataFrame,
    *,
    strict_metric_verification: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild validation predictions without point-model search or mutation."""

    features = _feature_frame(silver, model_contract)
    timestamps = pd.to_datetime(silver["effective_timestamp"], errors="raise")
    target_name = str(model_contract["objectives"]["regression_target"])
    selected_by_fold = point_manifest["selected_parameters_by_fold"]
    prediction_frames: list[pd.DataFrame] = []
    verification_rows: list[dict[str, Any]] = []

    for fold in folds:
        fold_id = int(fold.fold_id)
        parameters = dict(selected_by_fold[str(fold_id)])
        committed = committed_fold_results.loc[
            committed_fold_results["fold_id"].eq(fold_id)
        ]
        if len(committed) != 1:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} committed HGB result is not unique"
            )
        committed_row = committed.iloc[0]
        parameter_checks = {
            "learning_rate": float(committed_row["selected_learning_rate"]),
            "max_iter": int(committed_row["selected_max_iter"]),
            "max_leaf_nodes": int(committed_row["selected_max_leaf_nodes"]),
            "l2_regularization": float(
                committed_row["selected_l2_regularization"]
            ),
        }
        for name, expected in parameter_checks.items():
            observed = parameters[name]
            if not np.isclose(float(observed), float(expected), rtol=0.0, atol=0.0):
                raise Gate6E2ExecutionError(
                    f"Fold {fold_id} selected parameter {name} is inconsistent: "
                    f"observed={observed}, committed={expected}"
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
        training_target = targets.loc[training_index, target_name].astype(float)
        valid_training = training_target.notna()
        validation_target = targets.loc[validation_index, target_name].astype(float)
        if validation_target.isna().any():
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} reconstruction has missing validation targets"
            )

        pipeline = _fixed_pipeline(model_contract, parameters)
        pipeline.fit(
            features.loc[training_index].loc[valid_training],
            training_target.loc[valid_training],
        )
        prediction = np.asarray(
            pipeline.predict(features.loc[validation_index]),
            dtype=float,
        )
        actual = validation_target.to_numpy(dtype=float)
        peak_state = actual >= peak_threshold
        mae = float(np.mean(np.abs(actual - prediction)))
        peak_mae = float(np.mean(np.abs(actual[peak_state] - prediction[peak_state])))
        committed_mae = float(committed_row["mae"])
        committed_peak_mae = float(committed_row["peak_mae"])
        mae_matches = bool(
            np.isclose(mae, committed_mae, rtol=1e-10, atol=1e-12)
        )
        peak_mae_matches = bool(
            np.isclose(
                peak_mae,
                committed_peak_mae,
                rtol=1e-10,
                atol=1e-12,
            )
        )
        rows_match = len(prediction) == int(committed_row["validation_rows"])
        peaks_match = int(peak_state.sum()) == int(committed_row["peak_rows"])
        if strict_metric_verification and not mae_matches:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} reconstructed MAE does not match committed evidence: "
                f"observed={mae:.17g}, committed={committed_mae:.17g}, "
                f"absolute_difference={abs(mae - committed_mae):.17g}, "
                f"sklearn={sklearn.__version__}, numpy={np.__version__}, "
                f"pandas={pd.__version__}"
            )
        if strict_metric_verification and not peak_mae_matches:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} reconstructed peak MAE is inconsistent: "
                f"observed={peak_mae:.17g}, committed={committed_peak_mae:.17g}, "
                f"absolute_difference={abs(peak_mae - committed_peak_mae):.17g}, "
                f"sklearn={sklearn.__version__}"
            )
        if strict_metric_verification and not rows_match:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} reconstruction row count is inconsistent: "
                f"observed={len(prediction)}, "
                f"committed={int(committed_row['validation_rows'])}"
            )
        if strict_metric_verification and not peaks_match:
            raise Gate6E2ExecutionError(
                f"Fold {fold_id} reconstruction peak count is inconsistent: "
                f"observed={int(peak_state.sum())}, "
                f"committed={int(committed_row['peak_rows'])}, "
                f"threshold={peak_threshold:.17g}"
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "row_position": validation_index.to_numpy(dtype=int),
                    "prediction_origin": timestamps.loc[
                        validation_index
                    ].to_numpy(),
                    "task": "regression",
                    "candidate": "hist_gradient_boosting",
                    "actual": actual,
                    "prediction": prediction,
                    "peak_threshold_kwh": peak_threshold,
                    "is_peak_state": peak_state,
                    "selected_learning_rate": float(parameters["learning_rate"]),
                    "selected_max_iter": int(parameters["max_iter"]),
                    "selected_max_leaf_nodes": int(parameters["max_leaf_nodes"]),
                    "selected_l2_regularization": float(
                        parameters["l2_regularization"]
                    ),
                }
            )
        )
        verification_rows.append(
            {
                "fold_id": fold_id,
                "validation_rows": int(len(prediction)),
                "committed_validation_rows": int(committed_row["validation_rows"]),
                "peak_rows": int(peak_state.sum()),
                "committed_peak_rows": int(committed_row["peak_rows"]),
                "reconstructed_mae": mae,
                "committed_mae": committed_mae,
                "absolute_mae_difference": abs(mae - committed_mae),
                "reconstructed_peak_mae": peak_mae,
                "committed_peak_mae": committed_peak_mae,
                "absolute_peak_mae_difference": abs(
                    peak_mae - committed_peak_mae
                ),
                "mae_matches": mae_matches,
                "peak_mae_matches": peak_mae_matches,
                "rows_match": rows_match,
                "peaks_match": peaks_match,
                "metrics_match": (
                    mae_matches
                    and peak_mae_matches
                    and rows_match
                    and peaks_match
                ),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold_id", "row_position"],
        kind="stable",
    )
    if len(predictions) != int(point_manifest["prediction_row_count"]):
        raise Gate6E2ExecutionError(
            "Reconstructed point predictions do not cover the frozen origins"
        )
    if int(predictions["row_position"].max()) != int(
        point_manifest["maximum_prediction_origin"]
    ):
        raise Gate6E2ExecutionError(
            "Reconstructed point predictions cross the frozen boundary"
        )
    all_metrics_match = all(
        bool(record["metrics_match"]) for record in verification_rows
    )
    verification = {
        "schema_version": "1.0.0",
        "point_model_id": "v1_frozen_champion",
        "reconstruction_performed": True,
        "point_model_search_performed": False,
        "selected_parameters_mutated": False,
        "committed_fold_metrics_verified": all_metrics_match,
        "runtime": {
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "folds": verification_rows,
    }
    return predictions.reset_index(drop=True), verification
