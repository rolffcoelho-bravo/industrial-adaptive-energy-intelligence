from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.modeling.locked_test import (
    LockedTestEvaluationError,
    evaluate_locked_test_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    usage = np.array(
        [
            10.0,
            11.0,
            12.0,
            11.0,
            13.0,
            14.0,
            13.0,
            15.0,
            16.0,
            15.0,
            17.0,
            18.0,
            19.0,
            18.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
        ]
    )

    return pd.DataFrame(
        {
            "source_row_number": np.arange(1, 21),
            "effective_timestamp": pd.date_range(
                "2018-01-01 00:15:00",
                periods=20,
                freq="15min",
            ),
            "usage_kwh": usage,
            "nsm_seconds": np.arange(20) * 900,
            "load_type": [
                "Light_Load" if value % 2 == 0 else "Medium_Load"
                for value in range(20)
            ],
        }
    )


def _model_contract() -> dict:
    return {
        "feature_policy": {
            "numeric_features": ["usage_kwh", "nsm_seconds"],
            "categorical_features": ["load_type"],
        },
        "candidate_selection": {
            "hist_gradient_boosting_loss": "absolute_error",
            "hist_gradient_boosting_min_samples_leaf": 2,
            "hist_gradient_boosting_max_bins": 16,
            "hist_gradient_boosting_max_features": 1.0,
            "hist_gradient_boosting_early_stopping": False,
            "hist_gradient_boosting_random_state": 7,
            "hist_gradient_boosting_learning_rate_grid": [0.1],
            "hist_gradient_boosting_max_iter_grid": [5],
            "hist_gradient_boosting_max_leaf_nodes_grid": [5],
            "hist_gradient_boosting_l2_regularization_grid": [0.0],
        },
    }


def _target_contract() -> dict:
    return {
        "contract_version": "fixture-1",
        "prediction_origin": {
            "timestamp_column": "effective_timestamp",
        },
        "targets": {
            "regression": {
                "name": "usage_kwh_t_plus_1",
                "source_column": "usage_kwh",
                "horizon_steps": 1,
            },
            "peak_risk": {
                "name": "peak_within_next_60_minutes",
                "source_column": "usage_kwh",
                "horizon_steps": 4,
                "threshold": {
                    "quantile": 0.9,
                    "interpolation": "linear",
                },
            },
        },
    }


def _locked_contract() -> dict:
    return {
        "governance": {
            "decision_gate": "4E",
            "status": "locked_pending_execution",
            "maximum_evaluation_count": 1,
            "repeated_test_evaluation_prohibited": True,
        },
        "selected_model_source": {
            "required_governance_gate": "4D",
            "required_status": "frozen",
            "required_selection_basis": "validation_only",
            "required_selected_model": "hist_gradient_boosting",
            "required_locked_test_evaluated": False,
            "parameters_immutable": True,
        },
        "refit_boundary": {
            "training_origin_start": 0,
            "training_origin_stop_exclusive": 8,
            "purge_dependency_start": 8,
            "purge_dependency_stop_exclusive": 12,
        },
        "locked_test_boundary": {
            "locked_test_start": 12,
            "locked_test_stop_exclusive": 20,
            "maximum_target_horizon_steps": 4,
            "evaluation_origin_start": 12,
            "evaluation_origin_stop_exclusive": 16,
            "evaluation_origin_count": 4,
            "maximum_evaluation_origin": 15,
            "maximum_target_dependency": 19,
        },
        "targets": {
            "regression_target": "usage_kwh_t_plus_1",
            "peak_state_target": "peak_within_next_60_minutes",
        },
        "metrics": {
            "temporal_stability": {
                "block_count": 2,
                "equal_origin_count_per_block": 2,
                "blocks": [
                    {
                        "block_id": 1,
                        "origin_start": 12,
                        "origin_stop_exclusive": 14,
                    },
                    {
                        "block_id": 2,
                        "origin_start": 14,
                        "origin_stop_exclusive": 16,
                    },
                ],
            },
        },
        "outputs": {
            "predictions_required_columns": [
                "source_row_number",
                "effective_timestamp",
                "actual_usage_kwh_t_plus_1",
                "candidate_prediction",
                "persistence_prediction",
                "candidate_absolute_error",
                "persistence_absolute_error",
                "peak_within_next_60_minutes",
                "temporal_block_id",
            ],
        },
    }


def _selected_manifest() -> dict:
    return {
        "governance_gate": "4D",
        "status": "frozen",
        "selection_basis": "validation_only",
        "selected_model": "hist_gradient_boosting",
        "locked_test_evaluated": False,
        "selected_parameters": {
            "loss": "absolute_error",
            "learning_rate": 0.1,
            "max_iter": 5,
            "max_leaf_nodes": 5,
            "l2_regularization": 0.0,
            "min_samples_leaf": 2,
            "max_bins": 16,
            "max_features": 1.0,
            "random_state": 7,
        },
        "training_boundary": {
            "training_start": 0,
            "training_stop_exclusive": 8,
            "test_purge_start": 8,
            "test_purge_stop": 12,
            "locked_test_start": 12,
            "locked_test_stop": 20,
            "maximum_training_origin": 7,
            "maximum_target_dependency": 11,
        },
        "fitted_model_evidence": {
            "internal_early_stopping": False,
        },
        "test_access_controls": {
            "locked_test_features_used_for_prediction": False,
            "locked_test_targets_used_for_scoring": False,
            "locked_test_metrics_computed": False,
            "locked_test_predictions_written": False,
        },
    }


def _evaluate():
    return evaluate_locked_test_frame(
        _frame(),
        _model_contract(),
        _target_contract(),
        _locked_contract(),
        _selected_manifest(),
    )


def test_controlled_fixture_produces_exact_prediction_schema() -> None:
    evaluation = _evaluate()
    predictions = evaluation.predictions

    assert len(predictions) == 4
    assert list(predictions.columns) == (
        _locked_contract()["outputs"]["predictions_required_columns"]
    )
    assert predictions["source_row_number"].tolist() == [13, 14, 15, 16]
    assert predictions["temporal_block_id"].tolist() == [1, 1, 2, 2]
    assert predictions["peak_within_next_60_minutes"].eq(1).all()


def test_aggregate_and_peak_metrics_match_prediction_errors() -> None:
    evaluation = _evaluate()
    predictions = evaluation.predictions
    metrics = evaluation.results["metrics"]

    assert metrics["aggregate"]["candidate_mae"] == pytest.approx(
        predictions["candidate_absolute_error"].mean()
    )
    assert metrics["aggregate"]["persistence_mae"] == pytest.approx(
        predictions["persistence_absolute_error"].mean()
    )
    assert metrics["peak_state"]["row_count"] == 4
    assert metrics["peak_state"]["candidate_mae"] == pytest.approx(
        predictions["candidate_absolute_error"].mean()
    )


def test_temporal_metrics_cover_each_origin_once() -> None:
    evaluation = _evaluate()
    temporal = evaluation.results["metrics"]["temporal_blocks"]

    assert temporal["block_count"] == 2
    assert [row["origin_count"] for row in temporal["blocks"]] == [2, 2]
    assert sum(row["origin_count"] for row in temporal["blocks"]) == 4


def test_frozen_parameter_change_is_rejected() -> None:
    manifest = deepcopy(_selected_manifest())
    manifest["selected_parameters"]["max_iter"] = 10

    with pytest.raises(
        LockedTestEvaluationError,
        match="outside its approved grid",
    ):
        evaluate_locked_test_frame(
            _frame(),
            _model_contract(),
            _target_contract(),
            _locked_contract(),
            manifest,
        )


def test_frozen_training_boundary_change_is_rejected() -> None:
    manifest = deepcopy(_selected_manifest())
    manifest["training_boundary"]["training_stop_exclusive"] = 9

    with pytest.raises(
        LockedTestEvaluationError,
        match="Frozen training boundary mismatch",
    ):
        evaluate_locked_test_frame(
            _frame(),
            _model_contract(),
            _target_contract(),
            _locked_contract(),
            manifest,
        )


def test_in_memory_evaluation_preserves_terminal_artifacts() -> None:
    predictions_path = (
        ROOT / "outputs" / "modeling" / "locked_test_predictions.csv"
    )
    results_path = (
        ROOT / "outputs" / "modeling" / "locked_test_results.json"
    )
    before = {
        predictions_path: hashlib.sha256(
            predictions_path.read_bytes()
        ).hexdigest(),
        results_path: hashlib.sha256(
            results_path.read_bytes()
        ).hexdigest(),
    }

    _evaluate()

    after = {
        predictions_path: hashlib.sha256(
            predictions_path.read_bytes()
        ).hexdigest(),
        results_path: hashlib.sha256(
            results_path.read_bytes()
        ).hexdigest(),
    }

    assert after == before
