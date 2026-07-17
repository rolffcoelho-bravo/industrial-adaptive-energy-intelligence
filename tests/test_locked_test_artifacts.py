from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from iaei.modeling.locked_test import LockedTestEvaluation
from iaei.modeling.locked_test_artifacts import (
    LockedTestArtifactError,
    finalize_locked_test_outputs,
    reserve_locked_test_outputs,
)


def _contract() -> dict:
    return {
        "outputs": {
            "predictions_path": (
                "outputs/modeling/locked_test_predictions.csv"
            ),
            "results_path": (
                "outputs/modeling/locked_test_results.json"
            ),
            "write_once": True,
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


def _evaluation() -> LockedTestEvaluation:
    predictions = pd.DataFrame(
        {
            "source_row_number": [13, 14],
            "effective_timestamp": pd.to_datetime(
                ["2018-01-01 03:15:00", "2018-01-01 03:30:00"]
            ),
            "actual_usage_kwh_t_plus_1": [18.0, 19.0],
            "candidate_prediction": [17.5, 18.5],
            "persistence_prediction": [17.0, 18.0],
            "candidate_absolute_error": [0.5, 0.5],
            "persistence_absolute_error": [1.0, 1.0],
            "peak_within_next_60_minutes": [1, 1],
            "temporal_block_id": [1, 2],
        }
    )
    results = {
        "governance_gate": "4E",
        "status": "evaluation_complete_in_memory",
        "locked_test_evaluated": True,
        "prediction_row_count": 2,
        "metrics": {
            "aggregate": {
                "candidate_mae": 0.5,
                "persistence_mae": 1.0,
            },
        },
    }

    return LockedTestEvaluation(
        predictions=predictions,
        results=results,
    )


def test_reservation_creates_both_write_once_paths(
    tmp_path: Path,
) -> None:
    predictions_path, results_path = reserve_locked_test_outputs(
        tmp_path,
        _contract(),
    )

    assert predictions_path.is_file()
    assert results_path.is_file()
    assert predictions_path.read_text(encoding="utf-8") == (
        "reserved_pending_evaluation\n"
    )
    assert results_path.read_text(encoding="utf-8") == (
        "reserved_pending_evaluation\n"
    )


def test_second_reservation_is_rejected(tmp_path: Path) -> None:
    reserve_locked_test_outputs(tmp_path, _contract())

    with pytest.raises(
        LockedTestArtifactError,
        match="already exists",
    ):
        reserve_locked_test_outputs(tmp_path, _contract())


def test_finalization_writes_exact_csv_and_json(
    tmp_path: Path,
) -> None:
    predictions_path, results_path = reserve_locked_test_outputs(
        tmp_path,
        _contract(),
    )

    finalize_locked_test_outputs(
        tmp_path,
        _contract(),
        _evaluation(),
    )

    predictions = pd.read_csv(predictions_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert len(predictions) == 2
    assert list(predictions.columns) == (
        _contract()["outputs"]["predictions_required_columns"]
    )
    assert results["governance_gate"] == "4E"
    assert results["locked_test_evaluated"] is True
    assert not predictions_path.with_name(
        f"{predictions_path.name}.finalizing.tmp"
    ).exists()
    assert not results_path.with_name(
        f"{results_path.name}.finalizing.tmp"
    ).exists()


def test_invalid_schema_preserves_reservations(
    tmp_path: Path,
) -> None:
    predictions_path, results_path = reserve_locked_test_outputs(
        tmp_path,
        _contract(),
    )
    evaluation = _evaluation()
    malformed = LockedTestEvaluation(
        predictions=evaluation.predictions.drop(
            columns=["temporal_block_id"]
        ),
        results=evaluation.results,
    )

    with pytest.raises(
        LockedTestArtifactError,
        match="prediction schema",
    ):
        finalize_locked_test_outputs(
            tmp_path,
            _contract(),
            malformed,
        )

    assert predictions_path.read_text(encoding="utf-8") == (
        "reserved_pending_evaluation\n"
    )
    assert results_path.read_text(encoding="utf-8") == (
        "reserved_pending_evaluation\n"
    )


def test_existing_nonreservation_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    predictions_path, results_path = reserve_locked_test_outputs(
        tmp_path,
        _contract(),
    )
    predictions_path.write_text(
        "existing evidence\n",
        encoding="utf-8",
    )

    with pytest.raises(
        LockedTestArtifactError,
        match="reservation is not pending",
    ):
        finalize_locked_test_outputs(
            tmp_path,
            _contract(),
            _evaluation(),
        )

    assert results_path.read_text(encoding="utf-8") == (
        "reserved_pending_evaluation\n"
    )


def test_output_path_escape_is_rejected(tmp_path: Path) -> None:
    contract = deepcopy(_contract())
    contract["outputs"]["results_path"] = "../outside.json"

    with pytest.raises(
        LockedTestArtifactError,
        match="escapes the repository root",
    ):
        reserve_locked_test_outputs(tmp_path, contract)
