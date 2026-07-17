from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iaei.data.fingerprint import normalized_text_sha256


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = (
    ROOT / "outputs" / "modeling" / "locked_test_predictions.csv"
)
RESULTS_PATH = (
    ROOT / "outputs" / "modeling" / "locked_test_results.json"
)
CLOSURE_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "locked_test_closure_manifest.json"
)

EXPECTED_PREDICTIONS_SHA256 = (
    "ec5d1bad7ea3af6b7f2b4c7605be8e3a1efdf067cf452c91e22e8ac37a959b4c"
)
EXPECTED_RESULTS_SHA256 = (
    "7b312fe66dd8443b94646055fbfa619aa1bcb6210891cd968cc09dd6bd381a9b"
)
EXPECTED_EXECUTION_COMMIT = (
    "9d47bd598b08e9cf7a6b371448bc493d76c8b5e9"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_4f_closure_identity() -> None:
    closure = _load_json(CLOSURE_PATH)

    assert closure["contract_version"] == "2.0.0"
    assert closure["governance_gate"] == "4F"
    assert closure["status"] == "confirmatory_evaluation_closed"
    assert closure["outcome"] == "success"
    assert closure["execution"]["execution_commit"] == (
        EXPECTED_EXECUTION_COMMIT
    )
    assert closure["execution"]["evaluation_count"] == 1


def test_terminal_artifact_hashes_are_exact() -> None:
    closure = _load_json(CLOSURE_PATH)

    assert _sha256(PREDICTIONS_PATH) == EXPECTED_PREDICTIONS_SHA256
    assert _sha256(RESULTS_PATH) == EXPECTED_RESULTS_SHA256
    assert closure["artifacts"]["predictions"]["sha256"] == (
        EXPECTED_PREDICTIONS_SHA256
    )
    assert closure["artifacts"]["results"]["sha256"] == (
        EXPECTED_RESULTS_SHA256
    )


def test_prediction_schema_boundaries_and_blocks_are_exact() -> None:
    closure = _load_json(CLOSURE_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    prediction_record = closure["artifacts"]["predictions"]

    assert list(predictions.columns) == prediction_record[
        "required_columns"
    ]
    assert len(predictions) == prediction_record["row_count"] == 7_004
    assert int(predictions["source_row_number"].min()) == 28_032
    assert int(predictions["source_row_number"].max()) == 35_035
    assert predictions["source_row_number"].is_unique

    block_counts = (
        predictions["temporal_block_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert block_counts == {1: 1_751, 2: 1_751, 3: 1_751, 4: 1_751}


def test_confirmatory_metrics_match_terminal_results() -> None:
    closure = _load_json(CLOSURE_PATH)
    results = _load_json(RESULTS_PATH)
    metrics = closure["confirmatory_metrics"]
    blocks = metrics["temporal_blocks"]["blocks"]

    assert metrics["aggregate"] == results["metrics"]["aggregate"]
    assert metrics["peak_state"] == results["metrics"]["peak_state"]
    assert metrics["temporal_blocks"] == (
        results["metrics"]["temporal_blocks"]
    )
    assert metrics["aggregate"]["relative_mae_improvement"] > 0
    assert metrics["peak_state"]["relative_mae_improvement"] > 0
    assert all(
        block["relative_mae_improvement"] > 0
        for block in blocks
    )
    assert metrics[
        "minimum_block_relative_mae_improvement"
    ] == min(
        block["relative_mae_improvement"]
        for block in blocks
    )
    assert metrics[
        "maximum_block_relative_mae_improvement"
    ] == max(
        block["relative_mae_improvement"]
        for block in blocks
    )


def test_gate_4f_closure_prohibits_reuse_and_reestimation() -> None:
    closure = _load_json(CLOSURE_PATH)
    controls = closure["closure_checks"]

    assert controls["single_evaluation_consumed"] is True
    assert controls["second_evaluation_allowed"] is False
    assert controls["reestimation_performed"] is False
    assert controls["model_redevelopment_performed"] is False
    assert controls["evaluator_must_not_run_again"] is True
    assert controls["aggregate_superiority_confirmed"] is True
    assert controls["peak_state_superiority_confirmed"] is True
    assert controls["all_temporal_blocks_positive"] is True


def test_historical_execution_evidence_is_unchanged() -> None:
    closure = _load_json(CLOSURE_PATH)
    results = _load_json(RESULTS_PATH)

    assert closure["historical_source_evidence"] == (
        results["source_evidence"]
    )

    for record in results["source_evidence"].values():
        path = ROOT / record["path"]
        assert record["hash_contract"] == "utf8_lf_sha256_v1"
        assert normalized_text_sha256(path) == (
            record["normalized_text_sha256"]
        )


def test_evaluator_is_not_invoked_by_ci() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "evaluate_locked_test_once.py" not in workflow


def test_prediction_evidence_has_binary_git_rule() -> None:
    attributes = (ROOT / ".gitattributes").read_text(
        encoding="utf-8"
    )

    assert (
        "outputs/modeling/locked_test_predictions.csv binary"
        in attributes.splitlines()
    )
