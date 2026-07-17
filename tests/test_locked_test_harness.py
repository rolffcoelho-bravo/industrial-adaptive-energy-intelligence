from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iaei.modeling.locked_test import LockedTestEvaluation
from iaei.modeling.locked_test_harness import (
    HarnessEvidence,
    LockedTestHarnessError,
    enrich_locked_test_evaluation,
    execute_locked_test_once,
    required_authorization_phrase,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_locked_test_once.py"


def _locked_contract() -> dict:
    return {
        "governance": {
            "decision_gate": "4E",
            "status": "locked_pending_execution",
            "authorization": "single_locked_test_evaluation",
            "approved_source_commit": "source-commit",
            "maximum_evaluation_count": 1,
            "repeated_test_evaluation_prohibited": True,
            "model_redevelopment_inside_gate_prohibited": True,
            "result_must_be_reported_regardless_of_outcome": True,
        },
        "execution_controls": {
            "second_evaluation_allowed": False,
            "hyperparameter_changes_allowed": False,
        },
        "reporting": {
            "authorized_execution_script": (
                "scripts/evaluate_locked_test_once.py"
            ),
            "report_adverse_result_without_reestimation": True,
        },
        "metrics": {
            "primary": {"name": "aggregate_mae"},
            "peak_state": {"name": "peak_state_mae"},
            "temporal_stability": {"block_count": 2},
        },
        "outputs": {
            "predictions_path": "outputs/predictions.csv",
            "results_path": "outputs/results.json",
            "write_once": True,
            "predictions_required_columns": [
                "source_row_number",
                "effective_timestamp",
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
        "locked_test_prediction_rows_produced": 0,
        "data_identity": {
            "raw_csv_sha256": "raw-sha",
            "silver_row_count": 4,
            "silver_column_count": 4,
            "quality_flag_dq_any": 0,
        },
        "source_evidence": {"fixture": {}},
    }


def _evidence(tmp_path: Path) -> HarnessEvidence:
    paths = {
        name: tmp_path / f"{name}.txt"
        for name in (
            "locked",
            "model",
            "target",
            "silver",
            "selected",
            "processing",
            "raw",
        )
    }

    for path in paths.values():
        path.write_text("fixture\n", encoding="utf-8")

    return HarnessEvidence(
        locked_contract=_locked_contract(),
        model_contract={},
        target_contract={},
        silver_contract={},
        selected_manifest=_selected_manifest(),
        silver_processing_manifest={},
        raw_manifest={"csv_sha256": "raw-sha"},
        locked_contract_path=paths["locked"],
        model_contract_path=paths["model"],
        target_contract_path=paths["target"],
        silver_contract_path=paths["silver"],
        selected_manifest_path=paths["selected"],
        silver_processing_manifest_path=paths["processing"],
        raw_manifest_path=paths["raw"],
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_number": np.arange(4),
            "effective_timestamp": pd.date_range(
                "2018-01-01",
                periods=4,
                freq="15min",
            ),
            "usage_kwh": [1.0, 2.0, 3.0, 4.0],
            "dq_any": [False, False, False, False],
        }
    )


def _evaluation() -> LockedTestEvaluation:
    predictions = pd.DataFrame(
        {
            "source_row_number": [2, 3],
            "effective_timestamp": pd.date_range(
                "2018-01-01 00:30:00",
                periods=2,
                freq="15min",
            ),
        }
    )
    results = {
        "governance_gate": "4E",
        "locked_test_evaluated": True,
        "prediction_row_count": 2,
        "metrics": {
            "aggregate": {
                "candidate_mae": 1.0,
                "persistence_mae": 2.0,
            },
            "peak_state": {
                "candidate_mae": 1.5,
                "persistence_mae": 2.5,
            },
        },
    }

    return LockedTestEvaluation(predictions, results)


def test_authorization_phrase_is_contract_derived() -> None:
    assert required_authorization_phrase(_locked_contract()) == (
        "4E::single_locked_test_evaluation::1"
    )


def test_invalid_authorization_stops_before_repository_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.load_harness_evidence",
        lambda root: _evidence(tmp_path),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.verify_repository_checkpoint",
        lambda *args: calls.append("repository"),
    )

    with pytest.raises(
        LockedTestHarnessError,
        match="authorization phrase is invalid",
    ):
        execute_locked_test_once(tmp_path, "invalid", "commit")

    assert calls == []


def test_reservation_occurs_before_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    evidence = _evidence(tmp_path)
    authorization = required_authorization_phrase(
        evidence.locked_contract
    )

    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.load_harness_evidence",
        lambda root: evidence,
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.verify_repository_checkpoint",
        lambda *args: events.append("repository"),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.validate_harness_evidence",
        lambda *args: events.append("evidence"),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.reserve_locked_test_outputs",
        lambda *args: (
            events.append("reserve") or tmp_path / "predictions.csv",
            tmp_path / "results.json",
        ),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.reconstruct_governed_silver",
        lambda *args: (
            events.append("reconstruct") or _frame(),
            {
                "silver_contract_version": "fixture",
                "output": {"row_count": 4, "column_count": 4},
                "quality": {
                    "quality_flag_counts": {"dq_any": 0}
                },
            },
        ),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.evaluate_locked_test_frame",
        lambda *args: events.append("evaluate") or _evaluation(),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.enrich_locked_test_evaluation",
        lambda *args: events.append("enrich") or _evaluation(),
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.finalize_locked_test_outputs",
        lambda *args: (
            events.append("finalize") or tmp_path / "predictions.csv",
            tmp_path / "results.json",
        ),
    )

    run = execute_locked_test_once(
        tmp_path,
        authorization,
        "commit",
    )

    assert run.evaluation.results["locked_test_evaluated"] is True
    assert events == [
        "repository",
        "evidence",
        "reserve",
        "reconstruct",
        "evaluate",
        "enrich",
        "finalize",
    ]


def test_enrichment_records_commit_hashes_and_metric_contract(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    script = tmp_path / "scripts" / "evaluate_locked_test_once.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('fixture')\n", encoding="utf-8")

    enriched = enrich_locked_test_evaluation(
        tmp_path,
        evidence,
        _evaluation(),
        _frame(),
        {
            "silver_contract_version": "fixture",
            "output": {"row_count": 4, "column_count": 4},
            "quality": {"quality_flag_counts": {"dq_any": 0}},
        },
        "commit-sha",
        "2026-07-17T00:00:00+00:00",
    )

    results = enriched.results
    execution = results["execution_evidence"]
    identity = results["data_identity"]

    assert execution["execution_commit"] == "commit-sha"
    assert execution["authorization_phrase_recorded"] is False
    assert execution["retry_or_force_option_available"] is False
    assert len(identity["full_silver_logical_sha256"]) == 64
    assert len(identity["predictions_csv_sha256"]) == 64
    assert results["metric_contract"] == _locked_contract()["metrics"]


def test_cli_has_no_force_retry_or_overwrite_option() -> None:
    specification = importlib.util.spec_from_file_location(
        "gate4e_cli",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    parser = module.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--authorization" in options
    assert "--expected-commit" in options
    assert "--force" not in options
    assert "--retry" not in options
    assert "--overwrite" not in options


def test_ci_does_not_invoke_locked_test_script() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    assert "evaluate_locked_test_once.py" not in workflow


def test_real_locked_test_outputs_remain_absent() -> None:
    assert not (
        ROOT / "outputs" / "modeling" / "locked_test_predictions.csv"
    ).exists()
    assert not (
        ROOT / "outputs" / "modeling" / "locked_test_results.json"
    ).exists()

def test_failure_after_reservation_is_finalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    authorization = required_authorization_phrase(
        evidence.locked_contract
    )

    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.load_harness_evidence",
        lambda root: evidence,
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.verify_repository_checkpoint",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.validate_harness_evidence",
        lambda *args: None,
    )

    def fail_reconstruction(*args):
        raise RuntimeError("controlled reconstruction failure")

    monkeypatch.setattr(
        "iaei.modeling.locked_test_harness.reconstruct_governed_silver",
        fail_reconstruction,
    )

    with pytest.raises(
        LockedTestHarnessError,
        match="failed and was recorded",
    ):
        execute_locked_test_once(
            tmp_path,
            authorization,
            "commit-sha",
        )

    predictions_path = tmp_path / "outputs" / "predictions.csv"
    results_path = tmp_path / "outputs" / "results.json"
    predictions = pd.read_csv(predictions_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert predictions.empty
    assert results["status"] == "evaluation_failed"
    assert results["outcome"] == "failure"
    assert results["evaluation_attempted"] is True
    assert results["locked_test_evaluated"] is False
    assert results["prediction_row_count"] == 0
    assert results["failure"]["stage"] == "silver_reconstruction"
    assert results["failure"]["error_type"] == "RuntimeError"
    assert results["failure"]["reestimation_performed"] is False
    assert results["failure"]["second_evaluation_allowed"] is False
    assert results["execution_evidence"]["evaluation_count"] == 1
