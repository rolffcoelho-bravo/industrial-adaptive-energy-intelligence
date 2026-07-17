from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from iaei.contracts import load_yaml
from iaei.data import write_silver_artifacts
from iaei.data.fingerprint import (
    LOGICAL_FRAME_FLOAT_DECIMALS,
    LOGICAL_FRAME_HASH_CONTRACT,
    NORMALIZED_TEXT_HASH_CONTRACT,
    logical_frame_sha256,
    normalized_text_sha256,
)
from iaei.modeling.locked_test import (
    LockedTestEvaluation,
    evaluate_locked_test_frame,
)
from iaei.modeling.locked_test_artifacts import (
    finalize_locked_test_outputs,
    reserve_locked_test_outputs,
)


class LockedTestHarnessError(RuntimeError):
    """Raised when single-use execution governance is violated."""


@dataclass(frozen=True)
class HarnessEvidence:
    locked_contract: dict[str, Any]
    model_contract: dict[str, Any]
    target_contract: dict[str, Any]
    silver_contract: dict[str, Any]
    selected_manifest: dict[str, Any]
    silver_processing_manifest: dict[str, Any]
    raw_manifest: dict[str, Any]
    locked_contract_path: Path
    model_contract_path: Path
    target_contract_path: Path
    silver_contract_path: Path
    selected_manifest_path: Path
    silver_processing_manifest_path: Path
    raw_manifest_path: Path


@dataclass(frozen=True)
class HarnessRun:
    evaluation: LockedTestEvaluation
    predictions_path: Path
    results_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LockedTestHarnessError(message)


def required_authorization_phrase(
    locked_contract: dict[str, Any],
) -> str:
    governance = locked_contract["governance"]

    return "::".join(
        [
            str(governance["decision_gate"]),
            str(governance["authorization"]),
            str(governance["maximum_evaluation_count"]),
        ]
    )


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Required evidence is missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Invalid JSON evidence: {path}")

    return payload


def _resolve_inside_root(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()

    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise LockedTestHarnessError(
            "Evidence path escapes the repository root"
        ) from error

    return resolved_path


def load_harness_evidence(root: Path) -> HarnessEvidence:
    locked_contract_path = root / "configs" / "locked_test_contract.yml"
    model_contract_path = root / "configs" / "model_contract.yml"
    target_contract_path = root / "configs" / "target_contract.yml"
    silver_contract_path = root / "configs" / "silver_contract.yml"

    locked_contract = load_yaml(locked_contract_path)
    model_contract = load_yaml(model_contract_path)
    target_contract = load_yaml(target_contract_path)
    silver_contract = load_yaml(silver_contract_path)

    selected_manifest_path = _resolve_inside_root(
        root,
        str(
            locked_contract["selected_model_source"]["manifest_path"]
        ),
    )
    silver_processing_manifest_path = (
        root
        / "data"
        / "processed"
        / "steel_energy_processing_manifest.json"
    )
    raw_manifest_path = _resolve_inside_root(
        root,
        str(silver_contract["source"]["raw_manifest_path"]),
    )

    return HarnessEvidence(
        locked_contract=locked_contract,
        model_contract=model_contract,
        target_contract=target_contract,
        silver_contract=silver_contract,
        selected_manifest=_load_json(selected_manifest_path),
        silver_processing_manifest=_load_json(
            silver_processing_manifest_path
        ),
        raw_manifest=_load_json(raw_manifest_path),
        locked_contract_path=locked_contract_path,
        model_contract_path=model_contract_path,
        target_contract_path=target_contract_path,
        silver_contract_path=silver_contract_path,
        selected_manifest_path=selected_manifest_path,
        silver_processing_manifest_path=(
            silver_processing_manifest_path
        ),
        raw_manifest_path=raw_manifest_path,
    )


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def verify_repository_checkpoint(
    root: Path,
    expected_commit: str,
    approved_source_commit: str,
    authorized_script: str,
) -> None:
    _require(
        os.environ.get("CI", "").lower() not in {"1", "true", "yes"},
        "Locked-test execution is prohibited inside CI",
    )

    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    remote = _git(root, "rev-parse", "origin/main").stdout.strip()
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout.strip()

    _require(head == expected_commit, "HEAD differs from the authorized commit")
    _require(
        remote == expected_commit,
        "origin/main differs from the authorized commit",
    )
    _require(not status, "Repository must be clean before reservation")

    ancestry = _git(
        root,
        "merge-base",
        "--is-ancestor",
        approved_source_commit,
        expected_commit,
        check=False,
    )
    _require(
        ancestry.returncode == 0,
        "Approved Gate 4D source commit is not an ancestor",
    )

    tracked = _git(
        root,
        "ls-files",
        "--error-unmatch",
        authorized_script,
        check=False,
    )
    _require(
        tracked.returncode == 0,
        "Authorized execution script is not tracked",
    )


def _validate_selected_source_evidence(
    root: Path,
    selected_manifest: dict[str, Any],
) -> None:
    source_evidence = selected_manifest["source_evidence"]
    _require(bool(source_evidence), "Selected source evidence is empty")

    for name, record in source_evidence.items():
        path = _resolve_inside_root(root, str(record["path"]))

        _require(
            record["hash_contract"]
            == NORMALIZED_TEXT_HASH_CONTRACT,
            f"Unexpected source hash contract: {name}",
        )
        _require(
            normalized_text_sha256(path)
            == record["normalized_text_sha256"],
            f"Selected source evidence changed: {name}",
        )


def validate_harness_evidence(
    root: Path,
    evidence: HarnessEvidence,
) -> None:
    locked = evidence.locked_contract
    selected = evidence.selected_manifest
    processing = evidence.silver_processing_manifest
    raw = evidence.raw_manifest
    governance = locked["governance"]
    controls = locked["execution_controls"]
    reporting = locked["reporting"]

    _require(governance["decision_gate"] == "4E", "Wrong gate")
    _require(
        governance["status"] == "locked_pending_execution",
        "Gate 4E is not pending execution",
    )
    _require(
        int(governance["maximum_evaluation_count"]) == 1,
        "Exactly one evaluation must be authorized",
    )
    _require(
        governance["repeated_test_evaluation_prohibited"] is True,
        "Repeated evaluation is not prohibited",
    )
    _require(
        governance[
            "model_redevelopment_inside_gate_prohibited"
        ]
        is True,
        "Model redevelopment is not prohibited inside Gate 4E",
    )
    _require(
        governance[
            "result_must_be_reported_regardless_of_outcome"
        ]
        is True,
        "Gate 4E does not require reporting every outcome",
    )
    _require(
        controls["second_evaluation_allowed"] is False,
        "Second evaluation is unexpectedly allowed",
    )
    _require(
        controls["hyperparameter_changes_allowed"] is False,
        "Hyperparameter changes are unexpectedly allowed",
    )
    _require(
        reporting["authorized_execution_script"]
        == "scripts/evaluate_locked_test_once.py",
        "Unexpected authorized execution script",
    )
    _require(
        reporting["report_adverse_result_without_reestimation"]
        is True,
        "Adverse outcomes are not locked for reporting",
    )

    _require(selected["governance_gate"] == "4D", "Wrong source gate")
    _require(selected["status"] == "frozen", "Model is not frozen")
    _require(
        selected["selection_basis"] == "validation_only",
        "Model was not selected using validation only",
    )
    _require(
        selected["selected_model"] == "hist_gradient_boosting",
        "Unexpected selected model",
    )
    _require(
        selected["locked_test_evaluated"] is False,
        "Selected manifest already records test access",
    )
    _require(
        int(selected["locked_test_prediction_rows_produced"]) == 0,
        "Selected manifest already records test predictions",
    )

    data_identity = selected["data_identity"]
    _require(
        raw["csv_sha256"] == data_identity["raw_csv_sha256"],
        "Raw snapshot identity differs from Gate 4D",
    )
    _require(
        int(processing["output"]["row_count"])
        == int(data_identity["silver_row_count"]),
        "Silver row count differs from Gate 4D",
    )
    _require(
        int(processing["output"]["column_count"])
        == int(data_identity["silver_column_count"]),
        "Silver column count differs from Gate 4D",
    )
    _require(
        int(
            processing["quality"]["quality_flag_counts"]["dq_any"]
        )
        == int(data_identity["quality_flag_dq_any"])
        == 0,
        "Silver quality evidence differs from Gate 4D",
    )

    _validate_selected_source_evidence(root, selected)


def reconstruct_governed_silver(
    root: Path,
    evidence: HarnessEvidence,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with tempfile.TemporaryDirectory(
        prefix="iaei-gate-4e-silver-"
    ) as temporary:
        output_dir = Path(temporary)
        manifest = write_silver_artifacts(
            root,
            output_dir=output_dir,
        )
        frame = pd.read_parquet(
            output_dir / "steel_energy_silver.parquet"
        ).reset_index(drop=True)

    selected = evidence.selected_manifest
    identity = selected["data_identity"]
    refit = evidence.locked_contract["refit_boundary"]
    selection_stop = int(refit["purge_dependency_stop_exclusive"])

    _require(
        isinstance(frame.index, pd.RangeIndex),
        "Reconstructed Silver index is not a RangeIndex",
    )
    _require(
        len(frame) == int(identity["silver_row_count"]),
        "Reconstructed Silver row count is invalid",
    )
    _require(
        len(frame.columns) == int(identity["silver_column_count"]),
        "Reconstructed Silver column count is invalid",
    )
    _require(
        frame["source_row_number"].to_numpy().tolist()
        == list(range(len(frame))),
        "Reconstructed Silver source-row sequence is invalid",
    )
    _require(
        not frame["dq_any"].astype(bool).any(),
        "Reconstructed Silver contains quality failures",
    )

    selection_hash = logical_frame_sha256(
        frame.iloc[:selection_stop].copy()
    )
    _require(
        selection_hash
        == identity["selection_input_logical_sha256"],
        "Reconstructed pre-test logical identity differs from Gate 4D",
    )
    _require(
        identity["selection_input_logical_hash_contract"]
        == LOGICAL_FRAME_HASH_CONTRACT,
        "Unexpected selection-input hash contract",
    )
    _require(
        int(identity["selection_input_logical_float_decimals"])
        == LOGICAL_FRAME_FLOAT_DECIMALS,
        "Unexpected selection-input float precision",
    )

    return frame, manifest


def _file_record(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "hash_contract": NORMALIZED_TEXT_HASH_CONTRACT,
        "normalized_text_sha256": normalized_text_sha256(path),
    }


def enrich_locked_test_evaluation(
    root: Path,
    evidence: HarnessEvidence,
    evaluation: LockedTestEvaluation,
    reconstructed_frame: pd.DataFrame,
    reconstruction_manifest: dict[str, Any],
    expected_commit: str,
    executed_at_utc: str,
) -> LockedTestEvaluation:
    predictions = evaluation.predictions.copy(deep=True)
    results = deepcopy(evaluation.results)
    predictions_csv = predictions.to_csv(
        index=False,
        lineterminator="\n",
    )
    authorized_script = _resolve_inside_root(
        root,
        evidence.locked_contract["reporting"][
            "authorized_execution_script"
        ],
    )

    results["execution_evidence"] = {
        "evaluation_count": 1,
        "execution_commit": expected_commit,
        "executed_at_utc": executed_at_utc,
        "repository_clean_before_reservation": True,
        "authorized_execution_script": (
            authorized_script.relative_to(root).as_posix()
        ),
        "authorized_execution_script_sha256": (
            normalized_text_sha256(authorized_script)
        ),
        "authorization_phrase_recorded": False,
        "retry_or_force_option_available": False,
    }
    results["source_evidence"] = {
        "locked_test_contract": _file_record(
            root,
            evidence.locked_contract_path,
        ),
        "model_contract": _file_record(
            root,
            evidence.model_contract_path,
        ),
        "target_contract": _file_record(
            root,
            evidence.target_contract_path,
        ),
        "silver_contract": _file_record(
            root,
            evidence.silver_contract_path,
        ),
        "selected_model_manifest": _file_record(
            root,
            evidence.selected_manifest_path,
        ),
        "raw_manifest": _file_record(
            root,
            evidence.raw_manifest_path,
        ),
        "silver_processing_manifest": _file_record(
            root,
            evidence.silver_processing_manifest_path,
        ),
    }
    results["data_identity"] = {
        "raw_csv_sha256": evidence.raw_manifest["csv_sha256"],
        "full_silver_logical_hash_contract": (
            LOGICAL_FRAME_HASH_CONTRACT
        ),
        "full_silver_logical_float_decimals": (
            LOGICAL_FRAME_FLOAT_DECIMALS
        ),
        "full_silver_logical_sha256": logical_frame_sha256(
            reconstructed_frame
        ),
        "silver_row_count": int(len(reconstructed_frame)),
        "silver_column_count": int(len(reconstructed_frame.columns)),
        "quality_flag_dq_any": int(
            reconstructed_frame["dq_any"].astype(bool).sum()
        ),
        "predictions_logical_sha256": logical_frame_sha256(
            predictions
        ),
        "predictions_csv_sha256": hashlib.sha256(
            predictions_csv.encode("utf-8")
        ).hexdigest(),
    }
    results["metric_contract"] = deepcopy(
        evidence.locked_contract["metrics"]
    )
    results["reconstruction_evidence"] = {
        "silver_contract_version": reconstruction_manifest[
            "silver_contract_version"
        ],
        "row_count": int(
            reconstruction_manifest["output"]["row_count"]
        ),
        "column_count": int(
            reconstruction_manifest["output"]["column_count"]
        ),
        "quality_flag_dq_any": int(
            reconstruction_manifest["quality"][
                "quality_flag_counts"
            ]["dq_any"]
        ),
        "physical_parquet_hash_used_as_model_identity": False,
    }

    return LockedTestEvaluation(
        predictions=predictions,
        results=results,
    )


def build_failed_locked_test_evaluation(
    root: Path,
    evidence: HarnessEvidence,
    expected_commit: str,
    failure_stage: str,
    error: Exception,
    attempted_at_utc: str,
) -> LockedTestEvaluation:
    """Create terminal evidence for a consumed Gate 4E attempt."""
    required_columns = [
        str(value)
        for value in evidence.locked_contract["outputs"][
            "predictions_required_columns"
        ]
    ]
    predictions = pd.DataFrame(columns=required_columns)
    source_paths = {
        "locked_test_contract": evidence.locked_contract_path,
        "model_contract": evidence.model_contract_path,
        "target_contract": evidence.target_contract_path,
        "silver_contract": evidence.silver_contract_path,
        "selected_model_manifest": evidence.selected_manifest_path,
        "raw_manifest": evidence.raw_manifest_path,
        "silver_processing_manifest": (
            evidence.silver_processing_manifest_path
        ),
    }

    results: dict[str, Any] = {
        "governance_gate": "4E",
        "status": "evaluation_failed",
        "outcome": "failure",
        "selected_model": evidence.selected_manifest[
            "selected_model"
        ],
        "evaluation_attempted": True,
        "locked_test_evaluated": False,
        "prediction_row_count": 0,
        "failure": {
            "stage": failure_stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "result_reported_regardless_of_outcome": True,
            "reestimation_performed": False,
            "second_evaluation_allowed": False,
        },
        "execution_evidence": {
            "evaluation_count": 1,
            "execution_commit": expected_commit,
            "attempted_at_utc": attempted_at_utc,
            "authorization_phrase_recorded": False,
            "retry_or_force_option_available": False,
        },
        "source_evidence": {
            name: _file_record(root, path)
            for name, path in source_paths.items()
        },
        "metric_contract": deepcopy(
            evidence.locked_contract["metrics"]
        ),
    }

    return LockedTestEvaluation(
        predictions=predictions,
        results=results,
    )


def execute_locked_test_once(
    root: Path,
    authorization: str,
    expected_commit: str,
) -> HarnessRun:
    evidence = load_harness_evidence(root)
    expected_authorization = required_authorization_phrase(
        evidence.locked_contract
    )

    _require(
        authorization == expected_authorization,
        "Explicit Gate 4E authorization phrase is invalid",
    )

    governance = evidence.locked_contract["governance"]
    authorized_script = evidence.locked_contract["reporting"][
        "authorized_execution_script"
    ]

    verify_repository_checkpoint(
        root,
        expected_commit,
        str(governance["approved_source_commit"]),
        str(authorized_script),
    )
    validate_harness_evidence(root, evidence)

    predictions_path, results_path = reserve_locked_test_outputs(
        root,
        evidence.locked_contract,
    )
    attempted_at_utc = datetime.now(timezone.utc).isoformat()
    failure_stage = "silver_reconstruction"

    try:
        reconstructed_frame, reconstruction_manifest = (
            reconstruct_governed_silver(root, evidence)
        )
        failure_stage = "locked_test_evaluation"
        evaluation = evaluate_locked_test_frame(
            reconstructed_frame,
            evidence.model_contract,
            evidence.target_contract,
            evidence.locked_contract,
            evidence.selected_manifest,
        )
        failure_stage = "result_enrichment"
        executed_at_utc = datetime.now(timezone.utc).isoformat()
        enriched = enrich_locked_test_evaluation(
            root,
            evidence,
            evaluation,
            reconstructed_frame,
            reconstruction_manifest,
            expected_commit,
            executed_at_utc,
        )

        failure_stage = "artifact_finalization"
        finalized_predictions, finalized_results = (
            finalize_locked_test_outputs(
                root,
                evidence.locked_contract,
                enriched,
            )
        )
    except Exception as error:
        failed = build_failed_locked_test_evaluation(
            root,
            evidence,
            expected_commit,
            failure_stage,
            error,
            attempted_at_utc,
        )

        try:
            finalize_locked_test_outputs(
                root,
                evidence.locked_contract,
                failed,
            )
        except Exception as reporting_error:
            raise LockedTestHarnessError(
                "Authorized Gate 4E execution failed and its "
                "terminal failure evidence could not be finalized"
            ) from reporting_error

        raise LockedTestHarnessError(
            "Authorized Gate 4E execution failed and was recorded"
        ) from error

    _require(
        finalized_predictions == predictions_path,
        "Final prediction path differs from its reservation",
    )
    _require(
        finalized_results == results_path,
        "Final result path differs from its reservation",
    )

    return HarnessRun(
        evaluation=enriched,
        predictions_path=finalized_predictions,
        results_path=finalized_results,
    )
