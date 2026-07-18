from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from iaei.data.fingerprint import normalized_text_sha256


EVIDENCE_BASE_COMMIT = (
    "6dd856c0f430bdd85e995fd5d4b176b86e3144cf"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "ec5d1bad7ea3af6b7f2b4c7605be8e3a1efdf067cf452c91e22e8ac37a959b4c"
)
EXPECTED_RESULTS_SHA256 = (
    "7b312fe66dd8443b94646055fbfa619aa1bcb6210891cd968cc09dd6bd381a9b"
)

TABLE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "confirmatory_metrics.csv": (
        "metric_scope",
        "candidate_model",
        "reference_model",
        "candidate_mae",
        "reference_mae",
        "relative_mae_improvement",
        "origin_count",
        "peak_threshold_kwh",
        "source_artifact",
        "governance_gate",
    ),
    "data_quality_summary.csv": (
        "dataset_id",
        "dataset_name",
        "source_status",
        "license",
        "citation",
        "raw_csv_sha256",
        "raw_row_count",
        "source_column_count",
        "expected_frequency_minutes",
        "effective_sample_start",
        "effective_sample_end",
        "silver_row_count",
        "silver_column_count",
        "dq_any_count",
        "source_order_preserved",
        "supervised_targets_present",
    ),
    "model_ladder_summary.csv": (
        "model",
        "role",
        "mean_validation_mae",
        "mean_peak_state_mae",
        "worst_fold_mae",
        "relative_mae_improvement_vs_persistence",
        "promotion_decision",
        "governance_gate",
        "status",
        "validation_fold_count",
        "validation_origin_count",
        "locked_test_used_for_selection",
    ),
    "temporal_block_results.csv": (
        "block_id",
        "origin_start",
        "origin_stop_exclusive",
        "origin_count",
        "candidate_mae",
        "persistence_mae",
        "relative_mae_improvement",
        "source_artifact",
        "governance_gate",
    ),
    "evidence_lineage.csv": (
        "sequence",
        "governance_gate",
        "artifact_role",
        "artifact_path",
        "status",
        "decision",
        "hash_contract",
        "sha256",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _render_csv(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            field: _cell(row.get(field))
            for field in fieldnames
        })
    return buffer.getvalue()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _text_source_record(
    root: Path,
    relative: str,
) -> dict[str, str]:
    path = root / relative
    return {
        "path": relative,
        "hash_contract": "utf8_lf_sha256_v1",
        "sha256": normalized_text_sha256(path),
    }


def _binary_source_record(
    root: Path,
    relative: str,
) -> dict[str, str]:
    path = root / relative
    return {
        "path": relative,
        "hash_contract": "raw_bytes_sha256_v1",
        "sha256": _sha256(path),
    }


def _verify_terminal_evidence(
    results_path: Path,
    predictions_path: Path,
    closure: Mapping[str, Any],
) -> None:
    results_hash = _sha256(results_path)
    predictions_hash = _sha256(predictions_path)

    if results_hash != EXPECTED_RESULTS_SHA256:
        raise ValueError("Locked result evidence identity changed")
    if predictions_hash != EXPECTED_PREDICTIONS_SHA256:
        raise ValueError("Locked prediction evidence identity changed")
    if closure["artifacts"]["results"]["sha256"] != results_hash:
        raise ValueError("Closure result hash does not reconcile")
    if (
        closure["artifacts"]["predictions"]["sha256"]
        != predictions_hash
    ):
        raise ValueError("Closure prediction hash does not reconcile")
    if closure["execution"]["evaluation_count"] != 1:
        raise ValueError("Confirmatory evaluation count is not one")
    if closure["closure_checks"]["second_evaluation_allowed"]:
        raise ValueError("A second locked evaluation is prohibited")
    if not closure["closure_checks"]["evaluator_must_not_run_again"]:
        raise ValueError("Evaluator retirement control is missing")


def _candidate_row(
    manifest: Mapping[str, Any],
    *,
    role: str,
    fold_count: int,
    origin_count: int,
) -> dict[str, Any]:
    metrics = manifest["candidate_metrics"]
    return {
        "model": manifest["candidate"],
        "role": role,
        "mean_validation_mae": metrics["mean_mae"],
        "mean_peak_state_mae": metrics["mean_peak_mae"],
        "worst_fold_mae": metrics["worst_fold_mae"],
        "relative_mae_improvement_vs_persistence": (
            manifest["relative_evidence"]["mae_improvement"]
        ),
        "promotion_decision": manifest["promotion_decision"],
        "governance_gate": manifest["governance_gate"],
        "status": manifest["status"],
        "validation_fold_count": fold_count,
        "validation_origin_count": origin_count,
        "locked_test_used_for_selection": False,
    }


def build_reporting_evidence(
    root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    target_root = (output_root or root / "outputs").resolve()
    tables_root = target_root / "tables"

    raw_manifest_path = (
        root / "data" / "manifests" / "uci_steel_energy_manifest.json"
    )
    quality_path = (
        root / "data" / "processed" / "steel_energy_quality_report.json"
    )
    modeling = root / "outputs" / "modeling"
    benchmark_path = modeling / "benchmark_manifest.json"
    ridge_path = modeling / "ridge_candidate_manifest.json"
    elastic_path = modeling / "elastic_net_candidate_manifest.json"
    hist_path = (
        modeling / "hist_gradient_boosting_candidate_manifest.json"
    )
    selected_path = modeling / "selected_model_manifest.json"
    predictions_path = modeling / "locked_test_predictions.csv"
    results_path = modeling / "locked_test_results.json"
    closure_path = modeling / "locked_test_closure_manifest.json"

    raw = _load_json(raw_manifest_path)
    quality = _load_json(quality_path)
    benchmark = _load_json(benchmark_path)
    ridge = _load_json(ridge_path)
    elastic = _load_json(elastic_path)
    hist = _load_json(hist_path)
    selected = _load_json(selected_path)
    results = _load_json(results_path)
    closure = _load_json(closure_path)

    _verify_terminal_evidence(
        results_path,
        predictions_path,
        closure,
    )

    if selected["selected_model"] != "hist_gradient_boosting":
        raise ValueError("Unexpected frozen selected model")
    if selected["selection_basis"] != "validation_only":
        raise ValueError("Model selection was not validation only")
    if selected["test_access_controls"]["locked_test_metrics_computed"]:
        raise ValueError("Gate 4D manifest records locked metrics")
    if results["selected_model"] != selected["selected_model"]:
        raise ValueError("Selected model identity does not reconcile")
    if results["prediction_row_count"] != 7004:
        raise ValueError("Unexpected locked prediction count")
    if not closure["closure_checks"]["all_temporal_blocks_positive"]:
        raise ValueError("Temporal stability closure did not pass")

    aggregate = results["metrics"]["aggregate"]
    peak = results["metrics"]["peak_state"]
    temporal = results["metrics"]["temporal_blocks"]["blocks"]

    confirmatory_rows = [
        {
            "metric_scope": "aggregate",
            "candidate_model": results["selected_model"],
            "reference_model": "persistence",
            "candidate_mae": aggregate["candidate_mae"],
            "reference_mae": aggregate["persistence_mae"],
            "relative_mae_improvement": (
                aggregate["relative_mae_improvement"]
            ),
            "origin_count": results["prediction_row_count"],
            "peak_threshold_kwh": None,
            "source_artifact": (
                "outputs/modeling/locked_test_results.json"
            ),
            "governance_gate": results["governance_gate"],
        },
        {
            "metric_scope": "peak_state",
            "candidate_model": results["selected_model"],
            "reference_model": "persistence",
            "candidate_mae": peak["candidate_mae"],
            "reference_mae": peak["persistence_mae"],
            "relative_mae_improvement": (
                peak["relative_mae_improvement"]
            ),
            "origin_count": peak["row_count"],
            "peak_threshold_kwh": results["peak_threshold_kwh"],
            "source_artifact": (
                "outputs/modeling/locked_test_results.json"
            ),
            "governance_gate": results["governance_gate"],
        },
    ]

    data_rows = [
        {
            "dataset_id": raw["dataset_id"],
            "dataset_name": raw["dataset_name"],
            "source_status": raw["source_status"],
            "license": raw["license"],
            "citation": raw["citation"],
            "raw_csv_sha256": raw["csv_sha256"],
            "raw_row_count": raw["row_count"],
            "source_column_count": quality["source_column_count"],
            "expected_frequency_minutes": (
                quality["expected_frequency_minutes"]
            ),
            "effective_sample_start": (
                quality["effective_sample_start"]
            ),
            "effective_sample_end": quality["effective_sample_end"],
            "silver_row_count": quality["row_count"],
            "silver_column_count": quality["column_count"],
            "dq_any_count": quality["quality_flag_counts"]["dq_any"],
            "source_order_preserved": (
                quality["source_order_preserved"]
            ),
            "supervised_targets_present": (
                quality["supervised_targets_present"]
            ),
        }
    ]

    fold_count = benchmark["validation_fold_count"]
    origin_count = benchmark["validation_origin_count"]
    model_rows = [
        {
            "model": "persistence",
            "role": "formal_reference",
            "mean_validation_mae": (
                benchmark["regression_mean_mae"]["persistence"]
            ),
            "mean_peak_state_mae": (
                benchmark["regression_mean_peak_mae"]["persistence"]
            ),
            "worst_fold_mae": (
                benchmark["regression_worst_fold_mae"]["persistence"]
            ),
            "relative_mae_improvement_vs_persistence": 0.0,
            "promotion_decision": "formal_reference",
            "governance_gate": benchmark["governance_gate"],
            "status": benchmark["status"],
            "validation_fold_count": fold_count,
            "validation_origin_count": origin_count,
            "locked_test_used_for_selection": False,
        },
        _candidate_row(
            ridge,
            role="candidate",
            fold_count=fold_count,
            origin_count=origin_count,
        ),
        _candidate_row(
            elastic,
            role="candidate",
            fold_count=fold_count,
            origin_count=origin_count,
        ),
        _candidate_row(
            hist,
            role="selected_candidate",
            fold_count=fold_count,
            origin_count=origin_count,
        ),
    ]

    temporal_rows = [
        {
            **block,
            "source_artifact": (
                "outputs/modeling/locked_test_results.json"
            ),
            "governance_gate": results["governance_gate"],
        }
        for block in temporal
    ]

    lineage_rows = [
        {
            "sequence": 1,
            "governance_gate": "4B",
            "artifact_role": "formal_benchmark",
            "artifact_path": (
                "outputs/modeling/benchmark_manifest.json"
            ),
            "status": benchmark["status"],
            "decision": "persistence_reference_locked",
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(benchmark_path),
        },
        {
            "sequence": 2,
            "governance_gate": "4C1",
            "artifact_role": "ridge_candidate",
            "artifact_path": (
                "outputs/modeling/ridge_candidate_manifest.json"
            ),
            "status": ridge["status"],
            "decision": ridge["promotion_decision"],
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(ridge_path),
        },
        {
            "sequence": 3,
            "governance_gate": "4C2",
            "artifact_role": "elastic_net_candidate",
            "artifact_path": (
                "outputs/modeling/elastic_net_candidate_manifest.json"
            ),
            "status": elastic["status"],
            "decision": elastic["promotion_decision"],
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(elastic_path),
        },
        {
            "sequence": 4,
            "governance_gate": "4C3",
            "artifact_role": "hist_gradient_boosting_candidate",
            "artifact_path": (
                "outputs/modeling/"
                "hist_gradient_boosting_candidate_manifest.json"
            ),
            "status": hist["status"],
            "decision": hist["promotion_decision"],
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(hist_path),
        },
        {
            "sequence": 5,
            "governance_gate": "4D",
            "artifact_role": "frozen_selected_model",
            "artifact_path": (
                "outputs/modeling/selected_model_manifest.json"
            ),
            "status": selected["status"],
            "decision": "model_frozen",
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(selected_path),
        },
        {
            "sequence": 6,
            "governance_gate": "4E",
            "artifact_role": "locked_test_predictions",
            "artifact_path": (
                "outputs/modeling/locked_test_predictions.csv"
            ),
            "status": "terminal_evidence",
            "decision": "single_evaluation_consumed",
            "hash_contract": "raw_bytes_sha256_v1",
            "sha256": _sha256(predictions_path),
        },
        {
            "sequence": 7,
            "governance_gate": "4E",
            "artifact_role": "locked_test_results",
            "artifact_path": (
                "outputs/modeling/locked_test_results.json"
            ),
            "status": results["status"],
            "decision": "confirmatory_metrics_recorded",
            "hash_contract": "raw_bytes_sha256_v1",
            "sha256": _sha256(results_path),
        },
        {
            "sequence": 8,
            "governance_gate": "4F",
            "artifact_role": "confirmatory_closure",
            "artifact_path": (
                "outputs/modeling/locked_test_closure_manifest.json"
            ),
            "status": closure["status"],
            "decision": closure["outcome"],
            "hash_contract": "utf8_lf_sha256_v1",
            "sha256": normalized_text_sha256(closure_path),
        },
    ]

    table_rows: dict[str, list[dict[str, Any]]] = {
        "confirmatory_metrics.csv": confirmatory_rows,
        "data_quality_summary.csv": data_rows,
        "model_ladder_summary.csv": model_rows,
        "temporal_block_results.csv": temporal_rows,
        "evidence_lineage.csv": lineage_rows,
    }

    generated: dict[str, dict[str, Any]] = {}
    for filename, rows in table_rows.items():
        columns = TABLE_CONTRACTS[filename]
        path = tables_root / filename
        _atomic_text(path, _render_csv(columns, rows))
        generated[filename] = {
            "path": f"outputs/tables/{filename}",
            "row_count": len(rows),
            "columns": list(columns),
            "hash_contract": "raw_bytes_sha256_v1",
            "sha256": _sha256(path),
        }

    source_artifacts = {
        "raw_manifest": _text_source_record(
            root,
            "data/manifests/uci_steel_energy_manifest.json",
        ),
        "silver_quality_report": _text_source_record(
            root,
            "data/processed/steel_energy_quality_report.json",
        ),
        "benchmark_manifest": _text_source_record(
            root,
            "outputs/modeling/benchmark_manifest.json",
        ),
        "chronological_folds": _text_source_record(
            root,
            "outputs/modeling/chronological_folds.json",
        ),
        "ridge_candidate_manifest": _text_source_record(
            root,
            "outputs/modeling/ridge_candidate_manifest.json",
        ),
        "elastic_net_candidate_manifest": _text_source_record(
            root,
            "outputs/modeling/elastic_net_candidate_manifest.json",
        ),
        "hist_gradient_boosting_candidate_manifest": (
            _text_source_record(
                root,
                "outputs/modeling/"
                "hist_gradient_boosting_candidate_manifest.json",
            )
        ),
        "selected_model_manifest": _text_source_record(
            root,
            "outputs/modeling/selected_model_manifest.json",
        ),
        "locked_test_predictions": _binary_source_record(
            root,
            "outputs/modeling/locked_test_predictions.csv",
        ),
        "locked_test_results": _binary_source_record(
            root,
            "outputs/modeling/locked_test_results.json",
        ),
        "locked_test_closure_manifest": _text_source_record(
            root,
            "outputs/modeling/locked_test_closure_manifest.json",
        ),
        "builder_module": _text_source_record(
            root,
            "src/iaei/reporting/evidence.py",
        ),
        "builder_script": _text_source_record(
            root,
            "scripts/build_reporting_evidence.py",
        ),
    }

    manifest: dict[str, Any] = {
        "contract_version": "1.0.0",
        "governance_gate": "5B",
        "status": "reporting_evidence_synthesized",
        "evidence_base_commit": EVIDENCE_BASE_COMMIT,
        "deterministic_serialization": {
            "csv": "utf8_lf_header_order_v1",
            "json": "utf8_lf_sorted_keys_indent_2",
            "group_atomicity_claimed": False,
            "individual_artifact_atomic_replace": True,
        },
        "controls": {
            "locked_metrics_source": (
                "outputs/modeling/locked_test_results.json"
            ),
            "locked_metrics_recalculated_from_predictions": False,
            "locked_predictions_parsed": False,
            "locked_prediction_hash_verified": True,
            "model_fitting_performed": False,
            "reestimation_performed": False,
            "second_locked_test_evaluation_performed": False,
            "evaluator_imported": False,
            "evaluator_invoked": False,
            "report_payload_written": False,
            "charts_written": False,
            "pdf_written": False,
        },
        "confirmatory_summary": {
            "selected_model": results["selected_model"],
            "prediction_row_count": results["prediction_row_count"],
            "peak_state_row_count": results["peak_state_row_count"],
            "peak_threshold_kwh": results["peak_threshold_kwh"],
            "aggregate": aggregate,
            "peak_state": peak,
            "minimum_block_relative_mae_improvement": (
                closure["confirmatory_metrics"][
                    "minimum_block_relative_mae_improvement"
                ]
            ),
            "maximum_block_relative_mae_improvement": (
                closure["confirmatory_metrics"][
                    "maximum_block_relative_mae_improvement"
                ]
            ),
            "all_temporal_blocks_positive": (
                closure["closure_checks"][
                    "all_temporal_blocks_positive"
                ]
            ),
        },
        "source_artifacts": source_artifacts,
        "generated_artifacts": generated,
    }

    manifest_path = target_root / "reporting_evidence_manifest.json"
    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _atomic_text(manifest_path, manifest_content)

    return manifest
