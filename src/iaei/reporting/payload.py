from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from iaei.contracts import ContractError, validate_report_payload
from iaei.paths import ROOT


REPORT_PAYLOAD_PATH = ROOT / "outputs" / "report_payload.json"

RESULTS_PATH = (
    ROOT / "outputs" / "modeling" / "locked_test_results.json"
)
CLOSURE_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "locked_test_closure_manifest.json"
)
SELECTED_MODEL_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "selected_model_manifest.json"
)
REPORTING_MANIFEST_PATH = (
    ROOT / "outputs" / "reporting_evidence_manifest.json"
)
CONFIRMATORY_TABLE_PATH = (
    ROOT / "outputs" / "tables" / "confirmatory_metrics.csv"
)
DATA_TABLE_PATH = (
    ROOT / "outputs" / "tables" / "data_quality_summary.csv"
)
LADDER_TABLE_PATH = (
    ROOT / "outputs" / "tables" / "model_ladder_summary.csv"
)
BLOCK_TABLE_PATH = (
    ROOT / "outputs" / "tables" / "temporal_block_results.csv"
)
LOCKED_PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "modeling"
    / "locked_test_predictions.csv"
)

HASH_ONLY_INPUTS = (LOCKED_PREDICTIONS_PATH,)

PARSED_INPUTS = (
    RESULTS_PATH,
    CLOSURE_PATH,
    SELECTED_MODEL_PATH,
    REPORTING_MANIFEST_PATH,
    CONFIRMATORY_TABLE_PATH,
    DATA_TABLE_PATH,
    LADDER_TABLE_PATH,
    BLOCK_TABLE_PATH,
)

RAW_HASHES = {
    "outputs/modeling/locked_test_predictions.csv": (
        "ec5d1bad7ea3af6b7f2b4c7605be8e3a1efdf067cf452c91e22e8ac37a959b4c"
    ),
    "outputs/modeling/locked_test_results.json": (
        "7b312fe66dd8443b94646055fbfa619aa1bcb6210891cd968cc09dd6bd381a9b"
    ),
    "outputs/reporting_evidence_manifest.json": (
        "93b6ea5afed692d3038244fb6262b51bacdc286e7a0e64a8300b64a0b4e722bd"
    ),
    "outputs/tables/confirmatory_metrics.csv": (
        "bd6816dcd518fc3a18eed95049adbf5a4d9b6d0f19252740bfc2a362a5ff3073"
    ),
    "outputs/tables/data_quality_summary.csv": (
        "43ae0a766d3afa3e49a1653bcccd9eaaa61492dff1241eb728e860b168b6566a"
    ),
    "outputs/tables/model_ladder_summary.csv": (
        "a61cbab367ad83a4199919c04d5fc9285f9abffba1da9ab762134d03866d3c58"
    ),
    "outputs/tables/temporal_block_results.csv": (
        "58ab345ca592b279fc8a41e65642d663a868b334cee28a6fa8cfd0a517446d27"
    ),
    "outputs/tables/evidence_lineage.csv": (
        "ae3c7229370e556ed7e8bb22deb3d195d4e07506e4018edc9269f0c62db34507"
    ),
    "outputs/charts/confirmatory_forecasting_verdict.png": (
        "5bd81c6d72fa7f42de1a9fac341dd986d6d505293dcd39492d04bef9c0fdcb11"
    ),
    "outputs/charts/governed_data_architecture.png": (
        "e121f57a84fb70266b62bae545bde5f9cfabc3330371eb23d1136f4dbe18dc04"
    ),
    "outputs/charts/model_ladder_chronological_validation.png": (
        "b3836b2bf61d4a8dbaf9f16d2774ab8fd8509532ad619762a6761ddb415d453e"
    ),
    "outputs/charts/locked_test_temporal_stability.png": (
        "3770c9e100b607ca73af5b470d256d4beae088f95503afd44703d24663c92d46"
    ),
    "outputs/charts/evidence_governance_model_boundaries.png": (
        "65446260ef8167a5d24493569fc32501abf5d4b10f40f72c35d7fc1f5d9c1c04"
    ),
}

NORMALIZED_TEXT_HASHES = {
    "outputs/modeling/selected_model_manifest.json": (
        "6ebcaff485fb673be6ec58dd5c79f92de7c00474a61c36e4ec7a8102ab86bc8b"
    ),
    "outputs/modeling/locked_test_closure_manifest.json": (
        "cd13e05b628105ad66ab50d3d3ae841e5dd1f150600ea0d67e3863499cd58133"
    ),
}

VISUAL_PATHS = {
    "page_1": (
        "outputs/charts/confirmatory_forecasting_verdict.png"
    ),
    "page_2": (
        "outputs/charts/governed_data_architecture.png"
    ),
    "page_3": (
        "outputs/charts/model_ladder_chronological_validation.png"
    ),
    "page_4": (
        "outputs/charts/locked_test_temporal_stability.png"
    ),
    "page_5": (
        "outputs/charts/evidence_governance_model_boundaries.png"
    ),
}


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n") + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if not isinstance(value, dict):
        raise ContractError(f"Expected an object in {path}")

    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_close(
    observed: float,
    expected: float,
    message: str,
) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError(message)


def _verify_inputs() -> None:
    for relative_path, expected_hash in RAW_HASHES.items():
        path = ROOT / relative_path
        _require(path.exists(), f"Missing governed input: {path}")
        _require(
            _raw_sha256(path) == expected_hash,
            f"Raw-byte governed identity changed: {relative_path}",
        )

    for relative_path, expected_hash in (
        NORMALIZED_TEXT_HASHES.items()
    ):
        path = ROOT / relative_path
        _require(path.exists(), f"Missing governed input: {path}")
        _require(
            _normalized_text_sha256(path) == expected_hash,
            f"Normalized governed identity changed: {relative_path}",
        )


def _single_row(
    rows: list[dict[str, str]],
    label: str,
) -> dict[str, str]:
    _require(len(rows) == 1, f"{label} must contain one row")
    return rows[0]


def _metric_by_scope(
    rows: list[dict[str, str]],
    scope: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["metric_scope"] == scope
    ]
    return _single_row(matches, f"{scope} metric")


def _validate_sources(
    reporting: dict[str, Any],
    results: dict[str, Any],
    closure: dict[str, Any],
    selected: dict[str, Any],
    confirmatory_rows: list[dict[str, str]],
    data_row: dict[str, str],
    ladder_rows: list[dict[str, str]],
    block_rows: list[dict[str, str]],
) -> None:
    _require(
        reporting["governance_gate"] == "5B",
        "Unexpected reporting evidence gate",
    )
    _require(
        results["governance_gate"] == "4E",
        "Unexpected locked-test result gate",
    )
    _require(
        closure["governance_gate"] == "4F",
        "Unexpected locked-test closure gate",
    )
    _require(
        selected["governance_gate"] == "4D",
        "Unexpected selected-model gate",
    )

    controls = reporting["controls"]
    required_false = (
        "evaluator_imported",
        "evaluator_invoked",
        "locked_metrics_recalculated_from_predictions",
        "locked_predictions_parsed",
        "model_fitting_performed",
        "reestimation_performed",
        "second_locked_test_evaluation_performed",
    )

    for field in required_false:
        _require(
            controls[field] is False,
            f"Reporting control must remain false: {field}",
        )

    selected_model = results["selected_model"]
    _require(
        selected["selected_model"] == selected_model,
        "Selected-model evidence is inconsistent",
    )
    _require(
        closure["model"]["selected_model"] == selected_model,
        "Closure model evidence is inconsistent",
    )

    execution = results["execution_evidence"]
    checks = closure["closure_checks"]

    _require(
        execution["evaluation_count"] == 1,
        "Confirmatory evaluation count must equal one",
    )
    _require(
        checks["single_evaluation_consumed"] is True,
        "Single evaluation must remain consumed",
    )
    _require(
        checks["second_evaluation_allowed"] is False,
        "Second evaluation must remain prohibited",
    )
    _require(
        checks["reestimation_performed"] is False,
        "Reestimation must remain false",
    )

    predictions_hash = RAW_HASHES[
        "outputs/modeling/locked_test_predictions.csv"
    ]
    results_hash = RAW_HASHES[
        "outputs/modeling/locked_test_results.json"
    ]

    _require(
        results["data_identity"]["predictions_csv_sha256"]
        == predictions_hash,
        "Terminal prediction hash is inconsistent",
    )
    _require(
        closure["artifacts"]["predictions"]["sha256"]
        == predictions_hash,
        "Closure prediction hash is inconsistent",
    )
    _require(
        closure["artifacts"]["results"]["sha256"]
        == results_hash,
        "Closure result hash is inconsistent",
    )

    aggregate_row = _metric_by_scope(
        confirmatory_rows,
        "aggregate",
    )
    peak_row = _metric_by_scope(
        confirmatory_rows,
        "peak_state",
    )
    aggregate = results["metrics"]["aggregate"]
    peak = results["metrics"]["peak_state"]

    metric_checks = (
        (
            float(aggregate_row["candidate_mae"]),
            float(aggregate["candidate_mae"]),
            "Aggregate candidate MAE is inconsistent",
        ),
        (
            float(aggregate_row["reference_mae"]),
            float(aggregate["persistence_mae"]),
            "Aggregate persistence MAE is inconsistent",
        ),
        (
            float(aggregate_row["relative_mae_improvement"]),
            float(aggregate["relative_mae_improvement"]),
            "Aggregate improvement is inconsistent",
        ),
        (
            float(peak_row["candidate_mae"]),
            float(peak["candidate_mae"]),
            "Peak candidate MAE is inconsistent",
        ),
        (
            float(peak_row["reference_mae"]),
            float(peak["persistence_mae"]),
            "Peak persistence MAE is inconsistent",
        ),
        (
            float(peak_row["relative_mae_improvement"]),
            float(peak["relative_mae_improvement"]),
            "Peak improvement is inconsistent",
        ),
    )

    for observed, expected, message in metric_checks:
        _require_close(observed, expected, message)

    _require(
        int(data_row["raw_row_count"]) == 35040,
        "Unexpected raw row count",
    )
    _require(
        int(data_row["silver_row_count"]) == 35040,
        "Unexpected Silver row count",
    )
    _require(
        int(data_row["silver_column_count"]) == 57,
        "Unexpected Silver column count",
    )
    _require(
        int(data_row["dq_any_count"]) == 0,
        "Unexpected aggregate data-quality count",
    )

    expected_models = [
        "persistence",
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
    ]
    _require(
        [row["model"] for row in ladder_rows] == expected_models,
        "Model ladder is inconsistent",
    )

    _require(
        len(block_rows) == 4,
        "Temporal block evidence must contain four rows",
    )
    _require(
        [int(row["block_id"]) for row in block_rows]
        == [1, 2, 3, 4],
        "Temporal block identifiers are inconsistent",
    )
    _require(
        all(
            float(row["relative_mae_improvement"]) > 0.0
            for row in block_rows
        ),
        "Every temporal block must remain positive",
    )


def _build_payload() -> dict[str, Any]:
    reporting = _read_json(REPORTING_MANIFEST_PATH)
    results = _read_json(RESULTS_PATH)
    closure = _read_json(CLOSURE_PATH)
    selected = _read_json(SELECTED_MODEL_PATH)

    confirmatory_rows = _read_csv(CONFIRMATORY_TABLE_PATH)
    data_row = _single_row(
        _read_csv(DATA_TABLE_PATH),
        "Data-quality summary",
    )
    ladder_rows = _read_csv(LADDER_TABLE_PATH)
    block_rows = _read_csv(BLOCK_TABLE_PATH)

    _validate_sources(
        reporting,
        results,
        closure,
        selected,
        confirmatory_rows,
        data_row,
        ladder_rows,
        block_rows,
    )

    aggregate = results["metrics"]["aggregate"]
    peak = results["metrics"]["peak_state"]
    blocks = results["metrics"]["temporal_blocks"]
    checks = closure["closure_checks"]

    selected_rows = [
        row
        for row in ladder_rows
        if row["model"] == "hist_gradient_boosting"
    ]
    selected_row = _single_row(
        selected_rows,
        "Selected model ladder row",
    )

    model_version = "{}_gate_{}_contract_{}".format(
        selected["selected_model"],
        selected["governance_gate"],
        selected["contract_version"],
    )

    predictions_hash = RAW_HASHES[
        "outputs/modeling/locked_test_predictions.csv"
    ]
    results_hash = RAW_HASHES[
        "outputs/modeling/locked_test_results.json"
    ]

    return {
        "metadata": {
            "dataset_hash": data_row["raw_csv_sha256"],
            "dataset_citation": data_row["citation"],
            "run_timestamp_utc": (
                results["execution_evidence"]["executed_at_utc"]
            ),
            "git_commit": reporting["evidence_base_commit"],
            "model_version": model_version,
            "promotion_decision": "promote_challenger",
            "all_results_final": True,
            "evidence_gate": "4F",
            "locked_test_execution_commit": (
                results["execution_evidence"]["execution_commit"]
            ),
            "predictions_sha256": predictions_hash,
            "results_sha256": results_hash,
        },
        "visuals": VISUAL_PATHS,
        "executive": {
            "narrative": (
                "The frozen histogram gradient boosting model was "
                "evaluated once on the untouched confirmatory period. "
                "It reduced one-step-ahead MAE relative to persistence "
                "in aggregate, during the frozen peak state, and in "
                "every prespecified temporal block."
            ),
            "selected_model": results["selected_model"],
            "candidate_mae": aggregate["candidate_mae"],
            "persistence_mae": aggregate["persistence_mae"],
            "relative_mae_improvement": (
                aggregate["relative_mae_improvement"]
            ),
            "peak_relative_mae_improvement": (
                peak["relative_mae_improvement"]
            ),
            "prediction_row_count": results["prediction_row_count"],
            "all_temporal_blocks_positive": (
                checks["all_temporal_blocks_positive"]
            ),
            "decision_statement": (
                "The frozen histogram gradient boosting model reduced "
                "one-step-ahead energy-demand MAE by 27.6% relative "
                "to persistence on the untouched confirmatory period, "
                "retained a 22.6% advantage during peak-demand states, "
                "and outperformed the reference in every prespecified "
                "temporal block."
            ),
        },
        "data": {
            "narrative": (
                "The governed UCI-851 source contains 35,040 licensed "
                "15-minute observations. Source order, effective "
                "timestamps, analytical transformations, quality "
                "controls, and evidence identities remain explicit."
            ),
            "dataset_id": data_row["dataset_id"],
            "raw_row_count": int(data_row["raw_row_count"]),
            "interval_minutes": int(
                data_row["expected_frequency_minutes"]
            ),
            "silver_column_count": int(
                data_row["silver_column_count"]
            ),
            "dq_any_count": int(data_row["dq_any_count"]),
            "sample_start": data_row["effective_sample_start"],
            "sample_end": data_row["effective_sample_end"],
            "architecture_statement": (
                "Raw evidence, governed manifests, target and leakage "
                "controls, Silver transformations, chronological "
                "validation, locked-test artifacts, reporting tables, "
                "and approved figures remain separated by explicit "
                "contracts and governed identities."
            ),
        },
        "validation": {
            "narrative": (
                "Model selection used four expanding-window "
                "chronological folds and 7,004 validation origins. "
                "Persistence remained the formal reference, while "
                "the selected model was promoted using validation "
                "evidence without locked-test involvement."
            ),
            "validation_fold_count": int(
                selected_row["validation_fold_count"]
            ),
            "validation_origin_count": int(
                selected_row["validation_origin_count"]
            ),
            "candidate_ladder": (
                "persistence -> ridge -> elastic_net -> "
                "hist_gradient_boosting"
            ),
            "selected_model": selected["selected_model"],
            "selection_basis": selected["selection_basis"],
            "internal_early_stopping": (
                selected["fitted_model_evidence"][
                    "internal_early_stopping"
                ]
            ),
            "locked_test_used_for_selection": False,
        },
        "locked_test": {
            "narrative": (
                "The single authorized confirmatory execution covered "
                "7,004 origins. Peak-state evidence contained 761 rows "
                "above the frozen 80.96 kWh threshold, and relative "
                "MAE improvement remained positive across four equal "
                "prespecified temporal blocks."
            ),
            "evaluation_origin_count": results["prediction_row_count"],
            "peak_state_row_count": results["peak_state_row_count"],
            "peak_threshold_kwh": results["peak_threshold_kwh"],
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
            "temporal_block_count": blocks["block_count"],
            "all_temporal_blocks_positive": (
                checks["all_temporal_blocks_positive"]
            ),
            "evaluation_count": (
                results["execution_evidence"]["evaluation_count"]
            ),
        },
        "governance": {
            "narrative": (
                "Confirmatory evidence is closed and append-only. "
                "The single-use authorization was consumed, no "
                "reestimation occurred, and the retired evaluator "
                "is not imported or invoked by the reporting layer."
            ),
            "execution_commit": (
                results["execution_evidence"]["execution_commit"]
            ),
            "predictions_sha256": predictions_hash,
            "results_sha256": results_hash,
            "single_evaluation_consumed": (
                checks["single_evaluation_consumed"]
            ),
            "second_evaluation_allowed": (
                checks["second_evaluation_allowed"]
            ),
            "reestimation_performed": (
                checks["reestimation_performed"]
            ),
            "evaluator_retired": (
                checks["evaluator_must_not_run_again"]
            ),
            "model_boundary_statement": (
                "The evidence supports one-step-ahead forecasting on "
                "the governed UCI sample under frozen feature, target, "
                "chronology, and evaluation contracts. It does not "
                "establish live deployment performance, causal effects, "
                "structural drift findings, operational recommendations, "
                "or quantified economic outcomes."
            ),
        },
    }


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def build_report_payload(
    output_path: Path = REPORT_PAYLOAD_PATH,
) -> Path:
    _verify_inputs()
    payload = _build_payload()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        temporary_path.write_text(
            _serialize(payload),
            encoding="utf-8",
            newline="\n",
        )
        validate_report_payload(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path
