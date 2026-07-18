from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from iaei.reporting.evidence import (
    EXPECTED_PREDICTIONS_SHA256,
    EXPECTED_RESULTS_SHA256,
    build_reporting_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
MANIFEST = OUTPUTS / "reporting_evidence_manifest.json"

EXPECTED_TABLES = (
    "confirmatory_metrics.csv",
    "data_quality_summary.csv",
    "model_ladder_summary.csv",
    "temporal_block_results.csv",
    "evidence_lineage.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(filename: str) -> list[dict[str, str]]:
    with (TABLES / filename).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def test_reporting_evidence_rebuild_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    temporary_outputs = tmp_path / "outputs"
    build_reporting_evidence(ROOT, temporary_outputs)

    for filename in EXPECTED_TABLES:
        assert (
            temporary_outputs / "tables" / filename
        ).read_bytes() == (TABLES / filename).read_bytes()

    assert (
        temporary_outputs / "reporting_evidence_manifest.json"
    ).read_bytes() == MANIFEST.read_bytes()


def test_confirmatory_metrics_are_copied_exactly() -> None:
    source = json.loads(
        (
            OUTPUTS / "modeling" / "locked_test_results.json"
        ).read_text(encoding="utf-8")
    )
    rows = {
        row["metric_scope"]: row
        for row in _rows("confirmatory_metrics.csv")
    }

    aggregate = source["metrics"]["aggregate"]
    peak = source["metrics"]["peak_state"]

    assert rows["aggregate"]["candidate_mae"] == repr(
        aggregate["candidate_mae"]
    )
    assert rows["aggregate"]["reference_mae"] == repr(
        aggregate["persistence_mae"]
    )
    assert rows["aggregate"]["relative_mae_improvement"] == repr(
        aggregate["relative_mae_improvement"]
    )
    assert rows["aggregate"]["origin_count"] == str(
        source["prediction_row_count"]
    )

    assert rows["peak_state"]["candidate_mae"] == repr(
        peak["candidate_mae"]
    )
    assert rows["peak_state"]["reference_mae"] == repr(
        peak["persistence_mae"]
    )
    assert rows["peak_state"]["relative_mae_improvement"] == repr(
        peak["relative_mae_improvement"]
    )
    assert rows["peak_state"]["origin_count"] == str(
        peak["row_count"]
    )


def test_temporal_blocks_are_copied_exactly() -> None:
    source = json.loads(
        (
            OUTPUTS / "modeling" / "locked_test_results.json"
        ).read_text(encoding="utf-8")
    )
    rows = _rows("temporal_block_results.csv")
    blocks = source["metrics"]["temporal_blocks"]["blocks"]

    assert len(rows) == len(blocks) == 4

    for row, block in zip(rows, blocks, strict=True):
        assert row["block_id"] == str(block["block_id"])
        assert row["origin_start"] == str(block["origin_start"])
        assert row["origin_stop_exclusive"] == str(
            block["origin_stop_exclusive"]
        )
        assert row["origin_count"] == str(block["origin_count"])
        assert row["candidate_mae"] == repr(
            block["candidate_mae"]
        )
        assert row["persistence_mae"] == repr(
            block["persistence_mae"]
        )
        assert row["relative_mae_improvement"] == repr(
            block["relative_mae_improvement"]
        )


def test_model_ladder_reconciles_with_gate_manifests() -> None:
    rows = {
        row["model"]: row
        for row in _rows("model_ladder_summary.csv")
    }
    benchmark = json.loads(
        (
            OUTPUTS / "modeling" / "benchmark_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert set(rows) == {
        "persistence",
        "ridge",
        "elastic_net",
        "hist_gradient_boosting",
    }
    assert rows["persistence"]["mean_validation_mae"] == repr(
        benchmark["regression_mean_mae"]["persistence"]
    )
    assert rows["ridge"]["promotion_decision"] == "rejected"
    assert rows["elastic_net"]["promotion_decision"] == "rejected"
    assert (
        rows["hist_gradient_boosting"]["promotion_decision"]
        == "promoted"
    )
    assert all(
        row["locked_test_used_for_selection"] == "false"
        for row in rows.values()
    )


def test_data_quality_summary_reconciles() -> None:
    rows = _rows("data_quality_summary.csv")
    assert len(rows) == 1
    row = rows[0]

    assert row["dataset_id"] == "uci-851"
    assert row["raw_row_count"] == "35040"
    assert row["silver_row_count"] == "35040"
    assert row["silver_column_count"] == "57"
    assert row["expected_frequency_minutes"] == "15"
    assert row["dq_any_count"] == "0"
    assert row["source_order_preserved"] == "true"
    assert row["supervised_targets_present"] == "false"


def test_manifest_records_hashes_and_controls() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["governance_gate"] == "5B"
    assert manifest["status"] == "reporting_evidence_synthesized"
    assert manifest["controls"][
        "locked_metrics_recalculated_from_predictions"
    ] is False
    assert manifest["controls"]["locked_predictions_parsed"] is False
    assert manifest["controls"]["model_fitting_performed"] is False
    assert manifest["controls"]["reestimation_performed"] is False
    assert manifest["controls"][
        "second_locked_test_evaluation_performed"
    ] is False
    assert manifest["controls"]["evaluator_imported"] is False
    assert manifest["controls"]["evaluator_invoked"] is False
    assert manifest["controls"]["report_payload_written"] is False
    assert manifest["controls"]["charts_written"] is False
    assert manifest["controls"]["pdf_written"] is False

    for filename in EXPECTED_TABLES:
        record = manifest["generated_artifacts"][filename]
        assert record["sha256"] == _sha256(TABLES / filename)


def test_builder_does_not_parse_predictions_or_import_evaluator() -> None:
    path = ROOT / "src" / "iaei" / "reporting" / "evidence.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(
                alias.name for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")

    assert "pandas" not in imported_modules
    assert "read_csv" not in source
    assert "evaluate_locked_test_once" not in source


def test_gate_4e_terminal_hashes_remain_fixed() -> None:
    assert _sha256(
        OUTPUTS / "modeling" / "locked_test_predictions.csv"
    ) == EXPECTED_PREDICTIONS_SHA256
    assert _sha256(
        OUTPUTS / "modeling" / "locked_test_results.json"
    ) == EXPECTED_RESULTS_SHA256


def test_reporting_outputs_exclude_unsupported_claims() -> None:
    text = MANIFEST.read_text(encoding="utf-8").lower()
    for filename in EXPECTED_TABLES:
        text += (TABLES / filename).read_text(
            encoding="utf-8"
        ).lower()

    prohibited = (
        "business savings",
        "optimization recommendation",
        "live production",
        "causal effect",
        "proprietary company data",
    )

    for term in prohibited:
        assert term not in text


def test_versioning_rules_are_exact() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(
        encoding="utf-8"
    )
    attributes = (ROOT / ".gitattributes").read_text(
        encoding="utf-8"
    )

    for filename in EXPECTED_TABLES:
        assert f"!outputs/tables/{filename}" in ignore_text
        assert (
            f"outputs/tables/{filename} text eol=lf"
            in attributes
        )

    assert "!outputs/reporting_evidence_manifest.json" in ignore_text
    assert (
        "outputs/reporting_evidence_manifest.json text eol=lf"
        in attributes
    )
