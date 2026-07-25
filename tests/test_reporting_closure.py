from __future__ import annotations

import hashlib
from pathlib import Path

from iaei.contracts import validate_reporting_closure_manifest
from iaei.paths import ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_5d4_reporting_closure_is_valid_and_frozen() -> None:
    manifest = validate_reporting_closure_manifest()

    assert manifest["gate"] == "5D4"
    assert manifest["status"] == "closed"
    assert manifest["version"] == "v1"
    assert manifest["next_gate"] == "5E"

    canonical = manifest["canonical_pdf"]

    assert canonical["artifact_path"] == (
        "reports/latex/"
        "industrial_adaptive_energy_intelligence_technical_brief.pdf"
    )
    assert canonical["release_asset_name"] == (
        "industrial_adaptive_energy_intelligence_technical_brief.pdf"
    )
    assert canonical["publication_state"] == (
        "approved_pending_gate_5e_release"
    )
    assert canonical["sha256"] == (
        "35e331e0349e0afca4aa8695a3f4aeafeffa18f83cdd9420876662bc6c782ba3"
    )
    assert canonical["size_bytes"] == 1_218_669
    assert canonical["page_count"] == 5
    assert canonical["page_format"] == "A4"
    assert canonical["author"] == "Industrial Adaptive Energy Intelligence"

    latex_source = manifest["latex_source"]
    latex_path = ROOT / latex_source["path"]

    assert latex_path.exists()
    assert _sha256(latex_path) == latex_source["sha256"]

    approval = manifest["visual_approval"]
    approval_path = ROOT / approval["approval_path"]

    assert approval_path.exists()
    assert approval["dpi"] == 200
    assert len(approval["page_sha256"]) == 5
    assert len(set(approval["page_sha256"])) == 5

    report_payload = manifest["governed_inputs"]["report_payload"]
    payload_path = ROOT / report_payload["path"]

    assert _sha256(payload_path) == report_payload["sha256"]

    for relative_path, expected_hash in manifest["governed_inputs"][
        "figures"
    ].items():
        assert _sha256(ROOT / relative_path) == expected_hash

    for relative_path, expected_hash in manifest["governed_inputs"][
        "tables"
    ].items():
        assert _sha256(ROOT / relative_path) == expected_hash

    controls = manifest["closure_controls"]

    assert controls == {
        "model_reestimated": False,
        "locked_test_reused": False,
        "retired_evaluator_invoked": False,
        "locked_predictions_parsed": False,
        "metrics_recalculated": False,
        "approved_figures_changed": False,
        "report_payload_changed": False,
        "personal_author_metadata_present": False,
        "human_visual_approval_recorded": True,
        "canonical_pdf_frozen": True,
        "canonical_pdf_committed": False,
        "release_asset_pending": True,
        "v1_reporting_mutable": False,
    }
