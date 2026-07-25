from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from iaei.contracts import (
    validate_reporting_closure_manifest,
    validate_v1_release_manifest,
)
from iaei.paths import ROOT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_5e_release_manifest_is_valid_and_frozen() -> None:
    release = validate_v1_release_manifest()
    reporting = validate_reporting_closure_manifest()

    assert release["gate"] == "5E"
    assert release["status"] == "closed"
    assert release["release_version"] == "1.0.0"
    assert release["tag"] == "v1.0.0"
    assert release["next_gate"] == "6A"

    technical_brief = release["canonical_assets"]["technical_brief"]
    canonical = reporting["canonical_pdf"]

    assert technical_brief["sha256"] == canonical["sha256"]
    assert technical_brief["size_bytes"] == canonical["size_bytes"]
    assert technical_brief["source_artifact_id"] == 8621498159

    evidence = release["governed_evidence"]
    report_payload = evidence["report_payload"]
    latex_source = evidence["latex_source"]

    assert _sha256(ROOT / report_payload["path"]) == report_payload["sha256"]
    assert _sha256(ROOT / latex_source["path"]) == latex_source["sha256"]

    assert evidence["figures"] == reporting["governed_inputs"]["figures"]
    assert evidence["tables"] == reporting["governed_inputs"]["tables"]

    for relative_path, expected_hash in evidence["figures"].items():
        assert _sha256(ROOT / relative_path) == expected_hash

    for relative_path, expected_hash in evidence["tables"].items():
        assert _sha256(ROOT / relative_path) == expected_hash

    locked = _load_json(
        ROOT / "outputs" / "modeling" / "locked_test_closure_manifest.json"
    )
    release_locked = evidence["locked_test"]

    assert release_locked["execution_commit"] == locked["execution"][
        "execution_commit"
    ]
    assert release_locked["prediction_sha256"] == locked["artifacts"][
        "predictions"
    ]["sha256"]
    assert release_locked["results_sha256"] == locked["artifacts"][
        "results"
    ]["sha256"]
    assert release_locked["evaluation_count"] == 1
    assert release_locked["authorization_consumed"] is True
    assert release_locked["second_evaluation_permitted"] is False

    historical = {
        value["path"]: value["normalized_text_sha256"]
        for value in locked["historical_source_evidence"].values()
    }
    assert evidence["historical_contracts"] == historical

    controls = release["release_controls"]
    assert controls["main_only"] is True
    assert controls["immutable_tag"] is True
    assert controls["asset_overwrite_permitted"] is False
    assert controls["model_reestimated"] is False
    assert controls["locked_test_reused"] is False
    assert controls["retired_evaluator_invoked"] is False
    assert controls["locked_predictions_parsed"] is False
    assert controls["metrics_recalculated"] is False
    assert controls["approved_figures_changed"] is False
    assert controls["report_payload_changed"] is False
    assert controls["reporting_source_changed_after_approval"] is False
    assert controls["canonical_pdf_replaced"] is False
    assert controls["v1_mutable"] is False


def test_package_and_public_links_use_v1_release_identity() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert pyproject["project"]["version"] == "1.0.0"
    assert "/releases/tag/v1.0.0" in readme
    assert "/releases/download/v1.0.0/" in readme
    assert "outputs/v1_release_manifest.json" in readme


def test_release_notes_and_closure_document_are_present() -> None:
    release = validate_v1_release_manifest()

    for asset_name in (
        "release_manifest",
        "reporting_closure_manifest",
        "latex_source",
        "release_notes",
    ):
        path = ROOT / release["canonical_assets"][asset_name]["path"]
        assert path.exists()
        assert path.stat().st_size > 0

    closure = ROOT / "docs" / "GATE_5E_V1_RELEASE_CLOSURE.md"
    assert closure.exists()
    assert "Gate 6A" in closure.read_text(encoding="utf-8")
