from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from iaei.contracts import (
    validate_foundation_model_contract,
    validate_gate_6d1_closure_manifest,
    validate_repository_contracts,
)
from iaei.modeling.foundation_governance import (
    APPROVED_FOUNDATION_MODELS,
    RESEARCH_ONLY_MODELS,
    assert_no_gate_6d_execution_artifacts,
    build_gate_6d1_plan,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "iaei" / "modeling" / "foundation_governance.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "gate-6d1-foundation-contract.yml"
PROTOCOL_PATH = ROOT / "docs" / "GATE_6D_FOUNDATION_MODEL_PROTOCOL.md"
SCOPE_PATH = ROOT / "docs" / "GATE_6D1_SCOPE_LOCK.md"


def _validate_source_boundary() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    prohibited_functions = {"fit", "train", "predict", "forecast", "evaluate", "score"}
    observed_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = sorted(observed_functions & prohibited_functions)
    if overlap:
        raise SystemExit(f"Gate 6D1 contains execution functions: {overlap}")

    prohibited_import_roots = {
        "torch",
        "transformers",
        "chronos",
        "timesfm",
        "uni2ts",
        "gluonts",
        "huggingface_hub",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    overlap = sorted(imports & prohibited_import_roots)
    if overlap:
        raise SystemExit(f"Gate 6D1 imported execution frameworks: {overlap}")


def _validate_gate_6c_closure() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6c3_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Gate 6C closure validation failed:\n"
            f"{completed.stdout}{completed.stderr}"
        )


def _validate_cross_artifact_conformance() -> None:
    contract = validate_foundation_model_contract()
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    display_names = {
        "chronos_2_zero_shot": "Chronos-2",
        "timesfm_2_5_zero_shot": "TimesFM 2.5",
        "moirai_2_research_zero_shot": "Moirai 2.0",
    }
    for candidate_id, display_name in display_names.items():
        if candidate_id not in APPROVED_FOUNDATION_MODELS:
            raise SystemExit(f"Unexpected Gate 6D candidate mapping: {candidate_id}")
        if display_name not in protocol or display_name not in scope:
            raise SystemExit(f"Missing cross-artifact candidate: {display_name}")

    for candidate in contract["candidate_models"]:
        for field in ("model_id", "model_revision", "weight_sha256", "weights_license"):
            value = str(candidate[field])
            if value not in scope:
                raise SystemExit(
                    f"Gate 6D1 scope does not record {candidate['candidate_id']} {field}"
                )

    if "scripts/validate_gate_6d1.py" not in workflow:
        raise SystemExit("Gate 6D1 workflow does not invoke its validator")
    prohibited_workflow_fragments = (
        "pip install chronos",
        "pip install timesfm",
        "pip install uni2ts",
        "huggingface-cli",
        "run_gate_6d2",
        "contents: write",
    )
    hits = [fragment for fragment in prohibited_workflow_fragments if fragment in workflow]
    if hits:
        raise SystemExit(f"Gate 6D1 workflow contains execution fragments: {hits}")


def _validate_license_boundary() -> None:
    contract = validate_foundation_model_contract()
    candidates = {item["candidate_id"]: item for item in contract["candidate_models"]}
    for candidate_id in RESEARCH_ONLY_MODELS:
        candidate = candidates[candidate_id]
        if candidate["benchmark_admissible"] is not True:
            raise SystemExit("Research-only Gate 6D candidate is not benchmark admissible")
        if candidate["commercial_use_eligible"] is not False:
            raise SystemExit("Research-only Gate 6D candidate became commercially eligible")
        if candidate["promotion_eligible"] is not False:
            raise SystemExit("Research-only Gate 6D candidate became promotion eligible")


def main() -> None:
    validate_repository_contracts()
    plan = build_gate_6d1_plan()
    closure = validate_gate_6d1_closure_manifest()
    assert_no_gate_6d_execution_artifacts()
    _validate_source_boundary()
    _validate_cross_artifact_conformance()
    _validate_license_boundary()
    _validate_gate_6c_closure()

    if closure["status"] != "closed":
        raise SystemExit("Gate 6D1 closure is not closed")
    if closure["candidate_ids"] != list(APPROVED_FOUNDATION_MODELS):
        raise SystemExit("Gate 6D1 closure candidate identities changed")
    if closure["research_only_candidate_ids"] != list(RESEARCH_ONLY_MODELS):
        raise SystemExit("Gate 6D1 research-only candidate set changed")
    if plan.model_download_permitted or plan.inference_permitted:
        raise SystemExit("Gate 6D1 unexpectedly permits model execution")
    if plan.fine_tuning_permitted or plan.locked_test_access_permitted:
        raise SystemExit("Gate 6D1 weakens a frozen analytical boundary")
    if closure["next_gate_authorized"] is not False:
        raise SystemExit("Gate 6D2 requires a separate explicit authorization")

    print(
        "Gate 6D1 validation: PASS | candidates=3 | context=672 | horizon=1 | "
        "zero_shot=true | download=false | inference=false | locked_test=false | "
        "research_only=1 | next_gate=6D2_authorization_required"
    )


if __name__ == "__main__":
    main()
