from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from iaei.contracts import (
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
    validate_repository_contracts,
)
from iaei.modeling.neural_governance import (
    APPROVED_ALGORITHMS,
    APPROVED_SEEDS,
    assert_no_gate_6c_execution_artifacts,
    build_gate_6c1_plan,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "iaei" / "modeling" / "neural_governance.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "gate-6c1-neural-contract.yml"
PROTOCOL_PATH = ROOT / "docs" / "GATE_6C_NEURAL_FORECASTING_PROTOCOL.md"
SCOPE_PATH = ROOT / "docs" / "GATE_6C1_SCOPE_LOCK.md"
RECONCILIATION_PATH = ROOT / "docs" / "GATE_6C1_SEED_GOVERNANCE_RECONCILIATION.md"


def _validate_source_boundary() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    prohibited_functions = {"fit", "train", "predict", "evaluate", "score"}
    observed_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = sorted(observed_functions & prohibited_functions)
    if overlap:
        raise SystemExit(f"Gate 6C1 contains execution functions: {overlap}")

    prohibited_import_roots = {
        "torch",
        "tensorflow",
        "jax",
        "lightning",
        "pytorch_lightning",
        "neuralforecast",
        "darts",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    overlap = sorted(imports & prohibited_import_roots)
    if overlap:
        raise SystemExit(f"Gate 6C1 imported execution frameworks: {overlap}")


def _validate_cross_artifact_conformance() -> None:
    contract = validate_neural_forecasting_contract()
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    scope = SCOPE_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    reconciliation = RECONCILIATION_PATH.read_text(encoding="utf-8")

    required_text = {
        "nhits_compact": "N-HiTS",
        "tide_compact": "TiDE",
        "patchtst_compact": "PatchTST",
    }
    for algorithm_id, display_name in required_text.items():
        if algorithm_id not in APPROVED_ALGORITHMS:
            raise SystemExit(f"Unexpected algorithm mapping: {algorithm_id}")
        if display_name not in protocol or display_name not in scope:
            raise SystemExit(f"Missing cross-artifact candidate: {display_name}")

    for seed in APPROVED_SEEDS:
        seed_text = str(seed)
        if seed_text not in protocol or seed_text not in scope:
            raise SystemExit(f"Missing cross-artifact seed: {seed}")
        if seed_text not in reconciliation:
            raise SystemExit(f"Missing reconciled Gate 6A seed: {seed}")

    if "scripts/validate_gate_6c1.py" not in workflow:
        raise SystemExit("Gate 6C1 workflow does not invoke its validator")
    prohibited_workflow_fragments = (
        "run_gate_6c",
        "torch",
        "outputs/v2/gate_6c/",
        "contents: write",
    )
    hits = [fragment for fragment in prohibited_workflow_fragments if fragment in workflow]
    if hits:
        raise SystemExit(f"Gate 6C1 workflow contains execution fragments: {hits}")

    if contract["search"]["unique_configuration_count"] != 3:
        raise SystemExit("Gate 6C configuration count changed")
    if contract["search"]["seed_count"] != 5:
        raise SystemExit("Gate 6C seed count changed")


def _validate_gate_6b_closure() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_gate_6b_closure.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "Gate 6B closure validation failed:\n"
            f"{completed.stdout}{completed.stderr}"
        )


def main() -> None:
    validate_repository_contracts()
    plan = build_gate_6c1_plan()
    closure = validate_gate_6c1_closure_manifest()
    assert_no_gate_6c_execution_artifacts()
    _validate_source_boundary()
    _validate_cross_artifact_conformance()
    _validate_gate_6b_closure()

    if closure["status"] not in {"implementation_complete_pending_ci", "closed"}:
        raise SystemExit("Unexpected Gate 6C1 closure status")
    if closure["candidate_ids"] != list(APPROVED_ALGORITHMS):
        raise SystemExit("Gate 6C1 closure candidate set changed")
    if closure["seeds"] != list(APPROVED_SEEDS):
        raise SystemExit("Gate 6C1 closure seed set changed")
    if plan.fitting_permitted:
        raise SystemExit("Gate 6C1 unexpectedly permits fitting")

    print(
        "Gate 6C1 validation: PASS | candidates=3 | seeds=5 | "
        "fitting=false | locked_test=false | "
        f"status={closure['status']} | next_gate=6C2"
    )


if __name__ == "__main__":
    main()
