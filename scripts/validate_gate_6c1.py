from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from iaei.contracts import (
    ContractError,
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
    validate_neural_seed_governance_alignment,
    validate_repository_contracts,
)
from iaei.modeling.neural_governance import (
    APPROVED_ALGORITHMS,
    assert_no_gate_6c_execution_artifacts,
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

    for seed in (20260725, 20260726, 20260727):
        seed_text = str(seed)
        if seed_text not in protocol or seed_text not in scope:
            raise SystemExit(f"Missing proposed Gate 6C seed: {seed}")

    for seed in (20260721, 20260722, 20260723, 20260724, 20260725):
        if str(seed) not in reconciliation:
            raise SystemExit(f"Missing frozen Gate 6A seed in reconciliation: {seed}")

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
    closure = validate_gate_6c1_closure_manifest()
    contract = validate_neural_forecasting_contract()
    validate_repository_contracts()
    assert_no_gate_6c_execution_artifacts()
    _validate_source_boundary()
    _validate_cross_artifact_conformance()
    _validate_gate_6b_closure()

    try:
        validate_neural_seed_governance_alignment(contract)
    except ContractError as error:
        if closure["status"] != "blocked_pending_seed_governance_decision":
            raise SystemExit(str(error)) from error
        if "seed identities conflict" not in str(error):
            raise SystemExit(str(error)) from error
        print(
            "Gate 6C1 governance block: EXPECTED | "
            "reason=seed_contract_conflict | fitting=false | "
            "locked_test=false | next_action=human_seed_decision"
        )
        return

    if closure["status"] == "blocked_pending_seed_governance_decision":
        raise SystemExit("Gate 6C1 remains blocked after seed contracts aligned")

    print(
        "Gate 6C1 validation: PASS | candidates=3 | fitting=false | "
        "locked_test=false | next_gate=6C2"
    )


if __name__ == "__main__":
    main()
