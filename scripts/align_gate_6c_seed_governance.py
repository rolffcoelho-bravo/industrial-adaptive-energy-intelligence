from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIVE_SEEDS = [20260721, 20260722, 20260723, 20260724, 20260725]
ALGORITHMS = ["nhits_compact", "tide_compact", "patchtst_compact"]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def align_contract() -> None:
    path = ROOT / "configs" / "neural_forecasting_contract.yml"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["contract_version"] = "1.1.0"
    contract["search"]["seeds"] = FIVE_SEEDS
    contract["search"]["seed_count"] = len(FIVE_SEEDS)
    path.write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def align_schemas() -> None:
    contract_path = ROOT / "schemas" / "neural_forecasting_contract.schema.json"
    contract_schema = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_schema["properties"]["contract_version"] = {"const": "1.1.0"}
    search = contract_schema["properties"]["search"]["properties"]
    search["seeds"] = {"const": FIVE_SEEDS}
    search["seed_count"] = {"const": len(FIVE_SEEDS)}
    write_json(contract_path, contract_schema)

    seed_path = ROOT / "schemas" / "neural_seed_evidence.schema.json"
    seed_schema = json.loads(seed_path.read_text(encoding="utf-8"))
    seed_schema["properties"]["seed"] = {"enum": FIVE_SEEDS}
    write_json(seed_path, seed_schema)

    candidate_path = ROOT / "schemas" / "neural_candidate_evidence.schema.json"
    candidate_schema = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_schema["properties"]["seed_count"] = {"const": len(FIVE_SEEDS)}
    write_json(candidate_path, candidate_schema)

    closure_path = ROOT / "schemas" / "gate_6c1_closure_manifest.schema.json"
    closure_schema = json.loads(closure_path.read_text(encoding="utf-8"))
    props = closure_schema["properties"]
    props["status"] = {"enum": ["implementation_complete_pending_ci", "closed"]}
    props["seeds"] = {"const": FIVE_SEEDS}
    props["next_gate"] = {"const": "6C2"}
    props["blocked_gates"] = {"const": ["6C3", "6D"]}
    write_json(closure_path, closure_schema)


def align_closure_manifest() -> None:
    path = ROOT / "outputs" / "v2" / "gate_6c1_closure_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "implementation_complete_pending_ci"
    payload["seeds"] = FIVE_SEEDS
    payload["next_gate"] = "6C2"
    payload["blocked_gates"] = ["6C3", "6D"]
    write_json(path, payload)


def align_python_source() -> None:
    path = ROOT / "src" / "iaei" / "modeling" / "neural_governance.py"
    text = path.read_text(encoding="utf-8")
    old = "APPROVED_SEEDS = (20260725, 20260726, 20260727)"
    new = "APPROVED_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)"
    if old not in text and new not in text:
        raise RuntimeError("Unexpected Gate 6C seed constant")
    path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def align_validator() -> None:
    path = ROOT / "scripts" / "validate_gate_6c1.py"
    path.write_text(
        '''from __future__ import annotations

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
            "Gate 6B closure validation failed:\\n"
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
''',
        encoding="utf-8",
        newline="\n",
    )


def align_tests() -> None:
    path = ROOT / "tests" / "test_neural_forecasting_governance.py"
    path.write_text(
        '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from iaei.contracts import (
    ContractError,
    validate_gate_6c1_closure_manifest,
    validate_neural_forecasting_contract,
)
from iaei.modeling.neural_governance import (
    APPROVED_ALGORITHMS,
    APPROVED_SEEDS,
    Gate6C1ExecutionProhibited,
    assert_no_gate_6c_execution_artifacts,
    build_gate_6c1_plan,
    causal_window,
    deterministic_cpu_environment,
    prohibit_gate_6c1_execution,
)
from iaei.paths import ROOT, SCHEMAS


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _errors(schema_name: str, payload: dict[str, Any]) -> list[Any]:
    schema = _load_json(SCHEMAS / schema_name)
    return list(Draft202012Validator(schema).iter_errors(payload))


def test_gate_6c1_contract_and_plan_are_frozen() -> None:
    contract = validate_neural_forecasting_contract()
    plan = build_gate_6c1_plan()

    assert contract["contract_version"] == "1.1.0"
    assert contract["status"] == "approved_for_implementation"
    assert tuple(candidate.algorithm_id for candidate in plan.candidates) == APPROVED_ALGORITHMS
    assert plan.seeds == APPROVED_SEEDS
    assert len(plan.seeds) == 5
    assert plan.outer_fold_count == 4
    assert plan.purge_intervals == 4
    assert plan.maximum_prediction_origin_exclusive == 28028
    assert plan.maximum_target_dependency_exclusive == 28032
    assert plan.canonical_device == "cpu"
    assert plan.fitting_permitted is False


def test_gate_6c1_contract_preserves_human_decision_and_v1_boundary() -> None:
    contract = validate_neural_forecasting_contract()

    assert contract["promotion"]["automatic_promotion_permitted"] is False
    assert contract["promotion"]["final_authority"] == "human"
    assert contract["v1_boundary"]["locked_test_access_permitted"] is False
    assert contract["v1_boundary"]["locked_prediction_parsing_permitted"] is False
    assert contract["v1_boundary"]["confirmatory_evaluation_permitted"] is False
    assert contract["data_boundary"]["admissible_partitions"] == [
        "training",
        "validation",
    ]


def test_deterministic_cpu_environment_is_seed_bounded() -> None:
    for seed in APPROVED_SEEDS:
        environment = deterministic_cpu_environment(seed)
        assert environment["PYTHONHASHSEED"] == str(seed)
        assert environment["IAEI_GATE_6C_SEED"] == str(seed)
        assert environment["IAEI_CANONICAL_DEVICE"] == "cpu"
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["MKL_NUM_THREADS"] == "1"
        assert environment["OPENBLAS_NUM_THREADS"] == "1"
        assert environment["NUMEXPR_NUM_THREADS"] == "1"

    with pytest.raises(ContractError):
        deterministic_cpu_environment(7)


def test_causal_window_respects_context_and_locked_boundary() -> None:
    window = causal_window(
        prediction_origin=28027,
        context_length=96,
        horizon=1,
    )
    assert window.context_start == 27932
    assert window.context_end_exclusive == 28028
    assert window.prediction_origin == 28027
    assert window.target_index == 28028

    with pytest.raises(ContractError):
        causal_window(prediction_origin=28028, context_length=96)
    with pytest.raises(ContractError):
        causal_window(prediction_origin=40, context_length=96)
    with pytest.raises(ContractError):
        causal_window(prediction_origin=100, context_length=96, horizon=2)


def test_gate_6c1_fails_closed_on_model_execution_actions() -> None:
    for action in ("fit", "train", "predict", "evaluate", "score", "optimize", "search"):
        with pytest.raises(Gate6C1ExecutionProhibited):
            prohibit_gate_6c1_execution(action)

    prohibit_gate_6c1_execution("validate_contract")


def test_gate_6c1_has_no_execution_artifacts() -> None:
    assert_no_gate_6c_execution_artifacts(ROOT)


def test_gate_6c1_schemas_are_valid_json_schemas() -> None:
    schema_names = (
        "neural_forecasting_contract.schema.json",
        "neural_seed_evidence.schema.json",
        "neural_candidate_evidence.schema.json",
        "neural_promotion_decision.schema.json",
        "gate_6c1_closure_manifest.schema.json",
    )
    for schema_name in schema_names:
        Draft202012Validator.check_schema(_load_json(SCHEMAS / schema_name))


def test_future_seed_evidence_schema_enforces_cpu_and_boundaries() -> None:
    payload = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "nhits_compact",
        "seed": 20260721,
        "fold_id": 1,
        "partition": "validation",
        "prediction_origin_max": 28027,
        "target_dependency_max": 28028,
        "metrics": {"mae": 3.8, "peak_mae": 13.9},
        "resources": {
            "wall_clock_seconds": 10.0,
            "peak_memory_mb": 512.0,
            "model_size_bytes": 4096,
            "p95_inference_latency_ms_per_1000_rows": 2.0,
        },
        "portability": {
            "device": "cpu",
            "gpu_required": False,
            "cpu_portability_passed": True,
        },
        "status": "success",
    }
    assert _errors("neural_seed_evidence.schema.json", payload) == []

    invalid = dict(payload)
    invalid["seed"] = 20260726
    assert _errors("neural_seed_evidence.schema.json", invalid)

    invalid = dict(payload)
    invalid["prediction_origin_max"] = 28028
    assert _errors("neural_seed_evidence.schema.json", invalid)


def test_future_candidate_and_decision_schemas_preserve_governance() -> None:
    candidate = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "tide_compact",
        "seed_count": 5,
        "outer_fold_count": 4,
        "validation_origin_count_per_seed": 7004,
        "aggregate": {
            "mean_mae": 3.8,
            "mean_peak_mae": 14.0,
            "relative_mae_improvement_vs_v1": 0.02,
            "positive_outer_folds": 3,
        },
        "stability": {
            "across_seed_mae_standard_deviation": 0.02,
            "across_seed_peak_mae_standard_deviation": 0.04,
            "outer_fold_mae_dispersion": 0.1,
        },
        "constraints": {
            "chronology_passed": True,
            "locked_test_excluded": True,
            "v1_immutable": True,
            "resource_limits_passed": True,
            "cpu_portability_passed": True,
        },
        "promotion_eligible": True,
    }
    assert _errors("neural_candidate_evidence.schema.json", candidate) == []

    decision = {
        "schema_version": "1.0.0",
        "gate": "6C",
        "candidate_id": "tide_compact",
        "decision": "defer",
        "human_authority": True,
        "automatic_promotion_permitted": False,
        "locked_test_accessed": False,
        "confirmatory_evaluation_performed": False,
        "rationale": "Additional validation evidence is required before any promotion decision.",
        "next_action": "request_additional_validation",
    }
    assert _errors("neural_promotion_decision.schema.json", decision) == []

    invalid = dict(decision)
    invalid["automatic_promotion_permitted"] = True
    assert _errors("neural_promotion_decision.schema.json", invalid)


def test_gate_6c1_closure_manifest_is_design_only() -> None:
    closure = validate_gate_6c1_closure_manifest()
    assert closure["status"] in {"implementation_complete_pending_ci", "closed"}
    assert closure["implementation_only"] is True
    assert closure["candidate_ids"] == list(APPROVED_ALGORITHMS)
    assert closure["seeds"] == list(APPROVED_SEEDS)
    assert closure["controls"]["model_fitting_performed"] is False
    assert closure["controls"]["validation_predictions_generated"] is False
    assert closure["controls"]["validation_metrics_calculated"] is False
    assert closure["controls"]["locked_test_accessed"] is False
    assert closure["controls"]["confirmatory_evaluation_performed"] is False
    assert closure["controls"]["v1_immutable"] is True
    assert closure["next_gate"] == "6C2"
    assert closure["blocked_gates"] == ["6C3", "6D"]
''',
        encoding="utf-8",
        newline="\n",
    )


def align_docs_and_workflow() -> None:
    protocol_path = ROOT / "docs" / "GATE_6C_NEURAL_FORECASTING_PROTOCOL.md"
    protocol = protocol_path.read_text(encoding="utf-8")
    protocol = protocol.replace("use three fixed seeds per configuration;", "use five fixed seeds per configuration;")
    protocol = protocol.replace(
        "20260725\n20260726\n20260727",
        "20260721\n20260722\n20260723\n20260724\n20260725",
    )
    protocol_path.write_text(protocol, encoding="utf-8", newline="\n")

    scope_path = ROOT / "docs" / "GATE_6C1_SCOPE_LOCK.md"
    scope = scope_path.read_text(encoding="utf-8")
    scope = scope.replace(
        "- `20260725`;\n- `20260726`;\n- `20260727`.",
        "- `20260721`;\n- `20260722`;\n- `20260723`;\n- `20260724`;\n- `20260725`.",
    )
    scope_path.write_text(scope, encoding="utf-8", newline="\n")

    checklist_path = ROOT / "docs" / "GATE_6C1_IMPLEMENTATION_CHECKLIST.md"
    checklist = checklist_path.read_text(encoding="utf-8")
    checklist = checklist.replace(
        "fixed seeds: `20260725`, `20260726`, `20260727`;",
        "fixed seeds: `20260721`, `20260722`, `20260723`, `20260724`, `20260725`;",
    )
    checklist = checklist.replace(
        "Head of Research approval",
        "explicit human approval",
    )
    checklist_path.write_text(checklist, encoding="utf-8", newline="\n")

    workflow_path = ROOT / ".github" / "workflows" / "gate-6c1-neural-contract.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    workflow = workflow.replace(
        "- Seeds: 20260725, 20260726, 20260727",
        "- Seeds: 20260721, 20260722, 20260723, 20260724, 20260725",
    )
    workflow_path.write_text(workflow, encoding="utf-8", newline="\n")

    reconciliation_path = ROOT / "docs" / "GATE_6C1_SEED_GOVERNANCE_RECONCILIATION.md"
    reconciliation_path.write_text(
        '''# Gate 6C1 seed-governance reconciliation

## Status

Resolved before neural fitting.

## Approved decision

Gate 6C restores the five stochastic seeds frozen by the Gate 6A optimization-governance contract:

- `20260721`;
- `20260722`;
- `20260723`;
- `20260724`;
- `20260725`.

The Gate 6C contract is versioned as `1.1.0` to record the pre-execution reconciliation.

## Methodological effect

The decision preserves parent-contract consistency, satisfies the minimum stochastic seed count, keeps seed substitution prohibited, and strengthens across-seed stability evidence. It does not change candidate families, configurations, folds, purge intervals, performance objectives, promotion thresholds, resource limits, or the locked evidence boundary.

## Preserved controls

- model fitting performed: false;
- validation predictions generated: false;
- validation metrics calculated: false;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- V1 mutation: false;
- automatic promotion: false.

## Sequence

Gate 6C1 may close only after all contracts, schemas, source controls, tests, public-content checks, V1 immutability checks, and GitHub workflows are green. Gate 6C2 remains blocked until that closure is complete.
''',
        encoding="utf-8",
        newline="\n",
    )

    status_path = ROOT / "docs" / "GATE_6C1_IMPLEMENTATION_STATUS.md"
    status_path.write_text(
        '''# Gate 6C1 implementation status

## Current state

The Gate 6C1 implementation-only package is complete pending final GitHub validation and formal closure.

It contains:

- a versioned machine-readable neural forecasting contract;
- strict contract and evidence schemas;
- compact N-HiTS, compact TiDE, and compact PatchTST blueprints;
- five frozen Gate 6A seeds per candidate configuration;
- deterministic CPU controls and causal-window boundaries;
- fail-closed guards against fitting, prediction, evaluation, and optimization actions;
- repository contract registration;
- source, schema, boundary, and future-evidence tests;
- a read-only GitHub-native validation workflow;
- quarantine controls for the superseded PR #12 execution;
- an approved seed-governance reconciliation record.

## Preserved boundaries

- model fitting performed: false;
- validation predictions generated: false;
- validation metrics calculated: false;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- V1 mutation: false;
- automatic promotion: false.

## Assessment

The implementation is stronger than a documentation-only gate because the neural protocol is represented as executable contracts, schemas, source guards, chronology controls, and GitHub-native validation. Restoring five seeds increases the credibility of stability evidence while preserving the original Gate 6A governance.

## Next action

Run the complete Gate 6C1 validation package. After all required checks are green, record formal closure and request approval for Gate 6C2 governed validation execution.
''',
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    align_contract()
    align_schemas()
    align_closure_manifest()
    align_python_source()
    align_validator()
    align_tests()
    align_docs_and_workflow()
    print("Gate 6C five-seed governance alignment complete")


if __name__ == "__main__":
    main()
