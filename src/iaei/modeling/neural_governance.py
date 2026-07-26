from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iaei.contracts import ContractError, validate_neural_forecasting_contract
from iaei.paths import ROOT

APPROVED_ALGORITHMS = (
    "nhits_compact",
    "tide_compact",
    "patchtst_compact",
)
APPROVED_SEEDS = (20260725, 20260726, 20260727)
PROHIBITED_EXECUTION_ACTIONS = frozenset(
    {
        "fit",
        "train",
        "predict",
        "evaluate",
        "score",
        "optimize",
        "search",
    }
)


@dataclass(frozen=True)
class NeuralCandidateSpec:
    """Immutable Gate 6C candidate blueprint without executable model state."""

    algorithm_id: str
    hypothesis: str
    context_length: int
    horizon: int
    configuration: dict[str, int | float]


@dataclass(frozen=True)
class Gate6C1Plan:
    """Implementation-only plan used to verify Gate 6C protocol conformance."""

    candidates: tuple[NeuralCandidateSpec, ...]
    seeds: tuple[int, ...]
    outer_fold_count: int
    purge_intervals: int
    maximum_prediction_origin_exclusive: int
    maximum_target_dependency_exclusive: int
    canonical_device: str
    fitting_permitted: bool


@dataclass(frozen=True)
class CausalWindow:
    """Index-only causal window description. It does not access analytical data."""

    context_start: int
    context_end_exclusive: int
    prediction_origin: int
    target_index: int


class Gate6C1ExecutionProhibited(ContractError):
    """Raised when an execution action is attempted during Gate 6C1."""


def _candidate_specs(contract: dict[str, Any]) -> tuple[NeuralCandidateSpec, ...]:
    return tuple(
        NeuralCandidateSpec(
            algorithm_id=str(item["algorithm_id"]),
            hypothesis=str(item["hypothesis"]),
            context_length=int(item["context_length"]),
            horizon=int(item["horizon"]),
            configuration=dict(item["configuration"]),
        )
        for item in contract["candidate_families"]
    )


def _validate_candidate_shapes(candidates: tuple[NeuralCandidateSpec, ...]) -> None:
    by_id = {candidate.algorithm_id: candidate for candidate in candidates}

    if tuple(by_id) != APPROVED_ALGORITHMS:
        raise ContractError("Gate 6C candidate order or identity changed")

    for candidate in candidates:
        if candidate.context_length != 96 or candidate.horizon != 1:
            raise ContractError(
                f"Unexpected temporal shape for {candidate.algorithm_id}"
            )
        if int(candidate.configuration["max_epochs"]) != 40:
            raise ContractError(
                f"Unexpected epoch budget for {candidate.algorithm_id}"
            )
        if int(candidate.configuration["batch_size"]) != 256:
            raise ContractError(
                f"Unexpected batch size for {candidate.algorithm_id}"
            )

    patch = by_id["patchtst_compact"].configuration
    if int(patch["patch_length"]) != 16 or int(patch["stride"]) != 8:
        raise ContractError("PatchTST patch geometry changed")
    if int(patch["hidden_size"]) % int(patch["attention_heads"]) != 0:
        raise ContractError("PatchTST hidden size must divide by attention heads")


def build_gate_6c1_plan() -> Gate6C1Plan:
    """Build and validate the frozen implementation-only neural plan."""

    contract = validate_neural_forecasting_contract()
    if contract["gate"] != "6C":
        raise ContractError("Unexpected neural forecasting gate")
    if contract["status"] != "approved_for_implementation":
        raise ContractError("Gate 6C contract is not approved for implementation")

    search = contract["search"]
    candidates = _candidate_specs(contract)
    seeds = tuple(int(seed) for seed in search["seeds"])

    if seeds != APPROVED_SEEDS:
        raise ContractError("Gate 6C seed set changed")
    if int(search["configurations_per_family"]) != 1:
        raise ContractError("Gate 6C must use one configuration per family")
    if int(search["unique_configuration_count"]) != len(candidates):
        raise ContractError("Gate 6C configuration count is inconsistent")
    if int(search["seed_count"]) != len(seeds):
        raise ContractError("Gate 6C seed count is inconsistent")
    if int(search["max_parallel_trials"]) != 1:
        raise ContractError("Gate 6C canonical execution must remain serial")
    if bool(search["internal_early_stopping_permitted"]):
        raise ContractError("Gate 6C internal early stopping is prohibited")

    _validate_candidate_shapes(candidates)

    boundary = contract["data_boundary"]
    resources = contract["resource_constraints"]
    if boundary["admissible_partitions"] != ["training", "validation"]:
        raise ContractError("Gate 6C evidence partitions changed")
    if resources["canonical_device"] != "cpu":
        raise ContractError("Gate 6C canonical device must remain CPU")
    if bool(resources["gpu_required"]):
        raise ContractError("Gate 6C cannot require a GPU")

    v1 = contract["v1_boundary"]
    if any(
        bool(v1[key])
        for key in (
            "locked_test_access_permitted",
            "locked_prediction_parsing_permitted",
            "confirmatory_evaluation_permitted",
        )
    ):
        raise ContractError("Gate 6C violates the immutable V1 boundary")

    return Gate6C1Plan(
        candidates=candidates,
        seeds=seeds,
        outer_fold_count=int(boundary["outer_fold_count"]),
        purge_intervals=int(boundary["purge_intervals"]),
        maximum_prediction_origin_exclusive=int(
            boundary["maximum_prediction_origin_exclusive"]
        ),
        maximum_target_dependency_exclusive=int(
            boundary["maximum_target_dependency_exclusive"]
        ),
        canonical_device=str(resources["canonical_device"]),
        fitting_permitted=False,
    )


def deterministic_cpu_environment(seed: int) -> dict[str, str]:
    """Return the environment controls that Gate 6C2 must apply per seed."""

    if seed not in APPROVED_SEEDS:
        raise ContractError(f"Unapproved Gate 6C seed: {seed}")
    return {
        "PYTHONHASHSEED": str(seed),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "IAEI_CANONICAL_DEVICE": "cpu",
        "IAEI_GATE_6C_SEED": str(seed),
    }


def causal_window(
    *,
    prediction_origin: int,
    context_length: int,
    horizon: int = 1,
) -> CausalWindow:
    """Describe a causal context and target dependency without reading data."""

    plan = build_gate_6c1_plan()
    if context_length <= 0:
        raise ContractError("Context length must be positive")
    if horizon != 1:
        raise ContractError("Gate 6C horizon must remain one interval")
    if prediction_origin >= plan.maximum_prediction_origin_exclusive:
        raise ContractError("Prediction origin crosses the Gate 6C boundary")

    context_start = prediction_origin - context_length + 1
    if context_start < 0:
        raise ContractError("Prediction origin lacks the required causal history")

    target_index = prediction_origin + horizon
    if target_index >= plan.maximum_target_dependency_exclusive:
        raise ContractError("Target dependency crosses the Gate 6C boundary")

    return CausalWindow(
        context_start=context_start,
        context_end_exclusive=prediction_origin + 1,
        prediction_origin=prediction_origin,
        target_index=target_index,
    )


def prohibit_gate_6c1_execution(action: str) -> None:
    """Fail closed when Gate 6C1 is asked to execute a modeling action."""

    normalized = action.strip().lower()
    if normalized in PROHIBITED_EXECUTION_ACTIONS:
        raise Gate6C1ExecutionProhibited(
            f"Gate 6C1 is implementation-only; action is prohibited: {normalized}"
        )


def assert_no_gate_6c_execution_artifacts(root: Path = ROOT) -> None:
    """Ensure Gate 6C1 has not created model or validation evidence."""

    contract = validate_neural_forecasting_contract()
    output_dir = root / str(contract["outputs"]["directory"])
    if output_dir.exists() and any(output_dir.rglob("*")):
        raise Gate6C1ExecutionProhibited(
            f"Gate 6C execution artifacts exist during Gate 6C1: {output_dir}"
        )
