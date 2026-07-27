from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iaei.contracts import ContractError, validate_foundation_model_contract
from iaei.paths import ROOT

APPROVED_FOUNDATION_MODELS = (
    "chronos_2_zero_shot",
    "timesfm_2_5_zero_shot",
    "moirai_2_research_zero_shot",
)
RESEARCH_ONLY_MODELS = ("moirai_2_research_zero_shot",)
APPROVED_CONTEXT_LENGTH = 672
APPROVED_HORIZON = 1
MAXIMUM_PREDICTION_ORIGIN_EXCLUSIVE = 28028
MAXIMUM_TARGET_DEPENDENCY_EXCLUSIVE = 28032


class Gate6D1ExecutionProhibited(ContractError):
    """Raised when Gate 6D1 attempts model download, inference, or evaluation."""


@dataclass(frozen=True)
class FoundationCandidate:
    candidate_id: str
    model_id: str
    model_revision: str
    weight_sha256: str
    source_repository: str
    source_revision: str
    weights_license: str
    maximum_supported_context_intervals: int
    benchmark_admissible: bool
    commercial_use_eligible: bool
    promotion_eligible: bool


@dataclass(frozen=True)
class Gate6D1Plan:
    candidates: tuple[FoundationCandidate, ...]
    context_length_intervals: int
    horizon_intervals: int
    outer_fold_count: int
    purge_intervals: int
    validation_origin_count: int
    maximum_prediction_origin_exclusive: int
    maximum_target_dependency_exclusive: int
    canonical_device: str
    model_download_permitted: bool
    inference_permitted: bool
    fine_tuning_permitted: bool
    locked_test_access_permitted: bool
    final_authority: str


@dataclass(frozen=True)
class FoundationWindow:
    context_start: int
    context_end_exclusive: int
    prediction_origin: int
    target_index: int


def build_gate_6d1_plan(
    contract: dict[str, Any] | None = None,
) -> Gate6D1Plan:
    observed = contract or validate_foundation_model_contract()
    protocol = observed["benchmark_protocol"]
    data_boundary = observed["data_boundary"]
    controls = observed["network_and_artifact_controls"]

    candidates = tuple(
        FoundationCandidate(
            candidate_id=str(candidate["candidate_id"]),
            model_id=str(candidate["model_id"]),
            model_revision=str(candidate["model_revision"]),
            weight_sha256=str(candidate["weight_sha256"]),
            source_repository=str(candidate["source_repository"]),
            source_revision=str(candidate["source_revision"]),
            weights_license=str(candidate["weights_license"]),
            maximum_supported_context_intervals=int(
                candidate["maximum_supported_context_intervals"]
            ),
            benchmark_admissible=bool(candidate["benchmark_admissible"]),
            commercial_use_eligible=bool(candidate["commercial_use_eligible"]),
            promotion_eligible=bool(candidate["promotion_eligible"]),
        )
        for candidate in observed["candidate_models"]
    )

    if tuple(candidate.candidate_id for candidate in candidates) != (
        APPROVED_FOUNDATION_MODELS
    ):
        raise ContractError("Gate 6D candidate identities changed")
    if any(
        candidate.maximum_supported_context_intervals < APPROVED_CONTEXT_LENGTH
        for candidate in candidates
    ):
        raise ContractError("A Gate 6D candidate cannot support the frozen context")
    if any(
        candidate.promotion_eligible and not candidate.commercial_use_eligible
        for candidate in candidates
    ):
        raise ContractError("A non-commercial model cannot be promotion eligible")

    return Gate6D1Plan(
        candidates=candidates,
        context_length_intervals=int(protocol["context_length_intervals"]),
        horizon_intervals=int(protocol["horizon_intervals"]),
        outer_fold_count=int(data_boundary["outer_fold_count"]),
        purge_intervals=int(data_boundary["purge_intervals"]),
        validation_origin_count=int(data_boundary["validation_origin_count"]),
        maximum_prediction_origin_exclusive=int(
            data_boundary["maximum_prediction_origin_exclusive"]
        ),
        maximum_target_dependency_exclusive=int(
            data_boundary["maximum_target_dependency_exclusive"]
        ),
        canonical_device=str(observed["resource_constraints"]["canonical_device"]),
        model_download_permitted=bool(
            controls["gate_6d1_model_download_permitted"]
        ),
        inference_permitted=bool(controls["gate_6d1_inference_permitted"]),
        fine_tuning_permitted=bool(protocol["fine_tuning_permitted"]),
        locked_test_access_permitted=bool(
            observed["v1_boundary"]["locked_test_access_permitted"]
        ),
        final_authority=str(observed["promotion"]["final_authority"]),
    )


def causal_foundation_window(
    *,
    prediction_origin: int,
    context_length: int = APPROVED_CONTEXT_LENGTH,
    horizon: int = APPROVED_HORIZON,
) -> FoundationWindow:
    if context_length != APPROVED_CONTEXT_LENGTH:
        raise ContractError("Gate 6D context length is frozen at 672 intervals")
    if horizon != APPROVED_HORIZON:
        raise ContractError("Gate 6D forecast horizon is frozen at one interval")
    if prediction_origin >= MAXIMUM_PREDICTION_ORIGIN_EXCLUSIVE:
        raise ContractError("Gate 6D prediction origin crosses the validation boundary")

    context_end_exclusive = prediction_origin + 1
    context_start = context_end_exclusive - context_length
    if context_start < 0:
        raise ContractError("Insufficient causal history for the Gate 6D context")

    target_index = prediction_origin + horizon
    if target_index >= MAXIMUM_TARGET_DEPENDENCY_EXCLUSIVE:
        raise ContractError("Gate 6D target dependency crosses the frozen boundary")

    return FoundationWindow(
        context_start=context_start,
        context_end_exclusive=context_end_exclusive,
        prediction_origin=prediction_origin,
        target_index=target_index,
    )


def commercial_promotion_eligible(candidate_id: str) -> bool:
    plan = build_gate_6d1_plan()
    matches = [candidate for candidate in plan.candidates if candidate.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ContractError(f"Unknown Gate 6D candidate: {candidate_id}")
    candidate = matches[0]
    return candidate.commercial_use_eligible and candidate.promotion_eligible


def prohibit_gate_6d1_execution(action: str) -> None:
    prohibited = {
        "download",
        "load_weights",
        "infer",
        "forecast",
        "fit",
        "fine_tune",
        "calibrate",
        "evaluate",
        "score",
        "promote",
    }
    if action.strip().lower() in prohibited:
        raise Gate6D1ExecutionProhibited(
            f"Gate 6D1 is implementation-only; action is prohibited: {action}"
        )


def assert_no_gate_6d_execution_artifacts(root: Path = ROOT) -> None:
    output_directory = root / "outputs" / "v2" / "gate_6d"
    if not output_directory.exists():
        return
    observed = sorted(path.name for path in output_directory.iterdir())
    if observed:
        raise Gate6D1ExecutionProhibited(
            f"Gate 6D1 found unauthorized execution artifacts: {observed}"
        )
