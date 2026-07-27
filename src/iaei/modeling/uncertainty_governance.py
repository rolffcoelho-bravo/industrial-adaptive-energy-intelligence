from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iaei.paths import ROOT
from iaei.uncertainty_contracts import validate_uncertainty_contract


APPROVED_UNCERTAINTY_METHODS = (
    "expanding_absolute_conformal",
    "rolling_absolute_conformal",
    "adaptive_absolute_conformal",
)
REFERENCE_CONFIGURATION = "expanding_all"
PROHIBITED_EXECUTION_ARTIFACTS = (
    "configuration_results.csv",
    "coverage_results.csv",
    "outer_fold_results.csv",
    "interval_predictions.parquet",
    "calibration_lineage.json",
    "resource_evidence.csv",
    "failure_records.json",
    "promotion_recommendation.json",
    "gate_6e_execution_manifest.json",
)


@dataclass(frozen=True)
class Gate6E1Plan:
    method_ids: tuple[str, ...]
    configuration_count: int
    target_coverage_levels: tuple[float, ...]
    calibration_permitted: bool
    interval_generation_permitted: bool
    locked_test_access_permitted: bool
    point_model_search_permitted: bool
    automatic_promotion_permitted: bool
    next_gate_authorized: bool


def build_gate_6e1_plan(
    contract: dict[str, Any] | None = None,
) -> Gate6E1Plan:
    observed = contract or validate_uncertainty_contract()
    method_ids = tuple(str(item["method_id"]) for item in observed["candidate_families"])
    configuration_count = sum(
        len(item["configurations"]) for item in observed["candidate_families"]
    )
    return Gate6E1Plan(
        method_ids=method_ids,
        configuration_count=configuration_count,
        target_coverage_levels=tuple(
            float(value)
            for value in observed["calibration_protocol"]["target_coverage_levels"]
        ),
        calibration_permitted=False,
        interval_generation_permitted=False,
        locked_test_access_permitted=bool(
            observed["evidence_boundary"]["locked_test_partition_permitted"]
        ),
        point_model_search_permitted=bool(
            observed["point_model_boundary"]["point_model_search_permitted"]
        ),
        automatic_promotion_permitted=bool(
            observed["optimization"]["automatic_promotion_permitted"]
        ),
        next_gate_authorized=bool(observed["next_gate_authorized"]),
    )


def assert_no_gate_6e_execution_artifacts(root: Path = ROOT) -> None:
    output = root / "outputs" / "v2" / "gate_6e"
    hits = [name for name in PROHIBITED_EXECUTION_ARTIFACTS if (output / name).exists()]
    if hits:
        raise RuntimeError(
            "Gate 6E1 contains prohibited execution evidence: " + ", ".join(hits)
        )
