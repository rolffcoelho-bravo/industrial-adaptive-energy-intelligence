from __future__ import annotations

import json
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
    method_ids = tuple(
        str(item["method_id"])
        for item in observed["candidate_families"]
    )
    configuration_count = sum(
        len(item["configurations"])
        for item in observed["candidate_families"]
    )
    return Gate6E1Plan(
        method_ids=method_ids,
        configuration_count=configuration_count,
        target_coverage_levels=tuple(
            float(value)
            for value in observed["calibration_protocol"][
                "target_coverage_levels"
            ]
        ),
        calibration_permitted=False,
        interval_generation_permitted=False,
        locked_test_access_permitted=bool(
            observed["evidence_boundary"][
                "locked_test_partition_permitted"
            ]
        ),
        point_model_search_permitted=bool(
            observed["point_model_boundary"][
                "point_model_search_permitted"
            ]
        ),
        automatic_promotion_permitted=bool(
            observed["optimization"]["automatic_promotion_permitted"]
        ),
        next_gate_authorized=bool(observed["next_gate_authorized"]),
    )


def _validate_authorized_gate_6e2_successor(output: Path) -> None:
    observed_artifacts = {
        name
        for name in PROHIBITED_EXECUTION_ARTIFACTS
        if (output / name).exists()
    }
    expected_artifacts = set(PROHIBITED_EXECUTION_ARTIFACTS)
    if observed_artifacts != expected_artifacts:
        missing = sorted(expected_artifacts.difference(observed_artifacts))
        present = sorted(observed_artifacts)
        raise RuntimeError(
            "Gate 6E successor evidence is partial or inconsistent: "
            f"present={present}, missing={missing}"
        )

    manifest_path = output / "gate_6e_execution_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Gate 6E2 execution manifest is unreadable"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Gate 6E2 execution manifest is not an object")

    expected_values = {
        "gate": "6E",
        "subgate": "6E2",
        "status": "validation_complete_pending_human_decision",
        "retained_point_model": "v1_frozen_champion",
        "next_gate": "6E3",
        "next_gate_authorized": False,
        "automatic_promotion_permitted": False,
        "locked_test_accessed": False,
        "locked_predictions_parsed": False,
        "confirmatory_evaluation_performed": False,
        "point_model_search_performed": False,
        "point_model_mutated": False,
        "v1_immutable": True,
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": expected}
        for key, expected in expected_values.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Gate 6E2 successor manifest violates the Gate 6E1 boundary: "
            f"{mismatches}"
        )
    if manifest.get("blocked_gates") != ["6E3", "6F", "6G"]:
        raise RuntimeError(
            "Gate 6E2 successor manifest changed the blocked-gate sequence"
        )


def assert_gate_6e1_execution_boundary(root: Path = ROOT) -> None:
    output = root / "outputs" / "v2" / "gate_6e"
    hits = [
        name
        for name in PROHIBITED_EXECUTION_ARTIFACTS
        if (output / name).exists()
    ]
    if not hits:
        return
    _validate_authorized_gate_6e2_successor(output)


def assert_no_gate_6e_execution_artifacts(root: Path = ROOT) -> None:
    """Retained compatibility alias for the predecessor boundary check."""

    assert_gate_6e1_execution_boundary(root)
