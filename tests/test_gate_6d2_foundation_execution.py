from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from iaei.foundation_contracts import (
    validate_foundation_model_contract,
    validate_gate_6d1_closure_manifest,
)
from iaei.paths import SCHEMAS
from iaei.v2.foundation_adapters import (
    FoundationAdapterError,
    _quantile_indices,
    _timed_chunks,
)
from iaei.v2.foundation_evidence import _recommendation
from iaei.v2.foundation_forecasting import (
    FoundationForecastingError,
    _candidate,
    _causal_windows,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gate_6d2_contract_preserves_execution_boundaries() -> None:
    contract = validate_foundation_model_contract()
    closure = validate_gate_6d1_closure_manifest()

    assert contract["next_gate"] == "6D2"
    assert closure["status"] == "closed"
    assert closure["next_gate"] == "6D2"
    assert closure["next_gate_authorized"] is False
    assert contract["benchmark_protocol"]["mode"] == "zero_shot_univariate"
    assert contract["benchmark_protocol"]["context_length_intervals"] == 672
    assert contract["benchmark_protocol"]["horizon_intervals"] == 1
    assert contract["benchmark_protocol"]["fine_tuning_permitted"] is False
    assert contract["benchmark_protocol"]["calibration_permitted"] is False
    assert contract["benchmark_protocol"]["hyperparameter_search_permitted"] is False
    assert contract["v1_boundary"]["locked_test_access_permitted"] is False


def test_gate_6d2_candidate_and_license_identities_are_frozen() -> None:
    contract = validate_foundation_model_contract()
    expected = {
        "chronos_2_zero_shot": (True, True),
        "timesfm_2_5_zero_shot": (True, True),
        "moirai_2_research_zero_shot": (False, False),
    }
    observed = {
        item["candidate_id"]: (
            item["commercial_use_eligible"],
            item["promotion_eligible"],
        )
        for item in contract["candidate_models"]
    }
    assert observed == expected
    assert _candidate(contract, "chronos_2_zero_shot")["model_id"] == "amazon/chronos-2"
    with pytest.raises(FoundationForecastingError):
        _candidate(contract, "unknown")


def test_gate_6d2_causal_windows_are_origin_safe() -> None:
    source = np.arange(1000, dtype=float)
    origins = np.array([671, 672, 999], dtype=np.int64)
    windows = _causal_windows(source, origins, context_length=672)

    assert windows.shape == (3, 672)
    assert windows[0, 0] == 0.0
    assert windows[0, -1] == 671.0
    assert windows[2, 0] == 328.0
    assert windows[2, -1] == 999.0

    with pytest.raises(FoundationForecastingError):
        _causal_windows(source, np.array([670]), context_length=672)


def test_quantile_mapping_and_batched_adapter_are_deterministic() -> None:
    assert _quantile_indices([0.1, 0.5, 0.9], (0.1, 0.5, 0.9)) == [0, 1, 2]
    with pytest.raises(FoundationAdapterError):
        _quantile_indices([0.1, 0.5], (0.1, 0.5, 0.9))

    windows = np.arange(60, dtype=float).reshape(10, 6)

    def infer(chunk: np.ndarray) -> np.ndarray:
        median = chunk[:, -1]
        return np.column_stack([median - 1.0, median, median + 1.0])

    prediction, latency = _timed_chunks(
        windows,
        batch_size=4,
        infer_chunk=infer,
        maximum_seconds=10.0,
    )
    assert prediction.shape == (10, 3)
    assert latency and len(latency) == 3
    assert np.allclose(prediction[:, 1], windows[:, -1])


def test_gate_6d2_recommendation_never_promotes_noncommercial_model() -> None:
    contract = validate_foundation_model_contract()
    rows = []
    for candidate in contract["candidate_models"]:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": "success",
                "relative_mae_improvement_vs_v1": 0.05,
                "relative_peak_mae_change_vs_v1": -0.05,
                "positive_outer_folds": 4,
                "maximum_single_fold_mae_relative_degradation": 0.0,
                "chronology_passed": True,
                "resource_limits_passed": True,
                "cpu_execution_passed": True,
                "deterministic_replay_passed": True,
                "quantile_order_passed": True,
                "commercial_use_eligible": candidate["commercial_use_eligible"],
                "promotion_eligible_by_license": candidate["promotion_eligible"],
            }
        )
    recommendation = _recommendation(pd.DataFrame(rows), contract)
    decisions = {
        item["candidate_id"]: item for item in recommendation["candidates"]
    }
    assert decisions["chronos_2_zero_shot"]["promotion_eligible"] is True
    assert decisions["timesfm_2_5_zero_shot"]["promotion_eligible"] is True
    assert decisions["moirai_2_research_zero_shot"]["promotion_eligible"] is False
    assert "commercial_use_eligibility" in decisions[
        "moirai_2_research_zero_shot"
    ]["failed_requirements"]


def test_gate_6d2_schemas_are_valid_json_schemas() -> None:
    names = (
        "foundation_candidate_evidence.schema.json",
        "foundation_provenance_manifest.schema.json",
        "foundation_promotion_recommendation.schema.json",
        "foundation_execution_manifest.schema.json",
    )
    for name in names:
        Draft202012Validator.check_schema(_load_json(SCHEMAS / name))


def test_candidate_schema_accepts_recorded_replay_failure() -> None:
    schema = _load_json(SCHEMAS / "foundation_candidate_evidence.schema.json")
    payload = {
        "schema_version": "1.0.0",
        "gate": "6D",
        "subgate": "6D2",
        "candidate_id": "chronos_2_zero_shot",
        "status": "success",
        "benchmark_admissible": True,
        "commercial_use_eligible": True,
        "promotion_eligible_by_license": True,
        "validation_origin_count": 7004,
        "outer_fold_count": 4,
        "mean_mae": 4.743580802977936,
        "mean_peak_mae": 21.29347326376154,
        "relative_mae_improvement_vs_v1": -0.18599801157854728,
        "relative_peak_mae_change_vs_v1": 0.15691451243821594,
        "positive_outer_folds": 0,
        "maximum_single_fold_mae_relative_degradation": 0.21047123731077874,
        "chronology_passed": True,
        "resource_limits_passed": True,
        "cpu_execution_passed": True,
        "deterministic_replay_passed": False,
        "quantile_order_passed": False,
        "quantile_crossing_rate": 0.00028555111364934324,
    }
    assert list(Draft202012Validator(schema).iter_errors(payload)) == []
