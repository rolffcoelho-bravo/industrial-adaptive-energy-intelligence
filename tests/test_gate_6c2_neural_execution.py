from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from iaei.contracts import (  # noqa: E402
    validate_neural_forecasting_contract,
    validate_neural_seed_governance_alignment,
)
from iaei.v2.neural_forecasting import (  # noqa: E402
    _aggregate_results,
    _recommendation,
    _window_tensor,
)
from iaei.v2.neural_models import (  # noqa: E402
    build_neural_model,
    configure_deterministic_cpu,
    model_identity,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, object]:
    return validate_neural_seed_governance_alignment(
        validate_neural_forecasting_contract()
    )


def test_frozen_gate_6c2_identity_and_seed_count() -> None:
    contract = _contract()
    assert [item["algorithm_id"] for item in contract["candidate_families"]] == [
        "nhits_compact",
        "tide_compact",
        "patchtst_compact",
    ]
    assert contract["search"]["seeds"] == [
        20260721,
        20260722,
        20260723,
        20260724,
        20260725,
    ]
    assert contract["search"]["seed_count"] == 5
    assert contract["search"]["unique_configuration_count"] == 3


@pytest.mark.parametrize(
    "candidate_index",
    [0, 1, 2],
)
def test_compact_neural_candidates_are_deterministic(candidate_index: int) -> None:
    contract = _contract()
    candidate = contract["candidate_families"][candidate_index]
    values = torch.arange(4 * 96 * 8, dtype=torch.float32).view(4, 96, 8) / 100.0

    configure_deterministic_cpu(20260721)
    first = build_neural_model(
        candidate["algorithm_id"],
        context_length=96,
        input_dimension=8,
        configuration=candidate["configuration"],
    )
    first_prediction = first(values).detach().numpy()

    configure_deterministic_cpu(20260721)
    second = build_neural_model(
        candidate["algorithm_id"],
        context_length=96,
        input_dimension=8,
        configuration=candidate["configuration"],
    )
    second_prediction = second(values).detach().numpy()

    assert first_prediction.shape == (4,)
    assert np.array_equal(first_prediction, second_prediction)
    identity = model_identity(
        candidate["algorithm_id"],
        first,
        context_length=96,
        input_dimension=8,
    )
    assert identity.parameter_count > 0
    assert identity.horizon == 1


def test_compact_models_support_state_dict_roundtrip() -> None:
    candidate = _contract()["candidate_families"][2]
    configure_deterministic_cpu(20260722)
    model = build_neural_model(
        candidate["algorithm_id"],
        context_length=96,
        input_dimension=6,
        configuration=candidate["configuration"],
    )
    values = torch.randn(3, 96, 6)
    expected = model(values).detach().numpy()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    restored = build_neural_model(
        candidate["algorithm_id"],
        context_length=96,
        input_dimension=6,
        configuration=candidate["configuration"],
    )
    state = torch.load(io.BytesIO(buffer.getvalue()), weights_only=True)
    restored.load_state_dict(state)
    observed = restored(values).detach().numpy()
    assert np.array_equal(expected, observed)


def test_window_tensor_uses_only_causal_history() -> None:
    transformed = np.arange(40, dtype=np.float32).reshape(20, 2)
    origins = np.array([4, 8], dtype=np.int64)
    windows = _window_tensor(transformed, origins, context_length=4)
    assert windows.shape == (2, 4, 2)
    assert np.array_equal(windows[0].numpy(), transformed[1:5])
    assert np.array_equal(windows[1].numpy(), transformed[5:9])


def test_candidate_aggregation_preserves_human_decision_boundary() -> None:
    contract = _contract()
    rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for fold_id in range(1, 5):
        reference_rows.append(
            {"fold_id": fold_id, "mae": 5.0, "peak_mae": 10.0, "validation_rows": 1751}
        )
        for candidate_index, candidate in enumerate(contract["candidate_families"]):
            for seed_index, seed in enumerate(contract["search"]["seeds"]):
                rows.append(
                    {
                        "algorithm_id": candidate["algorithm_id"],
                        "seed": seed,
                        "fold_id": fold_id,
                        "validation_rows": 1751,
                        "peak_rows": 100,
                        "mae": 4.8 + candidate_index * 0.2 + seed_index * 0.001,
                        "peak_mae": 9.9 + candidate_index * 0.2,
                        "model_size_bytes": 1000 + candidate_index,
                        "p95_inference_latency_ms_per_1000_rows": 10.0 + candidate_index,
                        "peak_memory_mb": 500.0,
                        "wall_clock_seconds": 1.0,
                        "cpu_portability_passed": True,
                        "maximum_prediction_origin": 28027,
                        "maximum_target_dependency": 28028,
                    }
                )
    seed_results = pd.DataFrame(rows)
    reference = pd.DataFrame(reference_rows)
    outer, leaderboard, evidence = _aggregate_results(seed_results, reference, contract)
    recommendation = _recommendation(leaderboard)

    assert len(outer) == 12
    assert len(leaderboard) == 3
    assert len(evidence) == 3
    assert recommendation["human_decision_required"] is True
    assert recommendation["automatic_promotion_permitted"] is False
    assert recommendation["recommended_next_gate"] == "6C3"


def test_gate_6c2_source_excludes_locked_test_evidence() -> None:
    source = (ROOT / "src" / "iaei" / "v2" / "neural_forecasting.py").read_text(
        encoding="utf-8"
    )
    prohibited = (
        "locked_test_results.json",
        "locked_test_predictions",
        "run_locked_test",
        "evaluate_locked_test",
    )
    assert not any(value in source for value in prohibited)
