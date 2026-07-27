from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class FoundationAdapterError(RuntimeError):
    """Raised when a pinned foundation-model adapter violates its interface."""


def _quantile_indices(levels: list[float], required: tuple[float, ...]) -> list[int]:
    indices: list[int] = []
    values = np.asarray(levels, dtype=float)
    for level in required:
        matches = np.flatnonzero(np.isclose(values, level, rtol=0.0, atol=1e-8))
        if len(matches) != 1:
            raise FoundationAdapterError(
                f"Required quantile {level} is not uniquely available: {levels}"
            )
        indices.append(int(matches[0]))
    return indices


def _timed_chunks(
    windows: np.ndarray,
    *,
    batch_size: int,
    infer_chunk: Callable[[np.ndarray], np.ndarray],
    maximum_seconds: float,
) -> tuple[np.ndarray, list[float]]:
    outputs: list[np.ndarray] = []
    latency_per_1000_rows_ms: list[float] = []
    started_total = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        if time.perf_counter() - started_total > maximum_seconds:
            raise FoundationAdapterError("Candidate inference exceeded its wall-clock limit")
        chunk = windows[start : start + batch_size]
        started = time.perf_counter()
        prediction = np.asarray(infer_chunk(chunk), dtype=float)
        elapsed = time.perf_counter() - started
        if prediction.shape != (len(chunk), 3):
            raise FoundationAdapterError(
                f"Adapter returned {prediction.shape}; expected {(len(chunk), 3)}"
            )
        if not np.isfinite(prediction).all():
            raise FoundationAdapterError("Adapter returned nonfinite quantiles")
        outputs.append(prediction)
        latency_per_1000_rows_ms.append(elapsed * 1_000_000.0 / len(chunk))
    if not outputs:
        raise FoundationAdapterError("Adapter received no causal windows")
    return np.concatenate(outputs, axis=0), latency_per_1000_rows_ms


def _chronos_predictor(model_directory: Path, context_length: int) -> Callable[[np.ndarray], np.ndarray]:
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_directory),
        device_map="cpu",
    )
    indices = _quantile_indices(list(pipeline.quantiles), (0.1, 0.5, 0.9))

    def infer(chunk: np.ndarray) -> np.ndarray:
        inputs = np.asarray(chunk[:, None, :], dtype=np.float32)
        forecasts = pipeline.predict(
            inputs=inputs,
            prediction_length=1,
            batch_size=len(chunk),
            context_length=context_length,
            cross_learning=False,
            limit_prediction_length=True,
        )
        rows: list[np.ndarray] = []
        for forecast in forecasts:
            values = forecast.detach().cpu().numpy()
            if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 1:
                raise FoundationAdapterError(
                    f"Chronos-2 returned an unexpected tensor shape: {values.shape}"
                )
            rows.append(values[0, indices, 0])
        return np.asarray(rows, dtype=float)

    return infer


def _timesfm_predictor(model_directory: Path, context_length: int, batch_size: int) -> Callable[[np.ndarray], np.ndarray]:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        str(model_directory),
        local_files_only=True,
        torch_compile=False,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_length,
            max_horizon=1,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    def infer(chunk: np.ndarray) -> np.ndarray:
        _, quantiles = model.forecast(
            horizon=1,
            inputs=[np.asarray(row, dtype=np.float32) for row in chunk],
        )
        values = np.asarray(quantiles, dtype=float)
        if values.shape != (len(chunk), 1, 10):
            raise FoundationAdapterError(
                f"TimesFM 2.5 returned an unexpected array shape: {values.shape}"
            )
        return values[:, 0, [1, 5, 9]]

    return infer


def _moirai_predictor(model_directory: Path, context_length: int) -> Callable[[np.ndarray], np.ndarray]:
    import torch
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(
        str(model_directory),
        local_files_only=True,
    )
    model = Moirai2Forecast(
        module=module,
        prediction_length=1,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to("cpu")
    model.eval()
    levels = [float(value) for value in module.quantile_levels]
    indices = _quantile_indices(levels, (0.1, 0.5, 0.9))

    def infer(chunk: np.ndarray) -> np.ndarray:
        past_target = torch.from_numpy(
            np.asarray(chunk[:, :, None], dtype=np.float32)
        )
        observed = torch.ones_like(past_target, dtype=torch.bool)
        padding = torch.zeros(
            (len(chunk), context_length),
            dtype=torch.bool,
        )
        with torch.no_grad():
            output = model(
                past_target=past_target,
                past_observed_target=observed,
                past_is_pad=padding,
            )
        values = output.detach().cpu().numpy()
        if values.ndim != 3 or values.shape[0] != len(chunk) or values.shape[2] != 1:
            raise FoundationAdapterError(
                f"Moirai 2.0 returned an unexpected tensor shape: {values.shape}"
            )
        return values[:, indices, 0]

    return infer


def build_predictor(
    candidate_id: str,
    *,
    model_directory: Path,
    context_length: int,
    batch_size: int,
) -> Callable[[np.ndarray], np.ndarray]:
    if candidate_id == "chronos_2_zero_shot":
        return _chronos_predictor(model_directory, context_length)
    if candidate_id == "timesfm_2_5_zero_shot":
        return _timesfm_predictor(model_directory, context_length, batch_size)
    if candidate_id == "moirai_2_research_zero_shot":
        return _moirai_predictor(model_directory, context_length)
    raise FoundationAdapterError(f"Unknown foundation-model candidate: {candidate_id}")


def execute_batched_inference(
    candidate_id: str,
    *,
    model_directory: Path,
    windows: np.ndarray,
    context_length: int,
    batch_size: int,
    maximum_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictor = build_predictor(
        candidate_id,
        model_directory=model_directory,
        context_length=context_length,
        batch_size=batch_size,
    )
    prediction, latency = _timed_chunks(
        windows,
        batch_size=batch_size,
        infer_chunk=predictor,
        maximum_seconds=maximum_seconds,
    )
    replay_rows = min(8, len(windows))
    replay = np.asarray(predictor(windows[:replay_rows]), dtype=float)
    deterministic_replay_passed = bool(
        np.allclose(
            prediction[:replay_rows],
            replay,
            rtol=0.0,
            atol=1e-6,
            equal_nan=False,
        )
    )
    return prediction, {
        "p50_inference_latency_ms_per_1000_rows": float(np.percentile(latency, 50)),
        "p95_inference_latency_ms_per_1000_rows": float(np.percentile(latency, 95)),
        "deterministic_replay_passed": deterministic_replay_passed,
        "batch_count": len(latency),
        "batch_size": batch_size,
    }
