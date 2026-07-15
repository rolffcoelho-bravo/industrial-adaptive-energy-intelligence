from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iaei.contracts import ContractError, validate_target_contract
from iaei.targets import (
    TargetConstructionError,
    build_supervised_targets,
    fit_peak_threshold,
    training_mask_from_end,
)

ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2018-01-01 00:15:00",
        periods=10,
        freq="15min",
    )
    return pd.DataFrame(
        {
            "effective_timestamp": timestamps,
            "Usage_kWh": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        }
    )


def test_locked_target_contract_passes_schema_validation() -> None:
    contract = validate_target_contract()

    assert contract["governance"]["decision_gate"] == 2
    assert contract["governance"]["status"] == "locked"
    assert contract["targets"]["regression"]["horizon_minutes"] == 15
    assert contract["targets"]["peak_risk"]["horizon_minutes"] == 60
    assert contract["targets"]["peak_risk"]["threshold"]["quantile"] == 0.90


def test_regression_target_is_exactly_next_observed_usage() -> None:
    frame = _frame()
    training_mask = frame.index < 6

    artifacts = build_supervised_targets(frame, training_mask)
    target = artifacts.frame["usage_kwh_t_plus_1"]

    expected = frame["Usage_kWh"].shift(-1).astype(float)
    pd.testing.assert_series_equal(target, expected, check_names=False)
    assert pd.isna(target.iloc[-1])


def test_peak_label_uses_only_the_next_four_intervals() -> None:
    frame = _frame()
    training_mask = frame.index < 6

    artifacts = build_supervised_targets(frame, training_mask)
    threshold = artifacts.peak_threshold_kwh
    labels = artifacts.frame["peak_within_next_60_minutes"]

    expected_first = int(frame.loc[1:4, "Usage_kWh"].ge(threshold).any())
    assert labels.iloc[0] == expected_first
    assert labels.iloc[-4:].isna().all()


def test_peak_threshold_is_fit_on_training_partition_only() -> None:
    frame = _frame()
    training_mask = frame.index < 6

    baseline = build_supervised_targets(frame, training_mask).peak_threshold_kwh

    changed_future = frame.copy()
    changed_future.loc[6:, "Usage_kWh"] = [2000, 3000, 4000, 5000]
    changed = build_supervised_targets(
        changed_future,
        training_mask,
    ).peak_threshold_kwh

    assert baseline == changed
    expected = frame.loc[training_mask, "Usage_kWh"].quantile(
        0.90,
        interpolation="linear",
    )
    assert baseline == pytest.approx(expected)


def test_threshold_changes_when_training_observations_change() -> None:
    usage = pd.Series([10, 20, 30, 40, 50], dtype=float)
    training_mask = pd.Series([True, True, True, False, False])

    baseline = fit_peak_threshold(usage, training_mask)

    changed_training = usage.copy()
    changed_training.iloc[2] = 300
    changed = fit_peak_threshold(changed_training, training_mask)

    assert baseline != changed


def test_target_timestamps_are_strictly_after_prediction_origins() -> None:
    frame = _frame()
    training_mask = frame.index < 6

    targets = build_supervised_targets(frame, training_mask).frame

    regression_valid = targets["usage_kwh_t_plus_1_timestamp"].notna()
    assert (
        targets.loc[regression_valid, "usage_kwh_t_plus_1_timestamp"]
        > targets.loc[regression_valid, "prediction_origin"]
    ).all()

    peak_valid = targets["peak_within_next_60_minutes_window_end"].notna()
    assert (
        targets.loc[peak_valid, "peak_within_next_60_minutes_window_end"]
        > targets.loc[peak_valid, "prediction_origin"]
    ).all()


def test_training_mask_is_constructed_from_an_explicit_time_cutoff() -> None:
    frame = _frame()

    mask = training_mask_from_end(
        frame,
        "2018-01-01 01:30:00",
    )

    assert mask.tolist() == [True, True, True, True, True, True, False, False, False, False]


def test_incomplete_future_windows_are_unavailable_not_imputed() -> None:
    frame = _frame()
    training_mask = frame.index < 6

    targets = build_supervised_targets(frame, training_mask).frame

    assert pd.isna(targets["usage_kwh_t_plus_1"].iloc[-1])
    assert targets["peak_within_next_60_minutes"].iloc[-4:].isna().all()
    assert targets["peak_within_next_60_minutes_window_end"].iloc[-4:].isna().all()


def test_unsorted_prediction_origins_are_rejected() -> None:
    frame = _frame().iloc[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]].reset_index(drop=True)
    training_mask = frame.index < 6

    with pytest.raises(TargetConstructionError, match="chronologically ordered"):
        build_supervised_targets(frame, training_mask)


def test_contract_forbids_random_splits_and_full_sample_preprocessing() -> None:
    contract = validate_target_contract()
    forbidden = set(contract["leakage_policy"]["forbidden_operations"])

    assert contract["validation"]["random_split_forbidden"] is True
    assert "random_train_test_split" in forbidden
    assert "centered_rolling_window" in forbidden
    assert "full_sample_scaling" in forbidden
    assert "full_sample_encoding" in forbidden
    assert "full_sample_peak_threshold" in forbidden


def test_invalid_contract_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_contract = ROOT / "configs" / "target_contract.yml"
    invalid_schema = ROOT / "schemas" / "target_contract.schema.json"

    target_config = tmp_path / "configs"
    target_schema = tmp_path / "schemas"
    target_config.mkdir()
    target_schema.mkdir()

    contract_text = invalid_contract.read_text(encoding="utf-8").replace(
        "quantile: 0.90",
        "quantile: 0.95",
    )
    (target_config / "target_contract.yml").write_text(
        contract_text,
        encoding="utf-8",
    )
    (target_schema / "target_contract.schema.json").write_text(
        invalid_schema.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    import iaei.contracts as contracts

    monkeypatch.setattr(contracts, "CONFIGS", target_config)
    monkeypatch.setattr(contracts, "SCHEMAS", target_schema)

    with pytest.raises(ContractError, match="Target and leakage contract"):
        contracts.validate_target_contract()
