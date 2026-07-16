from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from iaei.data import build_silver_frame
from iaei.modeling import SplitContractError, build_expanding_window_folds


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = (
    ROOT
    / "data"
    / "raw"
    / "uci_steel_energy"
    / "Steel_industry_data.csv"
)


@pytest.fixture(scope="module")
def model_contract() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "model_contract.yml").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def timestamps() -> pd.Series:
    raw = pd.read_csv(RAW_CSV)
    silver = build_silver_frame(raw).frame
    return silver["effective_timestamp"]


@pytest.fixture(scope="module")
def folds(timestamps: pd.Series, model_contract: dict):
    return build_expanding_window_folds(timestamps, model_contract)


def test_exact_locked_boundaries(folds) -> None:
    observed = [
        (
            fold.train_start,
            fold.train_stop,
            fold.purge_start,
            fold.purge_stop,
            fold.validation_start,
            fold.validation_stop,
            fold.test_purge_start,
            fold.test_purge_stop,
            fold.test_start,
            fold.test_stop,
        )
        for fold in folds
    ]

    expected = [
        (0, 21020, 21020, 21024, 21024, 22775, 28028, 28032, 28032, 35040),
        (0, 22771, 22771, 22775, 22775, 24526, 28028, 28032, 28032, 35040),
        (0, 24522, 24522, 24526, 24526, 26277, 28028, 28032, 28032, 35040),
        (0, 26273, 26273, 26277, 26277, 28028, 28028, 28032, 28032, 35040),
    ]

    assert observed == expected


def test_training_window_expands_across_origins(folds) -> None:
    train_stops = [fold.train_stop for fold in folds]

    assert train_stops == sorted(train_stops)
    assert len(set(train_stops)) == len(train_stops)


def test_purge_boundary_is_four_intervals(folds) -> None:
    for fold in folds:
        assert fold.purge_stop - fold.purge_start == 4
        assert fold.train_stop == fold.purge_start
        assert fold.purge_stop == fold.validation_start


def test_validation_windows_do_not_overlap(folds) -> None:
    for current, following in zip(folds, folds[1:]):
        assert current.validation_stop == following.validation_start


def test_locked_test_has_a_four_step_boundary_purge(folds) -> None:
    for fold in folds:
        assert fold.test_purge_start == 28028
        assert fold.test_purge_stop == 28032
        assert fold.test_purge_stop - fold.test_purge_start == 4
        assert fold.validation_stop <= fold.test_purge_start

def test_locked_test_is_identical_across_folds(folds) -> None:
    test_boundaries = {
        (fold.test_start, fold.test_stop)
        for fold in folds
    }

    assert test_boundaries == {(28032, 35040)}


def test_validation_never_enters_locked_test(folds) -> None:
    for fold in folds:
        assert fold.validation_stop <= fold.test_start
        assert fold.train_stop < fold.validation_start


def test_timestamp_metadata_matches_boundaries(
    folds,
    timestamps: pd.Series,
) -> None:
    for fold in folds:
        assert fold.train_start_timestamp == pd.Timestamp(
            timestamps.iloc[fold.train_start]
        ).isoformat()
        assert fold.train_end_timestamp == pd.Timestamp(
            timestamps.iloc[fold.train_stop - 1]
        ).isoformat()
        assert fold.validation_start_timestamp == pd.Timestamp(
            timestamps.iloc[fold.validation_start]
        ).isoformat()
        assert fold.test_end_timestamp == pd.Timestamp(
            timestamps.iloc[fold.test_stop - 1]
        ).isoformat()


def test_nonchronological_timestamps_are_rejected(
    timestamps: pd.Series,
    model_contract: dict,
) -> None:
    altered = timestamps.copy()
    altered.iloc[10], altered.iloc[11] = altered.iloc[11], altered.iloc[10]

    with pytest.raises(SplitContractError, match="not chronological"):
        build_expanding_window_folds(altered, model_contract)


def test_duplicate_timestamps_are_rejected(
    timestamps: pd.Series,
    model_contract: dict,
) -> None:
    altered = timestamps.copy()
    altered.iloc[11] = altered.iloc[10]

    with pytest.raises(SplitContractError, match="duplicates"):
        build_expanding_window_folds(altered, model_contract)
