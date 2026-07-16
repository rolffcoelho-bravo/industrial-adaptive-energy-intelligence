from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


class SplitContractError(RuntimeError):
    """Raised when chronological evaluation boundaries are invalid."""


@dataclass(frozen=True)
class ChronologicalFold:
    fold_id: int
    train_start: int
    train_stop: int
    purge_start: int
    purge_stop: int
    validation_start: int
    validation_stop: int
    test_purge_start: int
    test_purge_stop: int
    test_start: int
    test_stop: int
    train_start_timestamp: str
    train_end_timestamp: str
    validation_start_timestamp: str
    validation_end_timestamp: str
    test_start_timestamp: str
    test_end_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp_at(timestamps: pd.Series, position: int) -> str:
    return pd.Timestamp(timestamps.iloc[position]).isoformat()


def build_expanding_window_folds(
    timestamps: pd.Series,
    contract: dict[str, Any],
) -> list[ChronologicalFold]:
    validation = contract["validation"]
    ordered = pd.Series(
        pd.to_datetime(timestamps, errors="raise")
    ).reset_index(drop=True)

    if ordered.empty:
        raise SplitContractError("Timestamp series is empty")

    if not ordered.is_monotonic_increasing:
        raise SplitContractError("Timestamps are not chronological")

    if ordered.duplicated().any():
        raise SplitContractError("Timestamps contain duplicates")

    train_fraction = float(validation["train_fraction"])
    validation_fraction = float(validation["validation_fraction"])
    test_fraction = float(validation["test_fraction"])

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-12:
        raise SplitContractError("Split fractions do not sum to one")

    row_count = len(ordered)
    validation_start = int(row_count * train_fraction)
    test_start = int(row_count * (train_fraction + validation_fraction))
    test_stop = row_count
    fold_count = int(validation["validation_folds"])
    purge_steps = int(validation["purge_steps"])
    test_boundary_purge_steps = int(
        validation["test_boundary_purge_steps"]
    )

    if fold_count < 1:
        raise SplitContractError("At least one validation fold is required")

    if purge_steps < 0:
        raise SplitContractError("Purge steps cannot be negative")

    if test_boundary_purge_steps < purge_steps:
        raise SplitContractError(
            "Locked-test purge is shorter than the target horizon"
        )

    validation_stop = test_start - test_boundary_purge_steps

    if not 0 < validation_start < validation_stop < test_start < test_stop:
        raise SplitContractError("Chronological split boundaries are invalid")

    validation_rows = validation_stop - validation_start
    boundaries = [
        validation_start + validation_rows * index // fold_count
        for index in range(fold_count + 1)
    ]
    boundaries[-1] = validation_stop

    folds: list[ChronologicalFold] = []

    for fold_id in range(fold_count):
        fold_validation_start = boundaries[fold_id]
        fold_validation_stop = boundaries[fold_id + 1]
        train_stop = fold_validation_start - purge_steps

        if train_stop <= 0:
            raise SplitContractError("Purge removes the complete training set")

        if fold_validation_stop <= fold_validation_start:
            raise SplitContractError("A validation fold is empty")

        folds.append(
            ChronologicalFold(
                fold_id=fold_id + 1,
                train_start=0,
                train_stop=train_stop,
                purge_start=train_stop,
                purge_stop=fold_validation_start,
                validation_start=fold_validation_start,
                validation_stop=fold_validation_stop,
                test_purge_start=validation_stop,
                test_purge_stop=test_start,
                test_start=test_start,
                test_stop=test_stop,
                train_start_timestamp=_timestamp_at(ordered, 0),
                train_end_timestamp=_timestamp_at(ordered, train_stop - 1),
                validation_start_timestamp=_timestamp_at(
                    ordered, fold_validation_start
                ),
                validation_end_timestamp=_timestamp_at(
                    ordered, fold_validation_stop - 1
                ),
                test_start_timestamp=_timestamp_at(ordered, test_start),
                test_end_timestamp=_timestamp_at(ordered, test_stop - 1),
            )
        )

    return folds
