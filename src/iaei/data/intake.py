from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_ID = "uci-851"
DATASET_DOI = "10.24432/C52G8C"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CITATION = (
    "V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption "
    "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C."
)
ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/851/"
    "steel%2Bindustry%2Benergy%2Bconsumption.zip"
)
EXPECTED_ROWS = 35040
EXPECTED_COLUMNS = [
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
]
SOURCE_DATE_FORMAT = "%d/%m/%Y %H:%M"
EXPECTED_FREQUENCY_MINUTES = 15
EXPECTED_OBSERVATIONS_PER_DAY = 96
EXPECTED_SOURCE_DAYS = 365
EXPECTED_MIDNIGHT_ROWS = 365
TIMESTAMP_CONVENTION = (
    "The source preserves 96-row operational-day blocks ordered from 00:15 through "
    "23:45, followed by a 00:00 row that represents the interval ending at the next "
    "calendar-day boundary. Raw row order is preserved; chronological validation uses "
    "an effective timestamp that adds one day only to NSM=0 rows."
)


class DataIntakeError(RuntimeError):
    """Raised when the immutable raw-data contract is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_effective_timestamps(
    frame: pd.DataFrame, *, enforce_full_contract: bool = True
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Parse source timestamps without mutating the raw file.

    The UCI source stores each operational day as 00:15..23:45 and then 00:00. The
    final 00:00 observation belongs to the next interval boundary. We preserve the
    source bytes and row order, while creating an in-memory effective timestamp for
    chronological validation and downstream Silver-layer construction.
    """

    source_timestamps = pd.to_datetime(
        frame["date"],
        format=SOURCE_DATE_FORMAT,
        errors="raise",
    )
    nsm = pd.to_numeric(frame["NSM"], errors="raise").astype("int64")

    if (nsm < 0).any() or (nsm >= 86400).any():
        raise DataIntakeError("NSM values must be within [0, 86400)")
    if (nsm.mod(EXPECTED_FREQUENCY_MINUTES * 60) != 0).any():
        raise DataIntakeError("NSM values are not aligned to 15-minute boundaries")

    literal_seconds = (
        source_timestamps.dt.hour.mul(3600)
        + source_timestamps.dt.minute.mul(60)
        + source_timestamps.dt.second
    ).astype("int64")
    nsm_mismatches = int((literal_seconds != nsm).sum())
    if nsm_mismatches != 0:
        raise DataIntakeError(f"NSM timestamp consistency failures: {nsm_mismatches}")

    midnight_mask = nsm.eq(0)
    effective_timestamps = source_timestamps + pd.to_timedelta(
        midnight_mask.astype("int64"), unit="D"
    )

    source_dates = source_timestamps.dt.normalize()
    daily_counts = frame.groupby(source_dates, sort=False).size()
    expected_block = list(range(900, 86400, 900)) + [0]
    source_block_order_matches = True

    if enforce_full_contract:
        if len(daily_counts) != EXPECTED_SOURCE_DAYS:
            raise DataIntakeError(
                f"Expected {EXPECTED_SOURCE_DAYS} source-day blocks; observed {len(daily_counts)}"
            )
        if not daily_counts.eq(EXPECTED_OBSERVATIONS_PER_DAY).all():
            raise DataIntakeError(
                "Each source-day block must contain exactly "
                f"{EXPECTED_OBSERVATIONS_PER_DAY} observations"
            )

        for _, group in frame.assign(_source_date=source_dates).groupby(
            "_source_date", sort=False
        ):
            observed_block = (
                pd.to_numeric(group["NSM"], errors="raise").astype(int).tolist()
            )
            if observed_block != expected_block:
                source_block_order_matches = False
                break
        if not source_block_order_matches:
            raise DataIntakeError(
                "Source operational-day ordering differs from the locked "
                "00:15..23:45,00:00 convention"
            )

    source_duplicate_timestamps = int(source_timestamps.duplicated().sum())
    effective_duplicate_timestamps = int(effective_timestamps.duplicated().sum())
    if source_duplicate_timestamps != 0:
        raise DataIntakeError(
            f"Duplicate source timestamps detected: {source_duplicate_timestamps}"
        )
    if effective_duplicate_timestamps != 0:
        raise DataIntakeError(
            f"Duplicate effective timestamps detected: {effective_duplicate_timestamps}"
        )
    if not effective_timestamps.is_monotonic_increasing:
        raise DataIntakeError("Effective timestamps are not monotonic increasing")

    interval_minutes = effective_timestamps.diff().dropna().dt.total_seconds().div(60)
    expected_interval_fraction = float(
        interval_minutes.eq(EXPECTED_FREQUENCY_MINUTES).mean()
    )
    if expected_interval_fraction < 0.999999:
        raise DataIntakeError(
            "Unexpected effective timestamp interval structure: "
            f"15-minute fraction={expected_interval_fraction:.6f}"
        )

    midnight_rows = int(midnight_mask.sum())
    if enforce_full_contract and midnight_rows != EXPECTED_MIDNIGHT_ROWS:
        raise DataIntakeError(
            f"Expected {EXPECTED_MIDNIGHT_ROWS} end-of-day midnight rows; "
            f"observed {midnight_rows}"
        )

    audit = {
        "date_format": SOURCE_DATE_FORMAT,
        "timestamp_convention": TIMESTAMP_CONVENTION,
        "source_day_blocks": int(len(daily_counts)),
        "observations_per_source_day": EXPECTED_OBSERVATIONS_PER_DAY,
        "source_midnight_rows": midnight_rows,
        "source_block_order_matches": source_block_order_matches,
        "raw_timestamp_order_monotonic": bool(source_timestamps.is_monotonic_increasing),
        "effective_timestamp_order_monotonic": bool(
            effective_timestamps.is_monotonic_increasing
        ),
        "source_duplicate_timestamps": source_duplicate_timestamps,
        "effective_duplicate_timestamps": effective_duplicate_timestamps,
        "expected_frequency_minutes": EXPECTED_FREQUENCY_MINUTES,
        "expected_interval_fraction": expected_interval_fraction,
        "nsm_mismatches": nsm_mismatches,
        "source_sample_start": source_timestamps.iloc[0].isoformat(),
        "source_sample_end": source_timestamps.iloc[-1].isoformat(),
        "sample_start": effective_timestamps.iloc[0].isoformat(),
        "sample_end": effective_timestamps.iloc[-1].isoformat(),
    }
    return source_timestamps, effective_timestamps, audit


def inspect_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        raise DataIntakeError(f"Raw snapshot not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise DataIntakeError(
            "Raw schema differs from contract. "
            f"Expected {EXPECTED_COLUMNS}; observed {list(frame.columns)}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise DataIntakeError(f"Expected {EXPECTED_ROWS} rows; observed {len(frame)}")

    missing_total = int(frame.isna().sum().sum())
    if missing_total != 0:
        raise DataIntakeError(f"Expected no missing values; observed {missing_total}")
    if (frame["Usage_kWh"] < 0).any():
        raise DataIntakeError("Negative energy usage detected")

    _, _, timestamp_audit = build_effective_timestamps(frame)

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "file_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": sha256(csv_path),
        "missing_values_total": missing_total,
        "duplicate_timestamps": timestamp_audit["effective_duplicate_timestamps"],
        **timestamp_audit,
    }


def build_snapshot_manifest(
    csv_path: Path,
    root: Path,
    downloaded_at_utc: str,
    archive_sha256: str,
) -> dict[str, Any]:
    audit = inspect_csv(csv_path)
    return {
        "dataset_id": DATASET_ID,
        "dataset_name": "Steel Industry Energy Consumption",
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "source_url": ARCHIVE_URL,
        "source_status": "official",
        "data_status": "immutable_raw_snapshot",
        "downloaded_at_utc": downloaded_at_utc,
        "archive_sha256": archive_sha256,
        "csv_path": csv_path.relative_to(root).as_posix(),
        **audit,
    }


def validate_snapshot(csv_path: Path, manifest_path: Path, root: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise DataIntakeError(f"Snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = inspect_csv(csv_path)

    audited_keys = [
        "row_count",
        "column_count",
        "file_size_bytes",
        "csv_sha256",
        "missing_values_total",
        "duplicate_timestamps",
        "date_format",
        "timestamp_convention",
        "source_day_blocks",
        "observations_per_source_day",
        "source_midnight_rows",
        "source_block_order_matches",
        "raw_timestamp_order_monotonic",
        "effective_timestamp_order_monotonic",
        "source_duplicate_timestamps",
        "effective_duplicate_timestamps",
        "expected_frequency_minutes",
        "expected_interval_fraction",
        "nsm_mismatches",
        "source_sample_start",
        "source_sample_end",
        "sample_start",
        "sample_end",
    ]
    expected_pairs: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "csv_path": csv_path.relative_to(root).as_posix(),
    }
    expected_pairs.update({key: audit[key] for key in audited_keys})

    mismatches = {
        key: {"manifest": manifest.get(key), "observed": observed}
        for key, observed in expected_pairs.items()
        if manifest.get(key) != observed
    }
    if mismatches:
        raise DataIntakeError(f"Snapshot/manifest mismatch: {mismatches}")
    return manifest
