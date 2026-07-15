from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iaei.data.intake import (
    build_effective_timestamps,
    inspect_csv,
    validate_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "uci_steel_energy" / "Steel_industry_data.csv"
MANIFEST_PATH = ROOT / "data" / "manifests" / "uci_steel_energy_manifest.json"


def test_source_midnight_convention_is_normalized_without_raw_mutation() -> None:
    frame = pd.DataFrame(
        {
            "date": [
                "01/01/2018 23:45",
                "01/01/2018 00:00",
                "02/01/2018 00:15",
            ],
            "NSM": [85500, 0, 900],
        }
    )
    source, effective, audit = build_effective_timestamps(
        frame, enforce_full_contract=False
    )
    assert source.iloc[1].isoformat() == "2018-01-01T00:00:00"
    assert effective.iloc[1].isoformat() == "2018-01-02T00:00:00"
    assert effective.iloc[2].isoformat() == "2018-01-02T00:15:00"
    assert audit["nsm_mismatches"] == 0


def test_committed_snapshot_passes_intake_contract() -> None:
    audit = inspect_csv(CSV_PATH)
    assert audit["row_count"] == 35040
    assert audit["column_count"] == 11
    assert audit["missing_values_total"] == 0
    assert audit["duplicate_timestamps"] == 0
    assert audit["nsm_mismatches"] == 0
    assert audit["source_day_blocks"] == 365
    assert audit["source_midnight_rows"] == 365
    assert audit["source_block_order_matches"] is True
    assert audit["raw_timestamp_order_monotonic"] is False
    assert audit["effective_timestamp_order_monotonic"] is True
    assert audit["expected_interval_fraction"] == 1.0
    assert audit["sample_start"] == "2018-01-01T00:15:00"
    assert audit["sample_end"] == "2019-01-01T00:00:00"


def test_manifest_matches_committed_snapshot() -> None:
    manifest = validate_snapshot(CSV_PATH, MANIFEST_PATH, ROOT)
    serialized = json.dumps(manifest).lower()
    assert manifest["data_status"] == "immutable_raw_snapshot"
    assert manifest["effective_timestamp_order_monotonic"] is True
    assert manifest["raw_timestamp_order_monotonic"] is False
    assert "synthetic" not in serialized
    assert "placeholder" not in serialized
