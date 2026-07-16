from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from iaei.contracts import validate_silver_contract
from iaei.data import build_silver_frame, write_silver_artifacts


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "uci_steel_energy" / "Steel_industry_data.csv"


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


@pytest.fixture(scope="module")
def silver_build(raw_frame: pd.DataFrame):
    return build_silver_frame(raw_frame)


def test_locked_silver_contract() -> None:
    contract = validate_silver_contract()

    assert contract["governance"]["decision_gate"] == 3
    assert contract["governance"]["status"] == "locked"
    assert contract["source"]["preserve_raw_columns"] is True
    assert contract["source"]["raw_sorting_allowed"] is False
    assert contract["chronology"]["frequency_minutes"] == 15
    assert contract["publication"]["supervised_targets_excluded"] is True


def test_source_values_are_preserved(
    raw_frame: pd.DataFrame,
    silver_build,
) -> None:
    contract = validate_silver_contract()
    frame = silver_build.frame

    for source_column, alias in contract["canonical_aliases"].items():
        raw_column = f"raw_{alias}"

        assert raw_column in frame.columns
        assert alias in frame.columns

        pd.testing.assert_series_equal(
            frame[raw_column],
            raw_frame[source_column],
            check_dtype=False,
            check_names=False,
        )

        pd.testing.assert_series_equal(
            frame[alias],
            raw_frame[source_column],
            check_dtype=False,
            check_names=False,
        )


def test_columns_are_unique_for_sql_engines(silver_build) -> None:
    columns = list(silver_build.frame.columns)
    casefolded = [column.casefold() for column in columns]

    assert len(columns) == len(set(columns))
    assert len(casefolded) == len(set(casefolded))


def test_effective_timestamps_are_regular(silver_build) -> None:
    timestamps = silver_build.frame["effective_timestamp"]

    assert timestamps.is_monotonic_increasing
    assert not timestamps.duplicated().any()
    assert timestamps.diff().dropna().eq(pd.Timedelta(15, unit="min")).all()


def test_past_only_features_use_prior_observations(
    raw_frame: pd.DataFrame,
    silver_build,
) -> None:
    usage = raw_frame["Usage_kWh"].astype(float)
    history = usage.shift(1)
    frame = silver_build.frame

    pd.testing.assert_series_equal(
        frame["usage_lag_1"],
        usage.shift(1),
        check_names=False,
    )

    pd.testing.assert_series_equal(
        frame["usage_lag_96"],
        usage.shift(96),
        check_names=False,
    )

    pd.testing.assert_series_equal(
        frame["usage_rolling_mean_4"],
        history.rolling(4, min_periods=4).mean(),
        check_names=False,
    )

    pd.testing.assert_series_equal(
        frame["usage_rolling_mean_96"],
        history.rolling(96, min_periods=96).mean(),
        check_names=False,
    )


def test_unavailable_history_remains_null(silver_build) -> None:
    frame = silver_build.frame

    assert pd.isna(frame["usage_lag_1"].iloc[0])
    assert frame["usage_lag_96"].iloc[:96].isna().all()
    assert frame["usage_rolling_mean_4"].iloc[:4].isna().all()
    assert frame["usage_rolling_mean_96"].iloc[:96].isna().all()


def test_future_mutation_does_not_change_earlier_features(
    raw_frame: pd.DataFrame,
) -> None:
    cutoff = 500
    baseline = build_silver_frame(raw_frame).frame
    mutated_raw = raw_frame.copy(deep=True)

    mutated_raw.loc[
        mutated_raw.index > cutoff,
        "Usage_kWh",
    ] = mutated_raw.loc[
        mutated_raw.index > cutoff,
        "Usage_kWh",
    ] * 1000.0

    mutated = build_silver_frame(mutated_raw).frame

    feature_columns = [
        "effective_timestamp",
        "origin_hour",
        "origin_day_of_week",
        "usage_lag_1",
        "usage_lag_4",
        "usage_lag_96",
        "usage_rolling_mean_4",
        "usage_rolling_mean_96",
    ]

    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff, feature_columns],
        mutated.loc[:cutoff, feature_columns],
    )


def test_quality_and_target_boundaries(silver_build) -> None:
    frame = silver_build.frame
    report = silver_build.quality_report

    assert len(frame) == 35040
    assert report["quality_flag_counts"]["dq_any"] == 0
    assert report["expected_interval_fraction"] == pytest.approx(1.0)
    assert report["supervised_targets_present"] is False
    assert "usage_kwh_t_plus_1" not in frame.columns
    assert "peak_within_next_60_minutes" not in frame.columns
    assert not silver_build.availability["uses_future_observation"].any()


def test_artifact_hashes_and_engine_parity(tmp_path: Path) -> None:
    manifest = write_silver_artifacts(ROOT, output_dir=tmp_path)

    parquet_path = tmp_path / "steel_energy_silver.parquet"
    expected_files = [
        parquet_path,
        tmp_path / "steel_energy_quality_report.json",
        tmp_path / "steel_energy_schema.json",
        tmp_path / "steel_energy_processing_manifest.json",
        tmp_path / "steel_energy_feature_availability.csv",
    ]

    for file_path in expected_files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    parquet_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    assert manifest["output"]["parquet_sha256"] == parquet_hash
    assert manifest["output"]["row_count"] == 35040
    assert manifest["parity"]["duckdb"]["passed"] is True
    assert manifest["parity"]["databricks_schema"]["passed"] is True
    assert manifest["quality"]["quality_flag_counts"]["dq_any"] == 0
