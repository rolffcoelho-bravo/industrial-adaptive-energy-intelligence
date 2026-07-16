from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from iaei.contracts import validate_silver_contract
from iaei.data.intake import (
    EXPECTED_COLUMNS,
    build_effective_timestamps,
    sha256,
    validate_snapshot,
)


class SilverLayerError(RuntimeError):
    """Raised when the governed Silver analytical layer is invalid."""


@dataclass(frozen=True)
class SilverBuild:
    frame: pd.DataFrame
    quality_report: dict[str, Any]
    availability: pd.DataFrame


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return bool(
            np.array_equal(
                left.to_numpy(),
                right.to_numpy(),
                equal_nan=True,
            )
        )
    return left.astype("string").equals(right.astype("string"))


def _availability_matrix(
    contract: dict[str, Any],
    source_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "feature_name": "effective_timestamp",
            "feature_group": "identifier",
            "available_at_origin": True,
            "uses_current_observation": True,
            "uses_future_observation": False,
            "transformation_fit_required": False,
            "lookback_steps": 0,
            "nullable_at_sample_start": False,
        }
    ]

    aliases = contract["canonical_aliases"]

    for source_column in source_columns:
        rows.append(
            {
                "feature_name": source_column,
                "feature_group": "preserved_source",
                "available_at_origin": True,
                "uses_current_observation": True,
                "uses_future_observation": False,
                "transformation_fit_required": False,
                "lookback_steps": 0,
                "nullable_at_sample_start": False,
            }
        )

    for alias in aliases.values():
        rows.append(
            {
                "feature_name": alias,
                "feature_group": "canonical_current",
                "available_at_origin": True,
                "uses_current_observation": True,
                "uses_future_observation": False,
                "transformation_fit_required": False,
                "lookback_steps": 0,
                "nullable_at_sample_start": False,
            }
        )

    for field in contract["calendar_fields"]:
        rows.append(
            {
                "feature_name": field,
                "feature_group": "deterministic_calendar",
                "available_at_origin": True,
                "uses_current_observation": False,
                "uses_future_observation": False,
                "transformation_fit_required": False,
                "lookback_steps": 0,
                "nullable_at_sample_start": False,
            }
        )

    for feature in contract["past_only_features"]:
        rows.append(
            {
                "feature_name": feature["name"],
                "feature_group": "past_only_usage",
                "available_at_origin": True,
                "uses_current_observation": False,
                "uses_future_observation": False,
                "transformation_fit_required": False,
                "lookback_steps": int(feature["lookback_steps"]),
                "nullable_at_sample_start": True,
            }
        )

    for flag in contract["quality_flags"]:
        rows.append(
            {
                "feature_name": flag,
                "feature_group": "quality_control",
                "available_at_origin": True,
                "uses_current_observation": True,
                "uses_future_observation": False,
                "transformation_fit_required": False,
                "lookback_steps": 0,
                "nullable_at_sample_start": False,
            }
        )

    availability = pd.DataFrame(rows).drop_duplicates(
        subset=["feature_name"],
        keep="first",
    )

    return availability.reset_index(drop=True)


def build_silver_frame(
    raw: pd.DataFrame,
    *,
    contract: dict[str, Any] | None = None,
    enforce_full_contract: bool = True,
) -> SilverBuild:
    locked = contract or validate_silver_contract()

    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise SilverLayerError(f"Required source columns are missing: {missing}")

    if list(raw.columns) != EXPECTED_COLUMNS:
        raise SilverLayerError("Source column order differs from the raw-data contract")

    source_timestamp, effective_timestamp, timestamp_audit = (
        build_effective_timestamps(
            raw,
            enforce_full_contract=enforce_full_contract,
        )
    )

    aliases: dict[str, str] = locked["canonical_aliases"]
    source_column_map = {
        source_column: f"raw_{alias}"
        for source_column, alias in aliases.items()
    }

    silver = raw.rename(columns=source_column_map).copy(deep=True)
    silver.insert(0, "source_row_number", np.arange(len(silver), dtype=np.int64))
    silver.insert(1, "source_timestamp", source_timestamp)
    silver.insert(2, "effective_timestamp", effective_timestamp)

    for source_column, alias in aliases.items():
        silver[alias] = raw[source_column]

    numeric_aliases = [
        "usage_kwh",
        "lagging_reactive_power_kvarh",
        "leading_reactive_power_kvarh",
        "co2_tco2",
        "lagging_power_factor",
        "leading_power_factor",
    ]

    for column in numeric_aliases:
        silver[column] = pd.to_numeric(silver[column], errors="raise").astype(float)

    silver["nsm_seconds"] = pd.to_numeric(
        silver["nsm_seconds"],
        errors="raise",
    ).astype("int64")

    for column in [
        "source_timestamp_text",
        "week_status",
        "day_name",
        "load_type",
    ]:
        silver[column] = silver[column].astype("string")

    origin = silver["effective_timestamp"]

    silver["origin_year"] = origin.dt.year.astype("int16")
    silver["origin_quarter"] = origin.dt.quarter.astype("int8")
    silver["origin_month"] = origin.dt.month.astype("int8")
    silver["origin_day"] = origin.dt.day.astype("int8")
    silver["origin_day_of_week"] = origin.dt.dayofweek.astype("int8")
    silver["origin_hour"] = origin.dt.hour.astype("int8")
    silver["origin_minute"] = origin.dt.minute.astype("int8")

    minute_of_day = origin.dt.hour.mul(60).add(origin.dt.minute)
    silver["origin_interval_of_day"] = (
        minute_of_day.div(15).astype("int16")
    )
    silver["origin_is_weekend"] = origin.dt.dayofweek.ge(5)

    silver["origin_time_sin"] = np.sin(
        2.0 * math.pi * minute_of_day / 1440.0
    )
    silver["origin_time_cos"] = np.cos(
        2.0 * math.pi * minute_of_day / 1440.0
    )
    silver["origin_day_of_week_sin"] = np.sin(
        2.0 * math.pi * origin.dt.dayofweek / 7.0
    )
    silver["origin_day_of_week_cos"] = np.cos(
        2.0 * math.pi * origin.dt.dayofweek / 7.0
    )

    target_timestamp = origin.add(pd.Timedelta(15, unit="min"))
    silver["target_timestamp_t_plus_1"] = target_timestamp
    silver["target_hour_t_plus_1"] = target_timestamp.dt.hour.astype("int8")
    silver["target_day_of_week_t_plus_1"] = (
        target_timestamp.dt.dayofweek.astype("int8")
    )
    silver["target_is_weekend_t_plus_1"] = (
        target_timestamp.dt.dayofweek.ge(5)
    )

    usage = silver["usage_kwh"]
    history = usage.shift(1)

    silver["usage_lag_1"] = usage.shift(1)
    silver["usage_lag_4"] = usage.shift(4)
    silver["usage_lag_96"] = usage.shift(96)

    rolling_4 = history.rolling(window=4, min_periods=4)
    silver["usage_rolling_mean_4"] = rolling_4.mean()
    silver["usage_rolling_std_4"] = rolling_4.std(ddof=0)

    rolling_96 = history.rolling(window=96, min_periods=96)
    silver["usage_rolling_mean_96"] = rolling_96.mean()
    silver["usage_rolling_std_96"] = rolling_96.std(ddof=0)
    silver["usage_rolling_min_96"] = rolling_96.min()
    silver["usage_rolling_max_96"] = rolling_96.max()

    literal_seconds = (
        source_timestamp.dt.hour.mul(3600)
        + source_timestamp.dt.minute.mul(60)
        + source_timestamp.dt.second
    )

    interval_gap = effective_timestamp.diff().ne(pd.Timedelta(15, unit="min"))
    interval_gap.iloc[0] = False

    silver["dq_missing_source_value"] = raw.isna().any(axis=1)
    silver["dq_negative_usage"] = silver["usage_kwh"].lt(0)
    silver["dq_nsm_timestamp_mismatch"] = literal_seconds.ne(
        silver["nsm_seconds"]
    )
    silver["dq_non_15_minute_gap"] = interval_gap
    silver["dq_duplicate_effective_timestamp"] = (
        effective_timestamp.duplicated(keep=False)
    )

    component_flags = [
        "dq_missing_source_value",
        "dq_negative_usage",
        "dq_nsm_timestamp_mismatch",
        "dq_non_15_minute_gap",
        "dq_duplicate_effective_timestamp",
    ]

    silver["dq_any"] = silver[component_flags].any(axis=1)

    casefolded_names = [column.casefold() for column in silver.columns]
    if len(casefolded_names) != len(set(casefolded_names)):
        raise SilverLayerError(
            "Silver columns are not unique under case-insensitive SQL rules"
        )

    for source_column, alias in aliases.items():
        if not _series_values_equal(silver[alias], raw[source_column]):
            raise SilverLayerError(
                f"Canonical alias differs from source values: {alias}"
            )

    if not effective_timestamp.is_monotonic_increasing:
        raise SilverLayerError("Silver effective timestamps are not ordered")

    if effective_timestamp.duplicated().any():
        raise SilverLayerError("Silver effective timestamps contain duplicates")

    prohibited_targets = {
        "usage_kwh_t_plus_1",
        "peak_within_next_60_minutes",
    }

    target_hits = sorted(prohibited_targets.intersection(silver.columns))
    if target_hits:
        raise SilverLayerError(
            f"Supervised targets entered the Silver table: {target_hits}"
        )

    if silver["dq_any"].any():
        counts = {
            column: int(silver[column].sum())
            for column in component_flags
        }
        raise SilverLayerError(f"Silver quality flags failed: {counts}")

    availability = _availability_matrix(
        locked,
        source_columns=list(source_column_map.values()),
    )

    unavailable_features = availability.loc[
        availability["uses_future_observation"],
        "feature_name",
    ].tolist()

    if unavailable_features:
        raise SilverLayerError(
            f"Future-dependent features are prohibited: {unavailable_features}"
        )

    history_columns = [
        feature["name"]
        for feature in locked["past_only_features"]
    ]

    quality_report = {
        "contract_version": locked["contract_version"],
        "row_count": int(len(silver)),
        "column_count": int(len(silver.columns)),
        "source_columns_preserved": source_column_map,
        "source_column_count": int(len(raw.columns)),
        "effective_sample_start": effective_timestamp.iloc[0].isoformat(),
        "effective_sample_end": effective_timestamp.iloc[-1].isoformat(),
        "expected_frequency_minutes": int(
            locked["chronology"]["frequency_minutes"]
        ),
        "expected_interval_fraction": float(
            timestamp_audit["expected_interval_fraction"]
        ),
        "quality_flag_counts": {
            column: int(silver[column].sum())
            for column in component_flags + ["dq_any"]
        },
        "history_feature_null_counts": {
            column: int(silver[column].isna().sum())
            for column in history_columns
        },
        "supervised_targets_present": False,
        "source_order_preserved": True,
    }

    return SilverBuild(
        frame=silver,
        quality_report=quality_report,
        availability=availability,
    )


def _spark_type(field: pa.Field) -> str:
    data_type = field.type

    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return "string"
    if pa.types.is_boolean(data_type):
        return "boolean"
    if pa.types.is_int8(data_type):
        return "byte"
    if pa.types.is_int16(data_type):
        return "short"
    if pa.types.is_int32(data_type):
        return "integer"
    if pa.types.is_int64(data_type):
        return "long"
    if pa.types.is_float32(data_type):
        return "float"
    if pa.types.is_float64(data_type):
        return "double"
    if pa.types.is_timestamp(data_type):
        return "timestamp"

    return "unsupported"


def _schema_payload(parquet_path: Path) -> dict[str, Any]:
    arrow_schema = pq.read_schema(parquet_path)

    fields = [
        {
            "name": field.name,
            "arrow_type": str(field.type),
            "spark_type": _spark_type(field),
            "nullable": bool(field.nullable),
        }
        for field in arrow_schema
    ]

    unsupported = [
        field["name"]
        for field in fields
        if field["spark_type"] == "unsupported"
    ]

    return {
        "format": "parquet",
        "column_count": len(fields),
        "fields": fields,
        "databricks_schema_compatible": not unsupported,
        "unsupported_columns": unsupported,
    }


def _duckdb_parity(
    parquet_path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    escaped = str(parquet_path.resolve()).replace("'", "''")

    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            "CREATE VIEW silver AS "
            f"SELECT * FROM read_parquet('{escaped}')"
        )

        row_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM silver"
            ).fetchone()[0]
        )

        minimum_timestamp, maximum_timestamp = connection.execute(
            "SELECT MIN(effective_timestamp), MAX(effective_timestamp) "
            "FROM silver"
        ).fetchone()

        usage_sum = float(
            connection.execute(
                "SELECT SUM(usage_kwh) FROM silver"
            ).fetchone()[0]
        )

        dq_count = int(
            connection.execute(
                "SELECT SUM(CASE WHEN dq_any THEN 1 ELSE 0 END) FROM silver"
            ).fetchone()[0]
        )

        duckdb_columns = connection.execute(
            "DESCRIBE silver"
        ).fetchdf()["column_name"].tolist()

    expected_usage_sum = float(frame["usage_kwh"].sum())

    passed = (
        row_count == len(frame)
        and duckdb_columns == list(frame.columns)
        and pd.Timestamp(minimum_timestamp)
        == pd.Timestamp(frame["effective_timestamp"].min())
        and pd.Timestamp(maximum_timestamp)
        == pd.Timestamp(frame["effective_timestamp"].max())
        and math.isclose(
            usage_sum,
            expected_usage_sum,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and dq_count == int(frame["dq_any"].sum())
    )

    if not passed:
        raise SilverLayerError("DuckDB parity validation failed")

    return {
        "passed": True,
        "row_count": row_count,
        "column_order_matches": True,
        "effective_sample_start": pd.Timestamp(
            minimum_timestamp
        ).isoformat(),
        "effective_sample_end": pd.Timestamp(
            maximum_timestamp
        ).isoformat(),
        "usage_sum": usage_sum,
        "dq_any_count": dq_count,
    }


def write_silver_artifacts(
    root: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    contract = validate_silver_contract()

    raw_csv = root / contract["source"]["raw_csv_path"]
    raw_manifest_path = root / contract["source"]["raw_manifest_path"]

    raw_manifest = validate_snapshot(
        raw_csv,
        raw_manifest_path,
        root,
    )

    raw = pd.read_csv(raw_csv)
    build = build_silver_frame(raw, contract=contract)

    if output_dir is None:
        output_paths = {
            name: root / relative
            for name, relative in contract["outputs"].items()
            if name.endswith("_path")
        }
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {
            "parquet_path": output_dir / "steel_energy_silver.parquet",
            "quality_report_path": (
                output_dir / "steel_energy_quality_report.json"
            ),
            "schema_path": output_dir / "steel_energy_schema.json",
            "processing_manifest_path": (
                output_dir / "steel_energy_processing_manifest.json"
            ),
            "feature_availability_path": (
                output_dir / "steel_energy_feature_availability.csv"
            ),
        }

    parquet_path = output_paths["parquet_path"]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_parquet = parquet_path.with_name(
        f"{parquet_path.name}.tmp"
    )

    build.frame.to_parquet(
        temporary_parquet,
        engine="pyarrow",
        compression=contract["outputs"]["parquet_compression"],
        index=False,
    )
    os.replace(temporary_parquet, parquet_path)

    _atomic_json(
        output_paths["quality_report_path"],
        build.quality_report,
    )

    _atomic_csv(
        output_paths["feature_availability_path"],
        build.availability,
    )

    schema_payload = _schema_payload(parquet_path)

    if not schema_payload["databricks_schema_compatible"]:
        raise SilverLayerError(
            "Parquet schema contains Databricks-incompatible columns"
        )

    _atomic_json(
        output_paths["schema_path"],
        schema_payload,
    )

    duckdb_report = _duckdb_parity(
        parquet_path,
        build.frame,
    )

    artifact_hashes = {
        "parquet_sha256": sha256(parquet_path),
        "quality_report_sha256": sha256(
            output_paths["quality_report_path"]
        ),
        "schema_sha256": sha256(output_paths["schema_path"]),
        "feature_availability_sha256": sha256(
            output_paths["feature_availability_path"]
        ),
    }

    manifest = {
        "dataset_id": raw_manifest["dataset_id"],
        "silver_contract_version": contract["contract_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "raw_csv_path": contract["source"]["raw_csv_path"],
            "raw_csv_sha256": raw_manifest["csv_sha256"],
            "raw_row_count": raw_manifest["row_count"],
        },
        "output": {
            "parquet_path": str(parquet_path),
            "row_count": int(len(build.frame)),
            "column_count": int(len(build.frame.columns)),
            **artifact_hashes,
        },
        "quality": build.quality_report,
        "parity": {
            "duckdb": duckdb_report,
            "databricks_schema": {
                "passed": True,
                "unsupported_columns": [],
            },
        },
        "publication_controls": {
            "atomic_artifact_writes": True,
            "processing_manifest_written_last": True,
            "supervised_targets_excluded": True,
        },
    }

    _atomic_json(
        output_paths["processing_manifest_path"],
        manifest,
    )

    return manifest
