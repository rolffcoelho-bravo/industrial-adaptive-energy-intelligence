# Databricks notebook source
# ruff: noqa: F821

from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text(
    "silver_path",
    "data/processed/steel_energy_silver.parquet",
)

silver_path = dbutils.widgets.get("silver_path")
silver = spark.read.parquet(silver_path)

# COMMAND ----------

required_columns = {
    "effective_timestamp",
    "usage_kwh",
    "usage_lag_1",
    "usage_lag_4",
    "usage_lag_96",
    "usage_rolling_mean_4",
    "usage_rolling_mean_96",
    "dq_any",
}

missing_columns = sorted(required_columns.difference(silver.columns))

if missing_columns:
    raise ValueError(
        f"Silver table is missing required columns: {missing_columns}"
    )

# COMMAND ----------

summary = silver.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("effective_timestamp").alias(
        "unique_effective_timestamps"
    ),
    F.min("effective_timestamp").alias("sample_start"),
    F.max("effective_timestamp").alias("sample_end"),
    F.sum(F.col("dq_any").cast("integer")).alias(
        "failed_quality_rows"
    ),
)

result = summary.collect()[0]

if result["row_count"] != result["unique_effective_timestamps"]:
    raise ValueError("Effective timestamps are not unique")

if result["failed_quality_rows"] != 0:
    raise ValueError("Silver table contains failed quality rows")

display(summary)

# COMMAND ----------

display(
    silver.groupBy(
        "origin_year",
        "origin_month",
        "load_type",
    )
    .agg(
        F.count("*").alias("observations"),
        F.avg("usage_kwh").alias("average_usage_kwh"),
        F.max("usage_kwh").alias("maximum_usage_kwh"),
    )
    .orderBy(
        "origin_year",
        "origin_month",
        "load_type",
    )
)
