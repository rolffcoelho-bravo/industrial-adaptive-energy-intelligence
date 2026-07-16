# Silver analytical layer

## Purpose

The Silver layer converts the immutable UCI steel energy snapshot into a typed, chronologically valid analytical table without altering the source evidence.

It separates data representation from target construction, model fitting, threshold estimation, validation, optimization, and reporting.

## Evidence preservation

All source values are retained under explicit `raw_` column names. Canonical analytical aliases provide stable field names for SQL, DuckDB, Spark, and Python workflows without case-insensitive naming collisions.

The governed effective timestamp applies the validated source convention for operational midnight rows. The raw CSV remains unchanged and unsorted.

## Feature availability

The layer contains:

- current interval measurements available at the prediction origin;
- deterministic origin and next-interval calendar fields;
- usage lags at one, four, and ninety-six intervals;
- trailing usage statistics calculated only from observations before the origin;
- row-level data-quality flags;
- a machine-readable availability-at-origin matrix.

No analytical feature uses a future operating observation.

## Transformation boundary

The Silver layer does not perform supervised target construction, full-sample scaling, centered rolling calculations, backward filling, model-based imputation, outlier deletion, winsorization, model fitting, or threshold estimation.

Unavailable historical features remain null at the beginning of the sample. They are not replaced with invented values.

## Generated evidence

Running `python scripts/build_silver.py` creates the governed Parquet table, quality report, schema record, processing manifest, and feature-availability matrix under `data/processed/`.

The processing manifest records the immutable input hash, generated artifact hashes, row and column counts, quality evidence, DuckDB parity, and Databricks schema compatibility.

## Portability

DuckDB independently verifies row count, column order, timestamps, energy totals, and quality status from the generated Parquet file.

The Databricks notebook applies equivalent uniqueness, chronology, quality, and aggregation checks using Spark.
