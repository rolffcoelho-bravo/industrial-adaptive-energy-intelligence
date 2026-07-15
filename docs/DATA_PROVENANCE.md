# Data provenance and immutable snapshot policy

The project uses the **Steel Industry Energy Consumption** dataset from the official UCI Machine Learning Repository, dataset 851, DOI `10.24432/C52G8C`. The dataset is licensed under **CC BY 4.0**.

## Dual reproducibility route

1. `scripts/download_data.py --refresh` retrieves the official UCI archive, validates the source bytes in a temporary staging area, computes SHA-256 hashes, and atomically publishes the governed snapshot and manifest.
2. The resulting CSV is committed under `data/raw/uci_steel_energy/` so the project remains reproducible when the external service is unavailable.
3. `scripts/download_data.py --verify` performs a network-free audit of the committed CSV, manifest, checksum file, source metadata, schema, and timestamp convention.

## Source timestamp convention

The source CSV contains 365 operational-day blocks with 96 observations per block. Within each block, rows are ordered from `00:15` through `23:45`, followed by `00:00`.

The final `00:00` row represents the interval ending at the next calendar-day boundary. This is a source convention, not a data-quality failure. The raw CSV remains byte-for-byte unchanged and is never sorted.

Chronological validation constructs an **effective timestamp in memory** by adding one calendar day only where `NSM = 0`. The resulting effective sequence is continuous at 15-minute intervals from `2018-01-01 00:15` through `2019-01-01 00:00`.

## Raw-layer restrictions

No cleaning, sorting, renaming, recoding, imputation, feature engineering, or row deletion is permitted in `data/raw/uci_steel_energy/`. Transformations begin only in governed in-memory checks and the Silver layer.

## Failure behavior

The downloader validates the candidate snapshot before replacing any repository artifact. The manifest is published last. If validation fails, no new snapshot is declared valid and no partial manifest is created.

## Independence

The dataset comes from an independent South Korean steel producer. It is not proprietary to any organization associated with this project.
