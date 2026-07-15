# UCI Steel Industry Energy Consumption snapshot

This directory contains an immutable byte-for-byte CSV snapshot downloaded from the official UCI Machine Learning Repository.

- Dataset ID: `uci-851`
- DOI: `10.24432/C52G8C`
- License: `CC BY 4.0`
- Expected observations: `35040`
- Observed columns: `11`
- SHA-256: `9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`

## Timestamp convention

The source preserves 96-row operational-day blocks ordered from 00:15 through 23:45, followed by a 00:00 row that represents the interval ending at the next calendar-day boundary. Raw row order is preserved; chronological validation uses an effective timestamp that adds one day only to NSM=0 rows.

The raw CSV is never sorted or rewritten. Chronological timestamps are constructed only in memory and in downstream governed layers.

## Scope

The snapshot is company-neutral and contains no proprietary data from a target organization.
Cleaning, feature engineering, and imputation are prohibited in this raw directory.

## Attribution

V E, S., Shin, C., & Cho, Y. (2021). Steel Industry Energy Consumption [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52G8C.
