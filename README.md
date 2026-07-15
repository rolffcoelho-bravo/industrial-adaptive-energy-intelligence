# Industrial Adaptive Energy Intelligence

[![CI](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml)

A governed industrial energy intelligence system that converts interval electricity measurements into near-term demand forecasts, peak-risk estimates, structural-drift signals, constrained operating recommendations, and traceable technical reporting.

> **Independence**
>
> This is an independent open-source research and engineering project. It is not affiliated with, endorsed by, or based on proprietary data from any industrial company.

## Decision question

Can an industrial analytical system forecast next-interval electricity demand, identify peak risk within the next hour, detect material changes in the predictive mechanism, and recommend feasible actions while preserving explicit model-risk controls?

## Analytical system

| Layer | Technical role |
|---|---|
| Data engineering | Immutable source intake, schema validation, chronology repair, and Bronze/Silver/Gold progression |
| Forecasting | Next-interval electricity-demand estimation with chronological validation |
| Peak-risk classification | Calibrated probability of a high-load interval within the next hour |
| Structural monitoring | Champion/challenger drift evidence and controlled adaptation states |
| Optimization | Feasible load-adjustment recommendations under cost, peak, and disruption constraints |
| Reporting | Machine-readable tables, decision-grade figures, and a governed technical brief |
| Enterprise execution | Databricks notebooks, SQL transformations, workflow orchestration, and traceable experiments |
| Portable execution | Python, DuckDB, Parquet, scikit-learn, Matplotlib, and GitHub Actions |

## Data and provenance

The project uses the UCI **Steel Industry Energy Consumption** dataset, dataset ID 851, collected from a South Korean steel producer.

The raw snapshot is:

- retrieved from the official source;
- preserved byte-for-byte;
- verified by SHA-256;
- recorded in a machine-readable manifest;
- attributed under CC BY 4.0;
- auditable without network access.

Dataset licensing is documented in [`data/DATA_LICENSE.md`](data/DATA_LICENSE.md). Provenance and timestamp treatment are documented in [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

No synthetic operating data or invented performance results are permitted.

## Architecture

```text
UCI licensed data
    |
    v
Bronze: immutable source, checksum, manifest
    |
    v
Silver: validated chronology, typed analytical table, leakage-safe features
    |
    +--> regression models
    +--> peak-risk classifiers
    +--> statistical diagnostics
              |
              v
       chronological validation
              |
              v
      structural-drift controls
              |
              v
       constrained optimization
              |
              v
Gold: forecasts, risk, drift, decisions, impact
              |
              v
      governed technical brief
```

## Locked target contract

Decision Gate 2 defines two operational targets:

| Target | Definition |
|---|---|
| `usage_kwh_t_plus_1` | `Usage_kWh` at the next 15-minute interval |
| `peak_within_next_60_minutes` | Whether any of the next four intervals reaches the training-only 90th-percentile threshold |

The threshold is estimated only from the applicable training partition. Random splitting, full-sample preprocessing, centered windows, future-value filling, and locked-test influence are prohibited.

See [`docs/TARGET_AND_LEAKAGE_CONTRACT.md`](docs/TARGET_AND_LEAKAGE_CONTRACT.md).

## Execution paths

### Local and GitHub Actions

The portable path uses pandas, scikit-learn, DuckDB, PyArrow, Matplotlib, and ReportLab. It remains the reproducibility baseline.

### Databricks

Databricks provides the enterprise execution layer for governed transformations, modular notebooks, workflow orchestration, SQL evidence, and traceable machine-learning experimentation. The portable path preserves auditability and avoids dependence on a single platform.

## Evidence integrity

The pipeline blocks reporting when required evidence is missing or invalid. Controls include:

- immutable data hashes;
- explicit schemas and contracts;
- chronological validation;
- leakage tests;
- recorded model-promotion decisions;
- assumption-bounded impact calculations;
- real figures generated from final machine-readable outputs;
- rejection of placeholder or provisional results.

## Current implementation

Implemented:

- repository contracts and CI;
- governed UCI data intake;
- immutable cross-platform snapshot;
- source-aware effective timestamps;
- target and leakage contract;
- training-only peak-threshold construction;
- automated chronology, boundary, and leakage tests.

Under implementation:

- Silver analytical tables and feature availability controls;
- chronological model ladder;
- structural-drift scoring;
- constrained optimization;
- final evidence tables, figures, and technical brief.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/DECISION_GATES.md`](docs/DECISION_GATES.md).

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS and Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/download_data.py --verify
python scripts/run_pipeline.py --check
python scripts/audit_public_content.py
pytest
```

When all analytical stages are implemented and validated:

```bash
python scripts/run_pipeline.py --all
python scripts/build_brief.py
```

The generated report is written to:

```text
outputs/brief/industrial_adaptive_energy_intelligence_technical_brief.pdf
```

## License and attribution

- Source code: MIT License.
- Dataset: CC BY 4.0, attributed separately.

Analysis, engineering, and technical brief by Rodolfo Pereira. Source code and data use are governed by the licenses stated above.
