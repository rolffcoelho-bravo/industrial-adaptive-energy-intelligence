# Industrial Adaptive Energy Intelligence

[![CI](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml)

A governed industrial energy forecasting and validation system that converts interval electricity measurements into leakage-controlled demand forecasts, peak-state diagnostics, chronological model evidence, and traceable technical reporting.

> **Independence**
>
> This is an independent open-source research and engineering project. It is not affiliated with, endorsed by, or based on proprietary data from any industrial company.

## Decision question

Can a governed industrial analytical system forecast next-interval electricity demand, quantify peak-state performance, and preserve chronological, leakage, and single-evaluation controls from model selection through public reporting?

## Analytical system

| Layer | Technical role |
|---|---|
| Data engineering | Immutable source intake, schema validation, chronology repair, and governed Bronze and Silver progression |
| Forecasting | Next-interval electricity-demand estimation with leakage-controlled features |
| Peak-state diagnostics | Training-derived peak labels and separate classification benchmark evidence |
| Model selection | Prespecified candidate ladder with chronological validation and training-only parameter selection |
| Confirmatory evaluation | One frozen model, one untouched test period, and one terminal evaluation |
| Reporting | Machine-readable tables, decision-grade figures, and a governed five-page technical brief |
| Enterprise execution | Databricks notebooks, SQL transformations, workflow orchestration, and traceable experiments |
| Portable execution | Python, DuckDB, Parquet, scikit-learn, Matplotlib, ReportLab, and GitHub Actions |

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
    +--> naive and persistence references
    +--> Ridge and Elastic Net candidates
    +--> histogram gradient boosting candidate
    +--> separate peak-classification benchmarks
              |
              v
       chronological validation
              |
              v
      validation-only model selection
              |
              v
          frozen model contract
              |
              v
      single locked-test evaluation
              |
              v
closure manifest, figures, tables, technical brief
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

- immutable data and terminal-result hashes;
- explicit schemas and contracts;
- chronological validation;
- leakage tests;
- recorded model-promotion decisions;
- frozen model identity before test access;
- a single authorized locked-test evaluation;
- permanent closure of the confirmatory gate;
- figures generated only from final machine-readable outputs;
- rejection of placeholder, provisional, or unsupported results.

## Confirmatory result

The frozen histogram gradient boosting model was evaluated once on 7,004 untouched prediction origins. Candidate MAE was **3.9435**, compared with **5.4445** for persistence, a relative improvement of **27.57%**.

During the 761 training-defined peak-state origins, candidate MAE was **14.3012**, compared with **18.4671** for persistence, a relative improvement of **22.56%**. Improvement remained positive in all four prespecified temporal blocks, ranging from **22.00%** to **31.40%**.

The confirmatory evaluation is closed. Re-estimation, threshold changes, alternative test slicing, and a second evaluation are prohibited.

Machine-readable evidence:

- [`locked_test_results.json`](outputs/modeling/locked_test_results.json)
- [`locked_test_closure_manifest.json`](outputs/modeling/locked_test_closure_manifest.json)

## Decision visualizations

Gate 5A converts the completed model and confirmatory evidence into five decision-grade Matplotlib figures. Each figure is tied to a specific forecasting, validation, data, or governance question.

| Figure | Decision value |
|---|---|
| Confirmatory forecasting verdict | Shows candidate-versus-persistence performance, peak-state robustness, and the closed test decision. |
| Governed data architecture | Connects provenance, chronology, Silver construction, data quality, and reproducible execution. |
| Model ladder and chronological validation | Shows benchmark evidence, candidate promotion decisions, folds, and validation-only selection. |
| Locked-test temporal stability | Shows all four prespecified blocks, peak-state results, and exact test boundaries. |
| Evidence governance and model boundaries | Shows gate lineage, immutable artifact identity, single-evaluation controls, and excluded claims. |

Every figure must reconcile with final machine-readable evidence. Units, sample dates, sources, thresholds, and model status are mandatory. Placeholder curves, decorative dashboards, unsupported causal claims, optimization recommendations, and savings claims are prohibited.

## Current implementation

Completed and governed:

- repository contracts and CI;
- governed UCI data intake and immutable cross-platform snapshot;
- source-aware effective timestamps;
- Silver analytical layer and feature-availability controls;
- target, boundary, and leakage contracts;
- persistence and naive benchmark evidence;
- Ridge, Elastic Net, and histogram gradient boosting validation;
- validation-only model selection and model freeze;
- single-use locked-test execution;
- immutable confirmatory closure with all four temporal blocks positive;
- evidence-aligned Gate 5A reporting contracts.

Approved Gate 5A build stage:

- final synthesis tables;
- five evidence-aligned figures;
- validated report payload;
- five-page technical brief.

Future decision gates remain separate and unclaimed: structural-drift scoring, constrained optimization, governed agents, and assumption-bounded business-impact analysis.

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

After the Gate 5A synthesis tables, figures, and report payload are generated and validated:

```bash
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
