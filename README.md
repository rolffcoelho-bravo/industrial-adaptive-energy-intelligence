# Industrial Adaptive Energy Intelligence

[![CI](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/rolffcoelho-bravo/industrial-adaptive-energy-intelligence/actions/workflows/ci.yml)

A public, reproducible proof of work for industrial data science: energy-demand forecasting, peak-risk classification, constrained optimization, structural-drift governance, and decision-grade reporting.

> **Independence statement**  
> This repository is an independent technical demonstration. It is not affiliated with, endorsed by, or based on proprietary data from any industrial company.

## Decision question

Can a governed analytical system forecast near-term industrial electricity demand, identify peak-load risk, detect a material change in the predictive mechanism, and recommend feasible actions without automatically destabilizing the production model?

## What the finished pipeline proves

| Capability | Evidence |
|---|---|
| Python | Features, modelling, monitoring, optimization, reporting |
| SQL | Quality controls, analytical tables, KPI calculation, query review |
| Databricks | Importable notebooks and workflow-oriented stage design |
| Regression | Next-period energy-demand forecast |
| Classification | Calibrated peak-load probability |
| Ensembles | Controlled combination of validated candidate models |
| Optimization | Constrained load-adjustment recommendation |
| Pipelines | Contract-driven Bronze/Silver/Gold progression |
| Business impact | Assumption-bounded operational and financial scenarios |
| GenAI and agents | Optional governed narrative and multi-agent interfaces |
| Model monitoring | Champion/challenger structural-drift gate |
| Visualization | Five decision-grade Matplotlib figures with publication gates |

## Public data

The project uses the UCI **Steel Industry Energy Consumption** dataset (dataset ID 851), collected from a South Korean steel company. The raw snapshot is downloaded by script, hashed, and recorded in a manifest. Dataset licensing and attribution are documented in [`data/DATA_LICENSE.md`](data/DATA_LICENSE.md).

No synthetic operating data or invented performance results are permitted.

## Architecture

```text
UCI real data
    |
    v
Bronze: immutable source + manifest
    |
    v
Silver: validated chronology + leakage-safe features
    |
    +--> regression models
    +--> peak-risk classifiers
    +--> statistical diagnostics
              |
              v
       chronological validation
              |
              v
     structural-drift decision gate
              |
              v
       constrained optimization
              |
              v
Gold: forecasts, risk, drift, decisions, impact
              |
              v
 optional governed agent summaries
              |
              v
exactly five-page technical brief (PDF)
```

## Two execution paths

1. **Local/GitHub Actions:** pandas, scikit-learn, DuckDB, Matplotlib, ReportLab. This is the reproducibility path and must always work without Databricks.
2. **Databricks Free Edition:** notebooks, SQL, workflow orchestration, and ML experiment evidence. This is the vacancy-alignment path, but the public project never depends exclusively on a proprietary platform.

## Strict report gate

The PDF generator reads only validated files under `outputs/`. Each page contains one dominant, publication-quality Matplotlib figure governed by [`configs/visualization_contract.yml`](configs/visualization_contract.yml) and [`docs/VISUALIZATION_STANDARD.md`](docs/VISUALIZATION_STANDARD.md). It refuses to publish the final brief when:

- required metrics are missing;
- a result is marked provisional;
- a chart is absent;
- the dataset manifest is missing;
- leakage checks fail;
- model promotion has no recorded decision;
- a business-impact number lacks assumptions.

The final artifact is always:

```text
outputs/brief/industrial_adaptive_energy_intelligence_technical_brief.pdf
```

and must contain exactly five pages.

## Repository status

**Foundation / Decision Gate 0.** Contracts, CI, report specification, public-repository policy, and execution structure are established. Modelling results are not yet claimed.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/DECISION_GATES.md`](docs/DECISION_GATES.md).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_pipeline.py --check
pytest
```

After all analytical stages are implemented and validated:

```bash
python scripts/run_pipeline.py --all
python scripts/build_brief.py
```

## Public-release rule

Do not invite reviewers to an empty or broken repository. Make the repository public only when the first end-to-end baseline passes CI, generates real tables and charts, and produces the five-page brief without placeholders.

## License

- Source code: MIT License.
- Dataset: CC BY 4.0, attributed separately.
- Generated analysis and brief: Ã‚Â© Rodolfo Pereira, released for portfolio review unless a later release states otherwise.
