# Portable enterprise execution stack

| Layer | Tool |
|---|---|
| Version control and review | GitHub repository |
| CI | Standard GitHub-hosted Actions runner |
| Local analytical engine | Python, pandas, scikit-learn, DuckDB |
| Columnar storage | Parquet and PyArrow |
| Charts | Matplotlib |
| PDF | ReportLab |
| Enterprise data and ML execution | Databricks notebooks, governed SQL, workflows, and experiment tracking |
| SQL evidence | DuckDB locally and Databricks SQL in workspace |
| Agent fallback | Deterministic Python summaries |
| Optional local LLM | Open-source model isolated from the mandatory pipeline |

The mandatory execution path remains fully reproducible with open-source components and does not depend on paid APIs, proprietary LLM endpoints, or platform-specific enterprise services.

## Execution architecture

Databricks provides the enterprise execution layer for modular notebooks, governed SQL transformations, workflow orchestration, and traceable machine-learning experimentation. A parallel open-source execution path preserves portability, auditability, and full reproducibility beyond any single platform.
