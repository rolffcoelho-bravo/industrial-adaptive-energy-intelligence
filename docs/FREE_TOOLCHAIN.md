# Free toolchain

| Layer | Tool |
|---|---|
| Version control and review | Public GitHub repository |
| CI | Standard GitHub-hosted Actions runner |
| Local analytical engine | Python, pandas, scikit-learn, DuckDB |
| Columnar storage | Parquet / PyArrow |
| Charts | Matplotlib |
| PDF | ReportLab |
| Industrial notebook evidence | Databricks Free Edition |
| SQL evidence | DuckDB locally and Databricks SQL in workspace |
| Agent fallback | Deterministic Python summaries |
| Optional local LLM | Open-source model, isolated from mandatory pipeline |

The mandatory pipeline cannot require a credit card, paid API key, proprietary LLM endpoint, or enterprise Databricks feature.
