# Databricks notebooks

The notebook sequence will mirror the local package rather than contain disconnected exploratory code.

Planned import order:

1. `00_environment_and_contracts.py`
2. `01_bronze_ingestion.py`
3. `02_silver_validation.py`
4. `03_feature_engineering.py`
5. `04_regression_models.py`
6. `05_peak_classification.py`
7. `06_chronological_validation.py`
8. `07_drift_gate.py`
9. `08_constrained_optimization.py`
10. `09_gold_reporting.py`
11. `10_governed_agents.py`
12. `11_build_technical_brief.py`

Every notebook must be exportable as source, callable by a workflow, and reproducible locally through the corresponding `src/iaei/` module.
