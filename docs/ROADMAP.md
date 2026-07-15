# Execution roadmap

## Gate 0 - Repository foundation

**Status:** established.

Acceptance criteria:

- company-neutral public identity;
- real-data-only rule;
- code/data/report licensing separated;
- CI and publication contracts present;
- final PDF path and exact page count frozen;
- no result placeholders accepted by the report builder.

## Gate 1 - Data intake and immutable manifest

- Download UCI dataset 851.
- Verify schema, row count, timestamps, duplicates, missingness, and non-negative usage.
- Record source URL, DOI, timestamps, hashes, license, and file sizes.
- Decide whether the licensed raw snapshot is committed or attached to a release.

## Gate 2 - Target and leakage contract

- Freeze the forecast horizon.
- Freeze the peak definition using training data only.
- Freeze permitted information at each prediction origin.
- Add explicit leakage tests.

## Gate 3 - Bronze/Silver/Gold implementation

- Local path: pandas/DuckDB/Parquet.
- Databricks path: notebooks, SQL, Delta-oriented tables, workflow evidence.
- Ensure both paths produce equivalent analytical tables.

## Gate 4 - Model ladder and chronological validation

- Baselines, linear/ridge, tree ensembles, calibrated classifiers.
- Expanding or rolling origins.
- Locked final test block.
- Average, peak-state, worst-window, calibration, and stability metrics.

## Gate 5 - Structural-drift governance

- Frozen champion.
- Local challenger estimated only from available information.
- Normalized feature, residual, forecast-disagreement, and operator/parameter drift.
- Stable/watch/adaptation-candidate states.
- Strict positive worst-window promotion rule.

## Gate 6 - Constrained optimization

- Transparent objective.
- Explicit flexibility and disruption constraints.
- No automatic operational action.
- Uncertainty and no-action state.

## Gate 7 - Governed GenAI/multi-agent layer

- Deterministic reporting remains the mandatory fallback.
- Optional local/open-source LLM or no-cost Databricks experiment.
- Data Quality, Model Risk, and Operations Intelligence agents.
- Supervisor routing, source traceability, and human approval.

## Gate 8 - Five-page publication build

- All report inputs final and contract-valid.
- Four required charts and four required tables.
- Exactly five pages.
- Rendered PDF visually inspected.
- GitHub Actions build artifact and tagged release.

## Gate 9 - Public release candidate

- Green CI.
- One-command reproduction.
- No secrets or proprietary claims.
- Clean README and architecture.
- Real benchmark results.
- Public release with PDF and checksums.
