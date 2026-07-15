# Validation gates

The project advances through explicit evidence contracts. Each stage must pass its technical controls before dependent stages are enabled.

| Gate | Technical decision | Status | Required evidence |
|---:|---|---|---|
| 0 | Repository identity and contracts | Implemented | CI, policies, schemas |
| 1 | Data source and immutable snapshot | Implemented | Manifest, hash, license, data-quality checks |
| 2 | Forecast target and peak definition | Implemented | Target schema, leakage controls, executable tests |
| 3 | Silver table and feature universe | Pending | Typed analytical table, availability-at-origin matrix, feature-causality tests |
| 4 | Validation and model ladder | Pending | Chronological backtest contract and benchmark evidence |
| 5 | Drift score and promotion rule | Pending | Multi-origin robustness evidence |
| 6 | Optimization assumptions | Pending | Feasibility checks and sensitivity analysis |
| 7 | Agent boundaries | Pending | Tool permissions, evidence traceability, human controls |
| 8 | Technical brief and release artifacts | Pending | Validated PDF, figures, tables, checksums, and visual QA |

## Decision Gate 2 boundary

Decision Gate 2 locks:

- `usage_kwh_t_plus_1` as the 15-minute regression target;
- `peak_within_next_60_minutes` as the primary risk target;
- a training-only 90th-percentile peak threshold;
- chronological validation and a locked final test block;
- prohibited leakage operations;
- unavailable treatment for incomplete forward windows.

The contract is defined in `configs/target_contract.yml`, validated by `schemas/target_contract.schema.json`, and explained in `docs/TARGET_AND_LEAKAGE_CONTRACT.md`.

The repository reports failed controls directly. It does not manufacture substitute observations, targets, metrics, or performance evidence.
