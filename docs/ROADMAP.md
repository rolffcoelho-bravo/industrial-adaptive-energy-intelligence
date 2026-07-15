# Implementation roadmap

## Gate 0: Repository foundation

**Status:** Implemented.

Evidence:

- reproducible Python package;
- CI quality controls;
- separated code and data licensing;
- real-data-only rule;
- schema and reporting contracts;
- placeholder rejection.

## Gate 1: Data intake and immutable manifest

**Status:** Implemented.

Evidence:

- official UCI dataset retrieval;
- immutable committed snapshot;
- SHA-256 verification;
- schema, row-count, missingness, duplicate, and non-negativity checks;
- source-aware timestamp treatment;
- offline verification path.

## Gate 2: Target and leakage contract

**Status:** Implemented.

Evidence:

- next-interval regression target;
- next-hour peak-risk target;
- training-only peak threshold;
- chronological ownership rules;
- unavailable boundary treatment;
- executable leakage and chronology tests.

## Gate 3: Silver analytical layer

Planned evidence:

- typed Parquet table;
- governed effective timestamp;
- source columns preserved;
- data-quality flags;
- deterministic calendar fields;
- input and output hashes;
- DuckDB and Databricks parity;
- feature availability-at-origin matrix;
- future-mutation causality tests.

## Gate 4: Model ladder and chronological validation

Planned evidence:

- naive and seasonal baselines;
- linear and ridge models;
- tree-based candidates;
- calibrated peak-risk classifiers;
- expanding or rolling origins;
- locked final test block;
- average, peak-state, worst-window, calibration, and stability metrics.

## Gate 5: Structural-drift governance

Planned evidence:

- frozen champion;
- locally estimated challenger using available information only;
- normalized feature, residual, disagreement, and parameter drift;
- stable, watch, and adaptation-candidate states;
- strict positive worst-window promotion rule.

## Gate 6: Constrained optimization

Planned evidence:

- transparent objective;
- explicit flexibility limits;
- disruption and feasibility constraints;
- uncertainty-aware no-action state;
- sensitivity analysis.

## Gate 7: Governed agent layer

Planned evidence:

- deterministic reporting fallback;
- Data Quality, Model Risk, and Operations Intelligence roles;
- restricted tool permissions;
- source traceability;
- human authorization for consequential actions.

## Gate 8: Technical brief and reproducible release

Planned evidence:

- final machine-readable tables;
- validated decision figures;
- visually inspected PDF;
- one-command reproduction;
- checksums and tagged release;
- no secrets, proprietary claims, placeholders, or fabricated results.
