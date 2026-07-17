# Technical brief evidence specification

## Page 1: Confirmatory forecasting verdict

The opening page reports the frozen selected model, candidate and persistence MAE, aggregate relative improvement, peak-state relative improvement, locked prediction count, and the closed confirmatory decision.

The page must not show forecast intervals, peak probabilities, operating states, or recommendations that are absent from the terminal evidence.

## Page 2: Governed data and analytical architecture

This page presents official dataset provenance, immutable raw identity, interval frequency, source-aware timestamps, Silver analytical-layer dimensions, data-quality results, and the separation between Databricks execution and the portable reproducibility baseline.

## Page 3: Model ladder and chronological validation

This page presents the persistence and naive references, Ridge, Elastic Net, and histogram gradient boosting candidates, four chronological validation folds, training-only parameter selection, promotion decisions, and the validation-only model freeze.

Classification diagnostics remain separate benchmark evidence. They must not be described as probability outputs from the selected regression model.

## Page 4: Locked-test stability and peak-state robustness

This page presents candidate and persistence MAE for each prespecified temporal block, relative improvement in every block, the training-derived peak threshold, peak-state row count, aggregate and peak-state performance, and exact evaluation boundaries.

All values must be read from the immutable Gate 4E result and Gate 4F closure artifacts.

## Page 5: Evidence governance and model boundaries

This page presents the evidence lineage from benchmark validation through confirmatory closure, the frozen execution commit, terminal artifact hashes, single evaluation count, prohibition of re-estimation, CI exclusion of the evaluator, and explicit model boundaries.

## Integrity controls

- No unsupported company-specific conclusion.
- No causal savings claim.
- No structural-drift result without machine-readable evidence.
- No optimization recommendation without a validated optimization artifact.
- No live-production claim.
- Every displayed value reconciles with a final machine-readable source.
- The locked evaluator is never invoked by the reporting build.
- Final rendering is visually inspected at PDF size.
- No appendix is used to hide unresolved evidence.

## Visual structure

Each page contains one dominant institutional Matplotlib figure and one compact evidence table. Multiple panels are permitted only when they form one coherent evidence chain. Decorative complexity is prohibited.
