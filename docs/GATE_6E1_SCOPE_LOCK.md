# Gate 6E1 uncertainty contract scope lock

## Decision

Gate 6E1 is closed as an implementation-only uncertainty contract. It freezes the admissible Gate 6E design before any calibration or interval execution.

The retained point model is `v1_frozen_champion`, the frozen histogram-gradient-boosting model. Gate 6E does not reopen advanced tabular, neural, or foundation-model selection.

## Frozen method set

1. **Expanding absolute-residual conformal**: reference configuration `expanding_all`.
2. **Rolling absolute-residual conformal**: configurations `rolling_672` and `rolling_2688`.
3. **Adaptive conformal inference**: six configurations combining windows `672` and `2,688` with update rates `0.005`, `0.01`, and `0.02`.

The complete Gate 6E configuration count is nine. No configuration may be added, removed, substituted, or tuned during Gate 6E2.

## Point prediction boundary

The committed histogram-gradient-boosting outer-fold predictions remain the interval centers and authoritative point evidence. Exact point-prediction parity is required at absolute tolerance `1e-12`.

Gate 6E2 may reconstruct training-only inner out-of-fold residuals using the fixed point-model identity. It may not:

- search point-model parameters;
- change the point-model family;
- center intervals on a rejected Gate 6B, Gate 6C, or Gate 6D challenger;
- use locked-test rows or locked predictions;
- reinterpret Gate 6E as a second confirmatory evaluation.

## Calibration boundary

Calibration is separate from point fitting. It uses absolute residual scores from chronology-safe inner out-of-fold predictions, the trailing fifteen percent of eligible residuals, and a minimum of `672` calibration origins.

The first outer interval uses training-only residuals. Sequential updates may use only targets revealed before the current target. In-sample residuals, random splits, full-sample residual pools, future residual updates, and missing-value rescue are prohibited.

## Coverage and interval rules

The frozen levels are 80%, 90%, and 95%, with 90% primary. Intervals are central, use the finite-sample higher-quantile rule, record raw bounds, and apply the same zero-kWh lower support floor to every method and fold.

Interval nesting is mandatory. Post-hoc sorting, candidate-specific clipping, coverage-level substitution, and retrospective widening are prohibited.

## Optimization boundary

The expanding configuration is the uncertainty reference. The primary objective is weighted interval score under hard coverage and governance constraints.

A challenger must improve weighted interval score by at least one percent, improve at least three outer folds, avoid more than two percent degradation in any fold, and avoid more than one percent peak-state weighted-score degradation. The reference may be selected if feasible. No action is mandatory when no configuration is feasible.

Final authority is human. GenAI has no vote and automatic promotion is prohibited.

## Resource boundary

Gate 6E2 is limited to:

- nine unique configurations;
- one serial CPU execution lane;
- zero retries;
- 120 total wall-clock minutes;
- 4,096 MB peak memory;
- 250 MB total evidence package;
- zero hosted API use;
- zero external API cost;
- deterministic seed `20260725`.

## Gate 6E1 prohibited actions

Gate 6E1 performs no:

- model fit or refit;
- residual construction;
- calibration;
- interval generation;
- coverage, width, interval-score, or optimization calculation;
- uncertainty execution artifact creation;
- locked-test access or locked-prediction parsing;
- confirmatory evaluation;
- automatic promotion;
- probabilistic-authority claim;
- V1 mutation.

## Closure

The contract, schema, closure manifest, governance module, validator, tests, documentation, and read-only workflow constitute the complete Gate 6E1 evidence.

Gate 6E2 is identified as the next subgate but is not authorized. Gate 6E2, Gate 6E3, Gate 6F, and Gate 6G remain blocked pending separate approval and their required predecessor evidence.
