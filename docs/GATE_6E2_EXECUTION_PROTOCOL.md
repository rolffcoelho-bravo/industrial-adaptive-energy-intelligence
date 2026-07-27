# Gate 6E2 governed uncertainty execution protocol

## Decision

Gate 6E2 is authorized to execute the uncertainty contract frozen by Gate 6E1. The stage produces validation-only interval evidence around the retained V1 histogram-gradient-boosting point champion.

Gate 6E2 does not reopen point-model selection and does not authorize Gate 6E3.

## Point-model boundary

The committed Gate 4C3 histogram-gradient-boosting outer-fold predictions remain the authoritative interval centers. Gate 6E2 must preserve the exact fold identifiers, row positions, targets, peak-state labels, and point predictions.

The absolute point-prediction parity tolerance is `1e-12`.

No Gate 6B, Gate 6C, or Gate 6D challenger may replace the retained point model.

## Calibration construction

For each outer fold, Gate 6E2 reconstructs training-side calibration residuals using:

- the already selected histogram-gradient-boosting parameters for that outer fold;
- the frozen feature and preprocessing contract;
- three chronological inner splits;
- a four-interval gap;
- absolute residual scores from inner out-of-fold predictions only;
- the trailing fifteen percent of eligible inner out-of-fold residuals;
- at least 672 initial calibration origins.

No parameter search is performed. In-sample residuals, random splits, full-sample residual pools, and locked-period residuals are prohibited.

## Sequential information timing

The first validation interval in each outer fold uses training-only calibration residuals. After an interval is issued, its residual may enter the calibration pool only for later origins.

No current or future target may influence its own interval.

## Frozen configuration set

The execution covers exactly nine configurations:

1. `expanding_all`;
2. `rolling_672`;
3. `rolling_2688`;
4. `aci_672_0p005`;
5. `aci_672_0p01`;
6. `aci_672_0p02`;
7. `aci_2688_0p005`;
8. `aci_2688_0p01`;
9. `aci_2688_0p02`.

The central coverage levels are 80%, 90%, and 95%, with 90% primary.

## Interval construction

All configurations use the same absolute-residual conformity score and finite-sample higher-quantile rule. Raw bounds are recorded before a uniform zero-kWh lower support floor is applied.

Interval nesting is mandatory. Post-hoc sorting, retrospective widening, candidate-specific clipping, and coverage substitution are prohibited.

## Required evidence

Gate 6E2 must publish:

- configuration-level results;
- coverage and sharpness results by fold and coverage level;
- outer-fold weighted interval scores;
- all 63,036 interval-prediction rows;
- calibration residuals and full calibration lineage;
- peak-state uncertainty evidence;
- rolling 672-origin coverage diagnostics;
- interval nesting and support diagnostics;
- exact point-prediction parity evidence;
- deterministic replay evidence;
- CPU portability, latency, memory, runtime, and artifact-size evidence;
- complete failure records;
- a constrained Pareto recommendation for human review;
- a hash-bound execution manifest.

## Selection boundary

Weighted interval score is the primary objective. Coverage, width, peak-state performance, temporal robustness, replay, portability, and governance controls remain hard constraints or secondary objectives as frozen in Gate 6E1.

The expanding configuration is the reference. A challenger is eligible only if it satisfies every hard constraint and every frozen relative-improvement requirement.

When no configuration is feasible, the required recommendation is `no_action`.

## Prohibited claims and actions

Gate 6E2 may not:

- access the locked test;
- parse locked predictions;
- perform a second confirmatory evaluation;
- search or mutate the point model;
- add or tune interval configurations;
- repair interval evidence after observing results;
- promote a configuration automatically;
- claim production readiness, savings, causal impact, drift control, optimization impact, or confirmatory authority.

## Closure boundary

Gate 6E2 ends with validation evidence and a recommendation pending human decision. Gate 6E3 remains mandatory and separately blocked until explicit approval is recorded.
