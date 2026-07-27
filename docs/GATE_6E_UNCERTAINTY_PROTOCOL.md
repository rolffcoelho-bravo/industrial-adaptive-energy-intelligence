# Gate 6E probabilistic optimization and uncertainty protocol

## Purpose

Gate 6E evaluates whether a governed prediction-interval layer can add calibrated decision support around the retained V1 histogram-gradient-boosting point champion. It does not reopen predictive-model selection and does not assume that an interval method will be accepted.

The gate sequence is:

1. Gate 6E1 freezes the uncertainty contract and implementation boundary.
2. Gate 6E2 executes the complete training-and-validation-only interval benchmark after separate explicit authorization.
3. Gate 6E3 records the human accept, reject, or defer decision and closes Gate 6E.

## Retained point-model boundary

The interval center is the validation prediction from `v1_frozen_champion`, implemented by the frozen histogram-gradient-boosting model. Gate 6E may reconstruct training-only residual evidence required for calibration, but it may not search for a different point model, modify the committed validation predictions, use a rejected challenger, or access the locked test.

Point-prediction parity is a hard constraint. Every Gate 6E2 interval row must reconcile with the committed histogram-gradient-boosting outer-fold prediction at an absolute tolerance of `1e-12`.

## Admissible methods

### Expanding absolute-residual conformal

This is the reference method. The interval radius is the finite-sample conformal quantile of the eligible absolute residual pool. After outer validation begins, the pool may expand only when a previously forecast target has been revealed.

### Rolling absolute-residual conformal

This challenger uses the same absolute-residual score but retains only the most recent `672` or `2,688` eligible residuals. These windows correspond to seven and twenty-eight days at the fifteen-minute cadence.

### Adaptive conformal inference

This challenger uses the rolling residual pool with bounded sequential updates to the effective miscoverage level. The frozen window lengths are `672` and `2,688`; the frozen update rates are `0.005`, `0.01`, and `0.02`.

The adaptive state for each coverage level may use only miss indicators whose targets are already observable before the current target. Its alpha state is bounded between `0.001` and `0.50`.

## Excluded methods

Conformalized quantile regression is excluded because it would require new lower- and upper-quantile model fitting. Ensemble batch prediction intervals are excluded because bootstrap predictive refitting belongs to a separately authorized ensemble design. Foundation-model native quantiles are excluded because Gate 6D treated them as uncalibrated diagnostics attached to rejected point challengers. Parametric Gaussian intervals are excluded because Gate 6E does not introduce an unsupported residual-distribution assumption.

These exclusions prevent Gate 6E from becoming a rescue lane for models or methods rejected in earlier gates.

## Calibration chronology

Gate 6E2 must use four expanding-window outer folds, three chronological inner folds, and a four-interval purge. Calibration residuals must come from inner out-of-fold point predictions. In-sample residuals, random splitting, full-sample preprocessing, and locked-test calibration are prohibited.

For each outer fold:

1. reconstruct the fixed V1 point-model identity on admissible outer-training data;
2. generate chronology-safe inner out-of-fold predictions;
3. form absolute residual scores;
4. retain the trailing fifteen percent of eligible residuals, with at least `672` origins;
5. initialize the first outer interval using training-only residuals;
6. update state only after earlier outer targets are revealed;
7. evaluate the frozen configuration once on that outer fold.

At prediction origin `t`, the uncertainty state may contain no residual whose target dependency lies after `t`.

## Interval construction

The target coverage levels are `0.80`, `0.90`, and `0.95`; `0.90` is primary. Intervals are central and use the finite-sample quantile rank based on `ceil((n + 1) * (1 - alpha))` with the higher empirical quantile.

Raw lower and upper bounds must be recorded. The final lower bound is uniformly clipped at the known target support of zero kWh. The clipping rule is frozen before execution, its activation rate must be reported, and interval nesting must remain exact across coverage levels.

## Required evidence

For every configuration, coverage level, fold, and required subgroup, Gate 6E2 must report empirical coverage, absolute coverage error, mean interval width, training-IQR-normalized width, interval score, peak-state coverage, peak-state width, and peak-state interval score.

Aggregate evidence must include weighted interval score, peak-state weighted interval score, maximum fold coverage error, maximum rolling-672 coverage error, longest miss run, nesting violations, support-floor activation, point-prediction parity, chronology and finite-evidence controls, deterministic replay, CPU portability, latency, memory, artifact size, failures, and complete lineage.

## Hard constraints

A feasible configuration must satisfy all contract constraints, including:

- aggregate absolute coverage error no greater than `0.03`, `0.02`, and `0.03` at 80%, 90%, and 95%;
- aggregate peak-state 90% coverage of at least `0.85`;
- maximum outer-fold 90% absolute coverage error no greater than `0.05`;
- maximum rolling-672 90% absolute coverage error no greater than `0.08`;
- zero nesting, support, point-parity, chronology, leakage, locked-test, nonfinite, missing-artifact, replay, and CPU-portability violations.

Failure of any hard constraint makes the configuration ineligible.

## Optimization and promotion

The primary objective is weighted interval score. Secondary objectives are 90% width, peak-state weighted interval score, fold calibration stability, rolling calibration stability, and latency.

The expanding reference may be selected when it is feasible. A challenger must additionally improve weighted interval score by at least one percent, improve at least three of four folds, avoid more than two percent single-fold weighted-score degradation, and avoid more than one percent peak weighted-score degradation.

Selection uses constrained Pareto evidence, deterministic tie-breaking, and final human approval. When no configuration is feasible, the required outcome is no action. Automatic promotion and a GenAI vote are prohibited.

## Execution and claim boundary

Gate 6E1 performs no point fitting, residual construction, calibration, interval generation, metric calculation, optimization, or promotion. Gate 6E2 requires separate explicit authorization. Gate 6E3 remains mandatory for probabilistic authority.

All Gate 6E evidence remains training-and-validation only. It cannot support confirmatory, production, savings, drift, causal, or optimization-impact claims.
