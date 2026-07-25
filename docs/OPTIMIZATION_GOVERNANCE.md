# Optimization governance

## Purpose

V2 optimization is governed before execution. The purpose is to prevent informal tuning, hidden search expansion, locked-test influence, single-metric promotion, and unsupported performance claims.

The machine-readable source of truth is [`configs/optimization_governance.yml`](../configs/optimization_governance.yml).

## Evidence boundary

Optimization may use only training and validation partitions. The locked test is not an optimization, calibration, ensemble, or selection input.

The required validation structure is nested expanding-window evaluation:

- four outer chronological folds provide promotion evidence;
- three inner chronological folds inside each outer training window support parameter selection;
- four intervals are purged at the relevant boundary;
- random splitting is prohibited;
- full-sample preprocessing is prohibited;
- ensemble evidence uses outer-fold out-of-fold predictions only.

A trailing 15 percent of the applicable training window is reserved for uncertainty calibration when calibration is required. The calibration partition must contain at least 672 origins.

## Reference models

Every candidate is evaluated against two references:

1. persistence, which preserves the operational benchmark;
2. the frozen V1 champion, `hist_gradient_boosting`, identified by release `v1.0.0`.

The V1 locked-test result is not reused for V2 promotion. V2 comparisons are based on admissible chronological validation evidence.

## Objective system

The objective set exposes tradeoffs rather than hiding them inside one weighted score.

| Objective | Direction | Purpose |
|---|---:|---|
| Aggregate MAE | Minimize | Point-forecast accuracy |
| Peak-state MAE | Minimize | High-demand robustness |
| Temporal-fold dispersion | Minimize | Stability across time |
| Calibration absolute error | Minimize | Interval coverage quality |
| Mean interval width | Minimize | Interval sharpness |
| Model size | Minimize | Complexity and portability |
| P95 inference latency | Minimize | Execution efficiency |
| Portability failure count | Minimize | Cross-environment reliability |

Not every objective applies to every gate. Point-forecast candidates must produce the point-forecast, stability, efficiency, and portability objectives. Uncertainty candidates must additionally produce calibration and interval-width evidence.

## Hard constraints

A trial is ineligible when any hard constraint fails. The frozen constraints require zero:

- chronology violations;
- leakage violations;
- locked-test accesses;
- missing required artifacts;
- nonfinite objective values;
- portability failures;
- ungoverned parameters.

A favorable MAE result cannot compensate for a governance failure.

## Search budgets

The contract defines maximum budgets, not targets that must be consumed.

| Candidate class | Maximum search | Parallelism | Wall-clock ceiling | Seeds per configuration |
|---|---:|---:|---:|---:|
| Advanced tabular | 96 trials | 4 | 360 minutes | 1 |
| Neural forecasting | 48 configurations | 4 | 720 minutes | 5 |
| Foundation models | 24 configurations | 2 | 720 minutes | 3 |
| Ensemble weights | 500 trials | 1 | 120 minutes | 1 |

Execution must stop when the approved search is complete or a budget ceiling is reached. A budget increase requires a new contract version and human approval before execution.

## Randomness governance

Deterministic candidates use seed `20260725` when a seed is required. Stochastic candidates use the fixed seed set:

```text
20260721, 20260722, 20260723, 20260724, 20260725
```

Seed substitution is prohibited. Stochastic evidence reports mean, standard deviation, minimum, and maximum across seeds. Missing seed runs make promotion evidence incomplete.

## Trial evidence

Every trial is serialized under [`schemas/trial_evidence.schema.json`](../schemas/trial_evidence.schema.json). The record includes:

- search-space identity and hash;
- candidate family and algorithm;
- complete parameter values;
- fold and seed identities;
- objective records;
- hard-constraint results;
- wall-clock, memory, model-size, and latency evidence;
- execution environment and dependency identity;
- code commit;
- outcome and complete failure record;
- final artifact hash.

Failed and pruned trials remain visible. Removing unfavorable trials from the evidence set is prohibited.

## Search-space approval

A search space must conform to [`schemas/governed_search_space.schema.json`](../schemas/governed_search_space.schema.json).

Before execution it must identify:

- gate and candidate family;
- algorithm;
- admissible data boundary;
- every parameter and range or categorical set;
- budget;
- seeds;
- objectives;
- hard constraints;
- code commit;
- human approval.

The executing system cannot add parameters or widen bounds.

## Pareto selection

Eligible trials are evaluated through constrained Pareto selection. A candidate remains eligible only when all hard constraints pass.

The promotion requirements are:

- at least 1 percent mean validation-MAE improvement relative to the frozen V1 champion;
- improvement in at least three of four outer folds;
- no single outer-fold MAE degradation greater than 2 percent;
- no peak-state MAE degradation greater than 1 percent;
- complete seed evidence when stochastic;
- complete portability evidence;
- complete resource evidence;
- Pareto eligibility.

These thresholds are prespecified for V2 development evidence. They do not authorize a new confirmatory claim.

## Human promotion decision

Promotion is never automatic. The deterministic system enforces constraints and prepares the Pareto evidence. A human records the final decision under [`schemas/promotion_decision.schema.json`](../schemas/promotion_decision.schema.json).

The decision can be:

- promote;
- reject;
- defer.

The record must preserve the candidate, references, evidence hash, requirement results, decision authority, rationale, code commit, and next action.

Generative AI has no vote and cannot calculate authoritative metrics.

## Ensemble governance

Ensemble optimization may use only outer-fold out-of-fold predictions. Locked-test predictions are prohibited.

Weights must be nonnegative, sum to one, and assign no more than 0.75 to one model. The ensemble uses the same objective system and requires human approval.

## Uncertainty governance

Point fitting and uncertainty calibration are separate. Calibration occurs inside the applicable outer training window.

The target coverage levels are 80, 90, and 95 percent. Absolute coverage error must not exceed 3 percentage points for eligibility. Interval width remains an explicit competing objective so nominal coverage cannot be achieved through uninformatively wide intervals.

## Claims and publication boundaries

Gate 6A creates architecture and governance evidence only. It does not demonstrate improved forecasting, optimization impact, production readiness, savings, drift, or causality.

Such claims require their own executable and prespecified evidence:

- optimization impact requires completed governed trials;
- production claims require deployed-system evidence;
- savings claims require an approved economic measurement design;
- drift claims require a prespecified drift protocol;
- causal claims require an identification strategy.

Provisional results must remain labelled as provisional.

## Change control

A governance change is valid only when it occurs before affected execution, receives human approval, creates a new contract version, and preserves the historical contract.

An active trial cannot mutate its governing search space, objective set, seed set, constraints, or budget.
