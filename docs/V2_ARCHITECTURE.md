# V2 architecture

## Status

Gate 6A defines the architecture for V2. It does not fit a model, execute an optimization trial, access the locked test, or change any V1 evidence.

V1 remains frozen at release `v1.0.0`, commit `9e8cfe567d2174639675cbc784f21fe968dafe92`.

## Architectural sequence

```text
Data Evidence
    -> Predictive Models
    -> Governed Model Optimization
    -> Uncertainty
    -> Robust Selection
    -> Generative Interpretation
    -> Human Decision
```

The sequence separates numerical truth, optimization, interpretation, and consequential approval. Each layer has an explicit owner, admissible inputs, governed outputs, and prohibited actions in [`configs/v2_architecture_contract.yml`](../configs/v2_architecture_contract.yml).

## 1. Data Evidence

The data-evidence layer owns provenance, chronology, schema integrity, feature availability, and partition identity. It accepts governed real observations and frozen data contracts. It produces validated evidence contracts and chronological partitions.

This layer must reject leakage before any model receives data. Random splitting, full-sample preprocessing, future-value filling, centered windows that cross an origin boundary, and locked-test influence are not admissible.

## 2. Predictive Models

The predictive-model layer receives training and validation partitions together with a governed search-space definition. It may produce fitted candidate artifacts and out-of-fold predictions.

Every candidate must record:

- algorithm identity;
- complete parameter values;
- code commit;
- dependency identity;
- fold identity;
- random seed;
- training and validation boundaries;
- resource measurements;
- failure evidence when execution does not complete.

No V2 candidate may read the V1 locked-test partition.

## 3. Governed Model Optimization

Optimization is a formal analytical layer, not an informal tuning step. Search spaces, objectives, hard constraints, budgets, seeds, stopping rules, and promotion requirements must be approved before execution.

The optimizer may execute only the approved search space. It produces complete trial evidence and a constrained Pareto set. It may not silently widen a range, add a candidate, change an objective, substitute a seed, or continue beyond the governed budget.

The numerical optimizer is deterministic with respect to the approved contract and recorded randomness. A failed trial remains part of the evidence record.

## 4. Uncertainty

Uncertainty calibration is separated from point-forecast fitting. Calibration uses a trailing portion of the applicable training window and remains inside each chronological outer fold.

The uncertainty layer reports at least:

- target coverage level;
- observed coverage;
- absolute coverage error;
- mean interval width;
- fold-level stability;
- calibration sample boundary.

Locked-test calibration is prohibited.

## 5. Robust Selection

Robust selection receives the constrained Pareto set, temporal evidence, peak-state evidence, uncertainty evidence when applicable, portability evidence, and resource evidence.

A candidate is not promoted by a single scalar score. The process first enforces hard constraints, then identifies Pareto-eligible candidates, then exposes tradeoffs to the human decision authority.

The frozen V1 champion and persistence remain explicit references. Automatic promotion is prohibited.

## 6. Generative Interpretation

Generative AI is advisory. It may propose candidate families, draft search-space rationale, summarize approved evidence, identify model-boundary questions, and prepare review prompts.

Generative AI may not:

- execute optimization trials;
- calculate authoritative metrics;
- fabricate evidence;
- approve promotion;
- access the locked test;
- convert provisional evidence into a confirmed claim.

Any interpretation must cite governed machine-readable evidence and distinguish measured facts from inference.

## 7. Human Decision

Human authority approves or rejects consequential promotion. The decision record must include the evidence identity, decision, rationale, approver role, and next action.

The human authority can stop execution, reject an otherwise Pareto-eligible candidate, require additional validation evidence, or authorize a separately governed future confirmatory protocol. Gate 6A does not authorize such a protocol.

## Provider-neutral core

The analytical core is provider-neutral. Domain contracts must not depend on a cloud SDK, workspace product, proprietary experiment tracker, or vendor-specific URI.

The provider-neutral interfaces cover:

- trial execution;
- objective evaluation;
- promotion authority;
- artifact storage.

Each interface is bound to machine-readable schemas. A provider adapter may map these contracts to an execution service, but it may not redefine analytical truth.

## GCP adapter boundary

GCP is the first planned cloud execution adapter. Its status in Gate 6A is `planned_not_implemented`.

A future GCP adapter must preserve contract-identical inputs and outputs, deterministic artifact identity, provider-neutral domain logic, chronology, and locked-test exclusion. It cannot redefine objectives, change promotion rules, access the locked test, or create unverified performance claims.

No production or cloud-execution claim is made in Gate 6A.

## Google Workspace boundary

Google Workspace is a collaboration, approval, and decision-distribution layer. It is not a training or metric-computation environment.

A future integration may publish approved summaries, collect human approvals, and distribute decision records. It may not fit models, calculate authoritative metrics, or mutate governed evidence.

## Databricks boundary

Databricks remains an enterprise execution path. The V2 core does not depend on Databricks. Any Databricks execution must preserve portable contracts, chronology, and complete evidence lineage.

## V1 immutability

The V2 architecture contract lists the frozen V1 paths. CI compares those paths with the immutable `v1.0.0` tag and verifies the governed hashes recorded in the V1 release manifest.

The following remain prohibited:

- changing V1 model or confirmatory evidence;
- parsing locked predictions for new analysis;
- reusing the locked test;
- reactivating the retired evaluator;
- recalculating confirmatory metrics;
- replacing a V1 release asset;
- altering the V1 conclusion.

## Gate transition

Gate 6A closes only the architecture and optimization-governance design. The next permitted stage is Gate 6B, which may define and validate advanced tabular challengers under these contracts using training and validation evidence only.
