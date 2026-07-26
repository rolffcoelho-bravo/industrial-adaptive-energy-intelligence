# Gate 6C neural forecasting protocol

## Status

Approved for governed implementation. Model execution remains blocked until the machine-readable contract, schemas, tests, implementation interfaces, and GitHub-native validation workflow are merged and green.

## Objective

Evaluate a compact set of neural forecasting hypotheses against the frozen V1 histogram-gradient-boosting incumbent using training and validation evidence only.

## Candidate hypotheses

Gate 6C will evaluate three deliberately distinct neural families:

1. **N-HiTS** for hierarchical multi-rate temporal decomposition.
2. **TiDE** for efficient nonlinear encoder-decoder forecasting with covariates.
3. **PatchTST** for patch-based temporal representation.

Temporal Fusion Transformer is deferred because the current one-step target and small covariate structure do not justify its additional parameter and interpretation surface at this gate. It may be reconsidered only through formal change control.

## Required sequence

### Gate 6C1: contract and implementation boundary

- freeze candidate families, context length, feature availability, seeds, budgets, objectives, hard constraints, and outputs;
- define schemas for the neural contract, seed evidence, candidate evidence, promotion decision, and closure manifest;
- add implementation-only interfaces for candidate blueprints, causal windows, deterministic CPU controls, resource limits, and fail-closed execution guards;
- add source-level tests that prohibit locked-test access, locked-prediction access, fitting, prediction generation, and validation metric production;
- add an executable conformance test across the protocol, contract, schemas, source, workflow, and closure manifest;
- add a GitHub-native implementation-validation workflow with read-only repository permissions;
- perform no neural fitting.

### Gate 6C2: governed validation execution

- execute only after Gate 6C1 is merged, green, and formally closed;
- use the four frozen expanding-window outer folds and a four-interval purge;
- use three fixed seeds per configuration;
- calculate outer-fold and across-seed evidence;
- record failures, runtime, peak memory, model size, and inference latency;
- generate out-of-fold predictions only for training and validation origins;
- compare with the frozen V1 incumbent using the Gate 6A objectives and hard constraints.

### Gate 6C3: human promotion decision and closure

- review the complete validation evidence;
- promote, reject, or defer each candidate;
- preserve final human authority;
- close Gate 6C before Gate 6D begins.

## Frozen evidence boundary

- admissible partitions: training and validation only;
- maximum prediction origin exclusive: 28,028;
- maximum target dependency exclusive: 28,032;
- locked-test access: prohibited;
- locked-prediction parsing: prohibited;
- confirmatory evaluation: prohibited;
- V1 mutation: prohibited.

## Repeated-seed governance

Every neural configuration must use the same fixed seeds:

```text
20260725
20260726
20260727
```

A candidate cannot be promoted from a single favorable seed. Evidence must include mean, standard deviation, minimum, and maximum performance across seeds for every outer fold and in aggregate.

## Promotion requirements

A neural candidate must:

- improve aggregate validation MAE by at least 1 percent relative to the frozen V1 incumbent;
- improve or remain within 1 percent degradation on peak-state MAE;
- show positive aggregate MAE improvement in at least three of four outer folds;
- avoid more than 2 percent MAE degradation in any outer fold;
- satisfy all chronology, leakage, finite-evidence, artifact, portability, memory, and runtime constraints;
- remain Pareto eligible under accuracy, robustness, seed stability, model size, latency, and memory objectives;
- receive explicit human approval.

Automatic promotion is prohibited.

## Resource boundary

The canonical GitHub-native execution is CPU-only. Each model configuration must remain within the frozen wall-clock and memory budgets. GPU evidence may be added later as an adapter comparison but cannot replace the CPU portability result.

## Claims boundary

Gate 6C produces validation evidence only. It does not support production, savings, drift, causal, optimization-impact, or confirmatory claims.
