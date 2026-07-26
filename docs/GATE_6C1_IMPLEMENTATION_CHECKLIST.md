# Gate 6C1 implementation checklist

## Status

Active. Gate 6C1 is implementation-only. Neural fitting and validation execution are prohibited until this checklist is complete, merged, and green.

## Approved candidate families

- compact N-HiTS;
- compact TiDE;
- compact PatchTST.

No substitution is permitted without formal change control and explicit human approval.

## Frozen execution design

- exactly one configuration per family;
- exactly three configurations in total;
- fixed seeds: `20260725`, `20260726`, `20260727`;
- four frozen expanding-window outer folds;
- four-interval purge;
- CPU-only canonical GitHub execution;
- maximum prediction origin exclusive: `28028`;
- maximum target dependency exclusive: `28032`;
- training and validation evidence only.

## Required Gate 6C1 deliverables

1. neural-forecasting contract aligned with the approved protocol;
2. JSON schema for the neural contract;
3. schemas for seed evidence, candidate evidence, promotion decision, and closure manifest;
4. repository contract registration and validation;
5. source-level tests prohibiting locked-test and locked-prediction access;
6. deterministic seed and CPU controls;
7. explicit context-length and feature-availability controls;
8. resource, runtime, memory, model-size, latency, and portability boundaries;
9. GitHub-native workflow scaffolding that validates implementation only;
10. public-content audit and V1 immutability verification;
11. Gate 6B closure verification;
12. green CI and dedicated Gate 6C1 workflow;
13. detailed Gate 6C1 closure summary, technical assessment, recommendations, perceived-value opportunities, and exact Gate 6C2 approval request.

## Prohibited in Gate 6C1

- neural model fitting;
- hyperparameter search;
- outer-fold prediction generation;
- validation metric production;
- locked-test access;
- locked-prediction parsing;
- confirmatory evaluation;
- V1 mutation;
- automatic promotion;
- production, savings, drift, causal, optimization-impact, or confirmatory claims.

## Sequence protection

Gate 6C2 remains blocked until Gate 6C1 is merged and formally closed. Gate 6C3 remains blocked until Gate 6C2 evidence is complete. Gate 6D remains blocked until Gate 6C3 closes Gate 6C.
