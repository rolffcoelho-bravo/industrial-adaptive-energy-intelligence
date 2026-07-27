# Validation gates

The project advances through explicit evidence contracts. Each stage must pass its technical controls before dependent stages are enabled. A later gate cannot reopen or silently mutate a closed earlier gate.

## V1 governed evidence sequence

| Gate | Technical decision | Status | Required evidence |
|---:|---|---|---|
| 0 | Repository identity and contracts | Closed | CI, policies, schemas, public-content controls |
| 1 | Data source and immutable snapshot | Closed | Official UCI source, manifest, hash, license, data-quality checks |
| 2 | Forecast target and leakage boundary | Closed | Target schema, chronology ownership, leakage tests |
| 3 | Governed Silver analytical layer | Closed | Typed table, effective timestamps, feature availability, parity checks |
| 4A-4D | Chronological validation, candidate ladder, and model freeze | Closed | Expanding-window folds, benchmarks, candidate evidence, frozen model contract |
| 4E | Single authorized confirmatory evaluation | Closed | One untouched test execution and immutable terminal evidence |
| 4F | Confirmatory closure | Closed | Closure manifest, retired evaluator, prohibited second evaluation |
| 5A | Reporting alignment | Closed | Evidence-aligned narrative, section contract, model boundaries |
| 5B | Deterministic reporting evidence | Closed | Final tables, lineage, hashes, manifest |
| 5C | Governed visual evidence | Closed | Five approved figures, visual identities, green CI |
| 5D1-5D2 | GitHub-native LaTeX reporting implementation | Closed | Governed source, build workflow, structural validation |
| 5D3 | Human visual approval | Closed | Five-page 200 DPI review and approval record |
| 5D4 | Formal reporting closure | Closed | Canonical PDF identity, visual fingerprints, closure schema and manifest |
| 5E | Frozen V1 release closure | Closed | Release manifest, `v1.0.0` tag, GitHub Release, canonical PDF asset, final checksum verification |

## V2 governed development sequence

| Gate | Technical decision | Status | Required evidence |
|---:|---|---|---|
| 6A | V2 architecture and optimization governance | Closed | Provider-neutral architecture, optimization contract, artifact schemas, V1 immutability checks, acceptance tests |
| 6B | Advanced tabular challengers | Closed | Approved search spaces, nested chronological evidence, complete trial records, Pareto evidence, human rejection record, retained V1 incumbent |
| 6C | Neural forecasting | Closed | Five governed seeds, chronological evidence, stability, resource, portability, human rejection record, retained V1 incumbent |
| 6D | Time-series foundation models | In progress | Gate 6D1 identity, license, context, adaptation, resource, and provenance lock; Gate 6D2 governed validation; Gate 6D3 human decision |
| 6E | Probabilistic optimization and uncertainty | Planned | Separate calibration, coverage and width evidence, governed uncertainty promotion |
| 6F | Ensembles and governed routing | Planned | Out-of-fold-only weights, routing rules, constrained Pareto evidence, human approval |
| 6G | Efficiency and portability | Planned | Runtime, memory, model-size, adapter parity, and reproducibility evidence |

## Closed confirmatory boundary

The V1 model and confirmatory evidence are immutable. The following actions are prohibited:

- model re-estimation against the locked period;
- a second locked-test evaluation;
- threshold, temporal-block, subgroup, or metric changes;
- parsing locked prediction rows for new exploratory analysis;
- reactivation of the retired evaluator;
- unsupported structural-drift, optimization, savings, causal, or live-production claims.

## Reporting closure boundary

Gate 5D4 freezes:

- the LaTeX source;
- the exact approved PDF identity;
- five rendered-page fingerprints;
- the report payload;
- five figures and five reporting tables;
- section order, conclusions, model boundaries, and approval evidence.

The exact approved PDF is the only admissible V1 release asset. A later timestamped LaTeX rebuild may be accepted only as a visual-equivalence check and cannot replace the frozen canonical binary.

## V1 release boundary

Gate 5E freezes:

- release version `1.0.0` and tag `v1.0.0`;
- the V1 release manifest and release notes;
- the exact canonical PDF release asset;
- reporting closure, payload, LaTeX source, figures, and tables;
- selected-model and locked-test identities;
- the checksum file published with the release;
- the rule that V1 assets cannot be overwritten or silently replaced.

## Gate 6A architecture boundary

Gate 6A freezes the V2 development rules before any V2 model execution:

- the seven-layer architecture from data evidence to human decision;
- provider-neutral execution, evidence, and decision interfaces;
- GCP as a planned adapter rather than a domain dependency;
- Google Workspace as a collaboration and approval layer rather than a training environment;
- nested expanding-window validation and complete locked-test exclusion;
- multiobjective definitions, hard constraints, search budgets, and random seeds;
- out-of-fold-only ensemble evidence;
- separate uncertainty calibration;
- constrained Pareto selection and final human promotion authority;
- schemas for search spaces, objectives, trials, and promotion decisions.

Gate 6A performed no model fitting, optimization trial, locked-test access, or confirmatory evaluation.

## Gate 6B closure boundary

Gate 6B freezes:

- XGBoost histogram, LightGBM L1, and CatBoost MAE as the evaluated advanced tabular families;
- exactly 12 prespecified configurations;
- four outer folds, three inner folds, and the four-interval purge;
- complete validation, trial, resource, portability, and out-of-fold evidence;
- the human decision rejecting all three challengers;
- retention of the frozen V1 histogram-gradient-boosting incumbent;
- the rule that Gate 6B evidence remains validation-only and cannot be represented as confirmatory evidence.

The maximum Gate 6B prediction origin is 28,027. The maximum target dependency is 28,028. The locked test begins at 28,032 and was not accessed.

## Gate 6C closure boundary

Gate 6C freezes:

- compact N-HiTS, compact TiDE, and compact PatchTST as the evaluated neural families;
- one prespecified configuration per family;
- five frozen seeds, four outer folds, and the four-interval purge;
- 60 candidate-seed-fold evaluations and 105,060 out-of-fold prediction rows;
- aggregate, peak-state, seed-stability, temporal-stability, resource, portability, failure, and lineage evidence;
- the human decision rejecting all three neural challengers;
- retention of the frozen V1 histogram-gradient-boosting incumbent;
- the rule that Gate 6C evidence remains validation-only and cannot be represented as confirmatory evidence.

The maximum Gate 6C prediction origin is 28,027. The maximum target dependency is 28,028. The locked test begins at 28,032 and was not accessed.

## Gate 6D1 closure boundary

Gate 6D1 freezes:

- Chronos-2, TimesFM 2.5, and Moirai 2.0 as the complete Gate 6D candidate set;
- exact public model revisions, source revisions, weight SHA-256 identities, access modes, and licenses;
- zero-shot univariate forecasting only;
- a common context of 672 intervals and a one-interval forecast horizon;
- the 0.50 quantile or model-equivalent median as the authoritative point forecast;
- the same four outer folds, four-interval purge, and 7,004 validation origins;
- no fine-tuning, adapters, calibration, search, model-specific context tuning, covariate lane, multivariate lane, ensemble, or fallback substitution;
- CPU, 6,144 MB memory, 1,200 MB download, 120-minute candidate, 360-minute total, batch-size, and cost boundaries;
- Chronos-2 and TimesFM 2.5 as commercially promotion eligible under Apache-2.0 weights;
- Moirai 2.0 as research-only under CC-BY-NC-4.0 weights and prohibited from promotion;
- no model download, library loading, inference, prediction generation, metric calculation, or execution evidence in Gate 6D1;
- final human promotion authority and a separate authorization requirement for Gate 6D2.

Gate 6D1 remains training-and-validation only in design. Locked-test access, locked-prediction parsing, confirmatory evaluation, automatic promotion, and V1 mutation are prohibited.

## Next permitted decision

V1 remains closed and immutable. Gates 6A, 6B, 6C, and Gate 6D1 are closed.

The next permitted decision is **explicit authorization for Gate 6D2 governed foundation-model validation execution**. Without that authorization, no model weight may be downloaded, loaded, or used to generate forecasts. Gate 6D3 and Gate 6E remain blocked.
