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
| 6D | Time-series foundation models | Closed | Gate 6D1 identity and license lock; Gate 6D2 complete validation evidence; Gate 6D3 human rejection record, retained V1 incumbent, closure manifest |
| 6E1 | Probabilistic uncertainty contract | Closed | Retained point-model identity, nine frozen conformal configurations, calibration chronology, metrics, hard constraints, budgets, closure manifest |
| 6E2 | Governed uncertainty execution | Blocked pending authorization | Complete coverage, width, interval-score, peak-state, temporal, resource, portability, failure, prediction, and lineage evidence |
| 6E3 | Human uncertainty decision and closure | Blocked | Human accept, reject, or defer record; retained point-model boundary; Gate 6E closure manifest |
| 6F | Ensembles and governed routing | Planned and blocked | Out-of-fold-only weights, routing rules, constrained Pareto evidence, human approval |
| 6G | Efficiency and portability | Planned and blocked | Runtime, memory, model-size, adapter parity, and reproducibility evidence |

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
- final human promotion authority.

## Gate 6D2 validation boundary

Gate 6D2 freezes the completed foundation-model validation evidence:

- three exact pinned model-weight identities and three exact source revisions;
- three isolated recorded execution environments;
- zero-shot univariate inference only;
- the common 672-interval context and one-interval horizon;
- four outer folds, four-interval purge, and 7,004 validation origins per candidate;
- 21,012 out-of-fold prediction rows;
- complete aggregate, peak-state, fold-level, diagnostic-quantile, resource, portability, failure, and lineage evidence;
- Chronos-2 mean MAE of 4.743581, 18.60% worse than V1;
- Moirai 2.0 mean MAE of 5.184533, 29.62% worse than V1;
- TimesFM 2.5 mean MAE of 5.374107, 34.36% worse than V1;
- zero positive outer folds for every candidate;
- no promotion-eligible candidate;
- raw native quantile crossings preserved without post-hoc sorting or calibration;
- Chronos-2 exact deterministic replay failure recorded;
- Moirai 2.0 commercial-use and promotion prohibition preserved;
- all candidates completed inside the frozen CPU, memory, download-size, and runtime boundaries;
- locked-test access, locked-prediction parsing, confirmatory evaluation, fine-tuning, calibration, search, and automatic promotion remained false;
- V1 remained immutable.

Gate 6D2 provides validation evidence only. It cannot be represented as confirmatory, production, savings, causal, drift, or optimization-impact evidence.

## Gate 6D3 closure boundary

Gate 6D3 freezes:

- the human decision rejecting Chronos-2 for promotion;
- the human decision rejecting TimesFM 2.5 for promotion;
- the human decision rejecting Moirai 2.0 for promotion and retaining it only as a research negative benchmark;
- retention of the frozen V1 histogram-gradient-boosting reference champion;
- exact candidate, fold, prediction, resource, provenance, failure, recommendation, and environment evidence hashes;
- the recorded Chronos-2 deterministic replay failure;
- the recorded native diagnostic quantile crossings without post-hoc repair;
- the Moirai non-commercial license and promotion boundary;
- the prohibition on context changes, multivariate or covariate rescue, fine-tuning, calibration, alternative revisions, additional model families, or reinterpretation inside Gate 6D;
- the rule that Gate 6D evidence remains validation-only and cannot be represented as confirmatory or production evidence;
- final closure of Gate 6D and unblocking of Gate 6E.

The maximum Gate 6D prediction origin is 28,027. The maximum target dependency is 28,028. The locked test begins at 28,032 and was not accessed.

## Gate 6E1 closure boundary

Gate 6E1 freezes:

- `v1_frozen_champion` as the immutable interval center and retained point model;
- exact point-prediction parity at absolute tolerance `1e-12`;
- expanding absolute-residual conformal as the reference;
- rolling absolute-residual conformal with 672- and 2,688-interval windows;
- adaptive conformal inference with both windows and update rates `0.005`, `0.01`, and `0.02`;
- exactly nine deterministic configurations;
- 80%, 90%, and 95% central intervals, with 90% primary;
- inner out-of-fold residual construction, a trailing 15% calibration tail, and at least 672 calibration origins;
- the finite-sample higher-quantile rule and revealed-target sequential update timing;
- uniform zero-kWh support clipping, raw-bound retention, and exact interval nesting;
- coverage, width, interval-score, peak-state, temporal-stability, resource, portability, failure, prediction, and lineage evidence requirements;
- hard coverage, point-parity, chronology, leakage, replay, portability, and artifact constraints;
- weighted interval score as the primary objective;
- one-percent challenger improvement, three positive folds, bounded fold degradation, and bounded peak degradation;
- feasible reference selection and a mandatory no-action outcome when no configuration is feasible;
- final human authority, no GenAI vote, and no automatic promotion;
- exclusion of quantile-model refits, bootstrap predictive ensembles, rejected foundation-model quantiles, and parametric Gaussian intervals;
- no point fitting, residual construction, calibration, interval generation, metric calculation, optimization execution, locked-test access, or probabilistic-authority claim in Gate 6E1.

Gate 6E1 is implementation-only. It closes the uncertainty contract but does not authorize Gate 6E2 execution.

## Gate 6E2 entry boundary

Gate 6E2 remains blocked until separate explicit authorization. When authorized, it must:

- execute all nine frozen configurations without substitution;
- preserve the four outer folds, three inner folds, four-interval purge, and 7,004 validation origins;
- center intervals on the committed V1 validation predictions;
- use only chronology-safe inner out-of-fold residuals and revealed prior targets;
- publish complete interval predictions, configuration results, coverage results, fold evidence, resource evidence, calibration lineage, failures, recommendation, and execution manifest;
- preserve locked-test exclusion and validation-only claims;
- leave the final accept, reject, or defer decision to Gate 6E3.

## Next permitted decision

V1 remains closed and immutable. Gates 6A, 6B, 6C, 6D, and Gate 6E1 are closed.

The next permitted decision is **explicit authorization for Gate 6E2 governed uncertainty execution**. Without that authorization, no residual construction, calibration, interval generation, uncertainty metric calculation, resource evidence, or promotion recommendation is permitted. Gate 6E3, Gate 6F, and Gate 6G remain blocked.
