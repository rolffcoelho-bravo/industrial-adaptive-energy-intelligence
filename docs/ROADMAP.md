# Implementation roadmap

## V1 governed forecasting evidence

### Gates 0-3: repository, data, targets, and Silver layer

**Status:** Closed.

Completed evidence:

- reproducible Python package and CI;
- separated code and data licensing;
- real-data-only rule and placeholder rejection;
- official UCI dataset retrieval and immutable snapshot;
- SHA-256 verification and machine-readable provenance;
- source-aware timestamps and deterministic quality controls;
- next-interval regression target and next-hour peak-risk target;
- training-only peak threshold and leakage tests;
- typed Silver analytical table and feature-availability controls.

### Gates 4A-4D: model ladder, validation, and freeze

**Status:** Closed.

Completed evidence:

- persistence and naive references;
- Ridge, Elastic Net, and histogram gradient boosting candidates;
- four expanding-window chronological folds;
- validation-only promotion decision;
- frozen feature, target, chronology, and model contracts;
- portable selected-model evidence.

### Gates 4E-4F: confirmatory evaluation and closure

**Status:** Closed.

Completed evidence:

- one authorized untouched confirmatory execution;
- 7,004 prediction origins;
- aggregate and peak-state evidence;
- four prespecified temporal blocks, all positive;
- immutable terminal results and closure manifest;
- consumed authorization and retired evaluator;
- permanent prohibition on a second evaluation.

### Gates 5A-5C: reporting evidence and figures

**Status:** Closed.

Completed evidence:

- evidence-aligned five-page reporting contract;
- deterministic payload and five final reporting tables;
- complete evidence lineage;
- five approved 300 DPI decision figures;
- exact figure identities and public-content controls.

### Gates 5D1-5D4: GitHub-native technical brief and reporting closure

**Status:** Closed.

Completed evidence:

- governed five-page LaTeX source;
- containerized TeXLive compilation in GitHub Actions;
- structural validation and project-level PDF metadata;
- human review of all pages at 200 DPI;
- Gate 5D3 visual approval record;
- canonical PDF SHA-256 and rendered-page fingerprints;
- Gate 5D4 closure schema, manifest, tests, and workflow enforcement;
- explicit separation between the frozen canonical PDF and timestamped rebuilds.

### Gate 5E: frozen V1 release closure

**Status:** Closed.

Completed evidence:

- final schema-validated V1 release manifest;
- package version `1.0.0` and immutable tag `v1.0.0`;
- GitHub Release published from synchronized `main`;
- exact approved canonical PDF attached as a permanent release asset;
- release asset hash verified against Gate 5D4;
- payload, tables, figures, source, approval, closure, and locked-test identities recorded;
- final CI, report, and release workflow success;
- V1 immutability statement, release notes, and checksum asset;
- independent post-publication verification.

## V2 governed development

V1 remains immutable. V2 uses a separate architecture, evidence sequence, and version history.

The architectural sequence is:

```text
Data Evidence
    -> Predictive Models
    -> Governed Model Optimization
    -> Uncertainty
    -> Robust Selection
    -> Generative Interpretation
    -> Human Decision
```

### Gate 6A: architecture and optimization governance

**Status:** Closed.

Completed evidence:

- exact seven-layer V2 architecture contract;
- provider-neutral trial, objective, artifact, and promotion interfaces;
- planned GCP adapter boundary without a cloud dependency in the analytical core;
- Google Workspace restricted to collaboration, approval, and decision distribution;
- Databricks retained as a separate enterprise execution adapter;
- nested expanding-window validation rules;
- locked-test exclusion across fitting, calibration, ensembles, and promotion;
- multiobjective accuracy, robustness, stability, calibration, complexity, latency, and portability definitions;
- hard constraints, search budgets, fixed seeds, and change control;
- constrained Pareto selection with final human authority;
- governed schemas for search spaces, objective records, trial evidence, and promotion decisions;
- CI enforcement of the immutable V1 boundary;
- Gate 6A closure manifest and acceptance tests.

Gate 6A performed no model fitting, optimization trial, uncertainty calibration, locked-test access, or confirmatory evaluation.

### Gate 6B: advanced tabular challengers

**Status:** Closed.

Completed evidence:

- prespecified XGBoost histogram, LightGBM L1, and CatBoost MAE candidate families;
- exactly 12 governed configurations;
- four expanding-window outer folds, three chronological inner folds, and a four-interval purge;
- 7,004 validation origins per candidate;
- complete inner-search, outer-fold, peak-state, resource, portability, trial, and out-of-fold evidence;
- immutable comparison with the frozen V1 champion;
- all hard constraints passed;
- no challenger satisfied every frozen promotion requirement;
- human rejection of all three challengers;
- retention of the V1 histogram-gradient-boosting incumbent;
- Gate 6B closure schema, decision record, manifest, tests, and GitHub-native validation.

Gate 6B used training and validation evidence only. Maximum prediction origin was 28,027, maximum target dependency was 28,028, locked-test access was false, and no confirmatory evaluation occurred.

### Gate 6C: neural forecasting

**Status:** Closed.

Completed evidence:

- compact N-HiTS, compact TiDE, and compact PatchTST candidate families;
- one frozen configuration per family;
- five governed seeds per configuration;
- four expanding-window outer folds and a four-interval purge;
- 60 candidate-seed-fold evaluations;
- 7,004 validation origins per candidate and seed;
- 105,060 out-of-fold prediction rows;
- complete aggregate, peak-state, seed-stability, temporal-stability, resource, portability, failure, and lineage evidence;
- no neural challenger improved the V1 incumbent on aggregate MAE;
- no neural challenger improved any outer fold;
- human rejection of all three neural challengers;
- retention of the frozen V1 histogram-gradient-boosting incumbent;
- Gate 6C3 decision schema, closure schema, manifest, tests, and GitHub-native validation.

Gate 6C used training and validation evidence only. Maximum prediction origin was 28,027, maximum target dependency was 28,028, locked-test access was false, and no confirmatory evaluation occurred.

### Gate 6D: time-series foundation models

**Status:** Validation complete pending Gate 6D3 human decision.

Completed Gate 6D1 evidence:

- Chronos-2, TimesFM 2.5, and Moirai 2.0 frozen as the three benchmark candidates;
- exact model revisions, source revisions, weight SHA-256 identities, access modes, and licenses;
- zero-shot univariate inference as the only admissible benchmark mode;
- common seven-day context of 672 intervals and one-interval forecast horizon;
- identical four-fold chronology, four-interval purge, and 7,004 validation origins;
- fine-tuning, LoRA, calibration, search, model-specific context tuning, covariate inference, and ensembles prohibited;
- CPU, memory, download-size, runtime, batch-size, cost, and portability limits frozen;
- Chronos-2 and TimesFM 2.5 commercially promotion eligible under Apache-2.0 weights;
- Moirai 2.0 retained as a research-only benchmark under CC-BY-NC-4.0 weights and prohibited from promotion;
- implementation-only governance code, schemas, tests, closure manifest, documentation, and read-only CI.

Completed Gate 6D2 evidence:

- exact public model revisions and weight SHA-256 identities verified before inference;
- exact source revisions and three isolated execution environments recorded;
- 21,012 out-of-fold prediction rows across three candidates;
- complete aggregate, peak-state, temporal, resource, portability, diagnostic-quantile, failure, and lineage evidence;
- Chronos-2 produced the strongest foundation-model result but remained 18.60% worse than V1 on aggregate MAE and 15.69% worse in peak states;
- Moirai 2.0 was 29.62% worse on aggregate MAE and remained ineligible under its non-commercial weight license;
- TimesFM 2.5 was 34.36% worse on aggregate MAE;
- no candidate improved any of the four outer folds;
- all three completed inside the CPU, memory, download-size, and runtime limits;
- raw native quantile crossings preserved as diagnostic evidence without calibration or post-hoc sorting;
- Chronos-2 failed the exact deterministic replay control;
- no foundation-model candidate satisfied every frozen promotion requirement;
- locked-test access, locked-prediction parsing, confirmatory evaluation, fine-tuning, calibration, and search remained false;
- V1 remained immutable.

Next required decision:

- Gate 6D3 must record the human promotion, rejection, or deferral decision;
- the evidence supports rejecting all three candidates and retaining the frozen V1 champion;
- Gate 6E remains blocked until Gate 6D3 formally closes Gate 6D.

### Gates 6E-6G

**Status:** Planned.

Sequence:

- Gate 6E: probabilistic optimization and uncertainty calibration;
- Gate 6F: ensembles and governed routing using out-of-fold evidence only;
- Gate 6G: efficiency, portability, and adapter parity.

Later stages may add executable cloud adapters, governed generative interpretation, and human decision integrations. Each stage requires separate evidence and cannot inherit an unsupported claim from architecture alone.

V2 must preserve V1 unchanged. Generative AI may propose and interpret, deterministic systems execute and measure, and humans approve consequential promotion decisions.
