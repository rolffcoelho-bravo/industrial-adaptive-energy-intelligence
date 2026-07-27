# Gate 6D governed time-series foundation-model protocol

## Status

Gate 6D is the next V2 predictive-model lane after the formal closure of Gate 6C. Gate 6D1 freezes the benchmark design, model provenance, licensing boundary, context rules, resource limits, evidence requirements, and prohibited actions before any model weight is downloaded or any forecast is generated.

Gate 6D1 is implementation-only. Gate 6D2 requires a separate explicit authorization.

## Objective

Gate 6D tests whether public pretrained time-series foundation models can improve one-step-ahead industrial energy-demand forecasting relative to the frozen V1 histogram-gradient-boosting incumbent under the same chronology, leakage, peak-state, resource, and human-promotion controls.

Foundation models are benchmark challengers. They are not presumed champions.

## Candidate models

| Candidate | Frozen model identity | Model revision | Weight SHA-256 | Weight license | Gate 6D role |
|---|---|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` | `95a9710` | `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` | `Apache-2.0` | Research benchmark and commercial-promotion eligible |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8` | `Apache-2.0` | Research benchmark and commercial-promotion eligible |
| Moirai 2.0 | `Salesforce/moirai-2.0-R-small` | `c1b8789901bf60568fc4b2726f7d3aa256ac4ac8` | `fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669` | `CC-BY-NC-4.0` | Research benchmark only; commercial promotion prohibited |

The source repositories are also frozen for implementation provenance:

- Chronos-2: `amazon-science/chronos-forecasting` at `7dc4435706a4454feb79df44ca9f33631f3027bf`;
- TimesFM 2.5: `google-research/timesfm` at `3dae50b20d7a724981e8ea36cda75578f80dd2dc`;
- Moirai 2.0: `SalesforceAIResearch/uni2ts` at `cfd46d4510ed8896f263116f32928eede05b0a75`.

A model revision, source revision, weight hash, access mode, or license change requires a new reviewed contract. Silent upstream substitution is prohibited.

## Benchmark hypothesis

The three candidates represent distinct pretrained forecasting hypotheses:

- Chronos-2 tests direct quantile forecasting from a pretrained encoder foundation model;
- TimesFM 2.5 tests long-context decoder-only continuous forecasting;
- Moirai 2.0 tests a compact recursive multi-quantile foundation model under a research-only license boundary.

The comparison asks whether pretrained cross-domain temporal representations provide incremental validation value beyond the current structured, feature-engineered incumbent for a one-interval horizon.

## Frozen analytical design

All candidates use one identical benchmark lane:

- zero-shot univariate forecasting only;
- source series: `usage_kwh`;
- target: `usage_kwh_t_plus_1`;
- cadence: 15 minutes;
- context: 672 intervals, equal to seven days;
- horizon: one interval;
- point forecast: the 0.50 quantile or model-equivalent median;
- required recorded quantiles: 0.10, 0.50, and 0.90;
- probabilistic outputs are diagnostic and non-authoritative until Gate 6E;
- four expanding-window outer folds;
- four-interval purge;
- 7,004 validation origins;
- maximum prediction origin exclusive: 28,028;
- maximum target dependency exclusive: 28,032;
- admissible partitions: training and validation only.

Model-specific context tuning is prohibited. A shared seven-day context prevents a foundation model from receiving a post-hoc information advantage and captures daily, weekday, and weekend structure while remaining within every candidate's declared context capacity.

## Prohibited rescue lanes

Gate 6D does not permit:

- fine-tuning;
- LoRA, adapters, or parameter-efficient training;
- calibration;
- hyperparameter search;
- model-specific context selection;
- covariate or multivariate inference;
- random sampling for the authoritative point forecast;
- ensembles or routing;
- fallback model substitution;
- missing-value imputation introduced only for a candidate;
- alternative target definitions;
- locked-test access;
- parsing locked predictions;
- confirmatory evaluation;
- automatic promotion.

A failed origin, dependency error, unsupported input, memory failure, or timeout remains part of the evidence. It cannot be silently removed or replaced.

## Gate sequence

### Gate 6D1: contract and feasibility lock

Gate 6D1 creates the contract, schemas, implementation-only governance utilities, tests, closure evidence, documentation, and read-only CI. It performs no network model retrieval, model loading, inference, metric calculation, or validation prediction generation.

### Gate 6D2: governed validation execution

Gate 6D2 may begin only after separate human authorization. It must download only the frozen revisions, verify every weight hash before loading, execute the zero-shot lane, and publish complete prediction, metric, failure, resource, cost, portability, and lineage evidence.

### Gate 6D3: human decision and closure

Gate 6D3 records the human accept, reject, or defer decision. Moirai 2.0 may inform research comparison but cannot be promoted into a commercially usable champion under its frozen non-commercial weight license. Gate 6E remains blocked until Gate 6D closes.

## Evaluation evidence

Gate 6D2 must report, at minimum:

- aggregate MAE;
- peak-state MAE;
- relative aggregate and peak-state change versus V1;
- outer-fold MAE and positive-fold count;
- maximum single-fold degradation;
- complete out-of-fold predictions;
- failed-origin records;
- weight size and verified hash;
- peak memory;
- p50 and p95 inference latency per 1,000 rows;
- total wall-clock time;
- CPU execution result;
- deterministic replay result;
- source and model revision verification;
- runner-cost evidence;
- commercial-use and promotion eligibility.

## Promotion controls

A candidate can be promotion eligible only when all frozen requirements pass:

- at least 1% mean validation-MAE improvement over V1;
- improvement in at least three of four outer folds;
- no single-fold MAE degradation above 2%;
- peak-state MAE degradation no greater than 1%;
- all chronology, provenance, resource, portability, and finite-evidence constraints pass;
- the model weights permit commercial use;
- a human records the final decision.

No single scalar score and no model reputation can override a failed hard constraint.

## Resource and access boundary

The canonical Gate 6D2 environment is CPU-only. Each candidate is bounded by:

- maximum peak memory: 6,144 MB;
- maximum download size: 1,200 MB;
- maximum candidate runtime: 120 minutes;
- maximum total runtime: 360 minutes;
- maximum batch size: 32;
- hosted API use: prohibited;
- paid API use: prohibited;
- external API cost: `$0.00`;
- actual runner-resource and cost evidence: required.

Gate 6D1 permits no model download or inference. Gate 6D2 may reuse a cache only after exact revision and SHA-256 verification. Unpinned revisions, remote-code trust, and arbitrary model code are prohibited.

## Claims boundary

Gate 6D can produce governed validation evidence only. It cannot support production, savings, structural-drift, causal, optimization-impact, or confirmatory claims. It also cannot support a general claim that foundation models are superior to conventional forecasting models.

The admissible conclusion is limited to what the frozen candidates demonstrate under the frozen one-step, seven-day-context, training-and-validation-only protocol.
