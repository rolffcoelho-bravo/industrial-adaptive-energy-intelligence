# Gate 6D1 foundation-model scope lock

## Decision

Gate 6D1 is approved and closed as an implementation-only contract subgate. It freezes the admissible foundation-model benchmark before any model weight download or forecast execution.

Gate 6D2 is identified as the next subgate but is not authorized by this closure.

## Frozen candidate set

### Chronos-2

- candidate ID: `chronos_2_zero_shot`;
- model ID: `amazon/chronos-2`;
- model revision: `95a9710`;
- weight SHA-256: `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`;
- source repository: `amazon-science/chronos-forecasting`;
- source revision: `7dc4435706a4454feb79df44ca9f33631f3027bf`;
- source license: `Apache-2.0`;
- weights license: `Apache-2.0`;
- benchmark admissible: true;
- commercial-use eligible: true;
- promotion eligible: true.

### TimesFM 2.5

- candidate ID: `timesfm_2_5_zero_shot`;
- model ID: `google/timesfm-2.5-200m-pytorch`;
- model revision: `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`;
- weight SHA-256: `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`;
- source repository: `google-research/timesfm`;
- source revision: `3dae50b20d7a724981e8ea36cda75578f80dd2dc`;
- source license: `Apache-2.0`;
- weights license: `Apache-2.0`;
- benchmark admissible: true;
- commercial-use eligible: true;
- promotion eligible: true.

### Moirai 2.0

- candidate ID: `moirai_2_research_zero_shot`;
- model ID: `Salesforce/moirai-2.0-R-small`;
- model revision: `c1b8789901bf60568fc4b2726f7d3aa256ac4ac8`;
- weight SHA-256: `fb5652a3db8ea572606221b7cb1e77bb8962b168e4d4cc752cf31ceb04074669`;
- source repository: `SalesforceAIResearch/uni2ts`;
- source revision: `cfd46d4510ed8896f263116f32928eede05b0a75`;
- source license: `Apache-2.0`;
- weights license: `CC-BY-NC-4.0`;
- benchmark admissible: true;
- commercial-use eligible: false;
- promotion eligible: false.

Moirai 2.0 is retained because it adds research comparison value through a compact pretrained architecture. Its non-commercial weights create a hard deployment boundary. No metric can override that boundary.

## Frozen comparison design

- mode: `zero_shot_univariate`;
- source: `usage_kwh`;
- target: `usage_kwh_t_plus_1`;
- cadence: 15 minutes;
- common context: 672 intervals, equal to seven days;
- forecast horizon: one interval;
- point-forecast rule: 0.50 quantile or model-equivalent median;
- recorded quantiles: 0.10, 0.50, and 0.90;
- probabilistic authority in Gate 6D: false;
- outer folds: four;
- purge: four intervals;
- validation origins: 7,004;
- maximum prediction origin exclusive: 28,028;
- maximum target dependency exclusive: 28,032;
- partitions: training and validation only.

The common seven-day context is frozen for all candidates. Model-specific context tuning, extra covariates, multivariate inputs, and adaptation are outside Gate 6D.

## Frozen execution boundary

Gate 6D1 permits:

- contract parsing;
- schema validation;
- license and identity inspection;
- causal-window construction;
- promotion-eligibility inspection;
- source-boundary validation;
- tests using controlled in-memory fixtures;
- read-only CI.

Gate 6D1 prohibits:

- network model retrieval;
- downloading weights;
- loading foundation-model libraries;
- inference;
- prediction generation;
- metric calculation;
- fine-tuning;
- calibration;
- parameter-efficient adaptation;
- hyperparameter search;
- locked-test access;
- locked-prediction parsing;
- confirmatory evaluation;
- automatic promotion;
- creation of `outputs/v2/gate_6d/*` execution evidence.

## Frozen resource boundary for Gate 6D2

The future execution contract is bounded before authorization:

- canonical device: CPU;
- GPU required: false;
- maximum peak memory: 6,144 MB;
- maximum download per candidate: 1,200 MB;
- maximum runtime per candidate: 120 minutes;
- maximum total runtime: 360 minutes;
- maximum batch size: 32;
- hosted APIs: prohibited;
- paid APIs: prohibited;
- external API cost: `$0.00`;
- runner-cost measurement: required.

A candidate that cannot complete inside these limits must be recorded as a governed resource failure. The protocol cannot be widened during execution to rescue it.

## Frozen promotion boundary

Every commercially promotion-eligible candidate must:

1. improve mean validation MAE by at least 1% relative to `v1_frozen_champion`;
2. improve at least three of four outer folds;
3. avoid more than 2% MAE degradation in any one fold;
4. avoid more than 1% peak-state MAE degradation;
5. pass chronology, hash, revision, license, resource, portability, and evidence controls;
6. have weights that permit commercial use;
7. receive a human promotion decision.

Automatic promotion remains prohibited.

## Closed predecessors

Gate 6D1 depends on the immutable V1 release and the closed Gate 6C decision:

- V1 tag: `v1.0.0`;
- retained model: `v1_frozen_champion`;
- Gate 6C status: closed;
- locked-test access: false;
- confirmatory evaluation: false;
- next gate recorded by Gate 6C: `6D`.

Gate 6D1 cannot change or reinterpret Gate 6B or Gate 6C results.

## Closure evidence

- contract: `configs/foundation_model_contract.yml`;
- contract schema: `schemas/foundation_model_contract.schema.json`;
- closure schema: `schemas/gate_6d1_closure_manifest.schema.json`;
- closure manifest: `outputs/v2/gate_6d1_closure_manifest.json`;
- governance module: `src/iaei/modeling/foundation_governance.py`;
- contract validator: `src/iaei/foundation_contracts.py`;
- gate validator: `scripts/validate_gate_6d1.py`;
- tests: `tests/test_foundation_model_governance.py`;
- workflow: `.github/workflows/gate-6d1-foundation-contract.yml`.

## Next decision

Gate 6D1 is closed. Gate 6D2 remains blocked until a separate explicit human authorization permits the pinned-weight, validation-only benchmark execution.
