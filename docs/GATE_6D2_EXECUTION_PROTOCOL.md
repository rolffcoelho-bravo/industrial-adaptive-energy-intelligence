# Gate 6D2 governed foundation-model validation

## Status

Gate 6D2 is authorized for execution under the closed Gate 6D1 contract. This document defines the executable boundary before validation evidence is generated.

## Objective

Gate 6D2 tests whether three pinned public time-series foundation models can improve one-step-ahead industrial energy-demand forecasting relative to the frozen V1 histogram-gradient-boosting incumbent.

The candidates are benchmarks rather than presumed improvements:

- `chronos_2_zero_shot`;
- `timesfm_2_5_zero_shot`;
- `moirai_2_research_zero_shot`.

Moirai 2.0 remains research-only because its weights use CC-BY-NC-4.0. It cannot become commercially promotion eligible regardless of its validation metrics.

## Environment isolation

Each candidate executes in a separate Python virtual environment. This is required because the pinned Uni2TS source constrains Torch and NumPy below the versions used by the Chronos-2 and TimesFM lanes.

The environments are created serially, recorded through `pip freeze --all`, and deleted after their candidate finishes. Only their environment locks and governed evidence enter the final package.

The source implementations are installed from exact Git commits:

- Chronos: `amazon-science/chronos-forecasting@7dc4435706a4454feb79df44ca9f33631f3027bf`;
- TimesFM: `google-research/timesfm@3dae50b20d7a724981e8ea36cda75578f80dd2dc`;
- Moirai: `SalesforceAIResearch/uni2ts@cfd46d4510ed8896f263116f32928eede05b0a75`.

Model snapshots are downloaded only from the exact Hugging Face revisions and are rejected unless `model.safetensors` matches the Gate 6D1 SHA-256 identity.

## Forecast design

The comparison uses:

- source series: `usage_kwh`;
- forecast target: `usage_kwh_t_plus_1`;
- 15-minute cadence;
- 672-interval causal context, equal to seven days;
- one-interval horizon;
- zero-shot univariate inference;
- a common maximum batch size of 16 inside the Gate 6D1 limit of 32;
- 0.10, 0.50, and 0.90 quantiles;
- the 0.50 quantile as the authoritative point forecast;
- four expanding-window outer folds;
- four-interval purge;
- 7,004 validation origins per candidate.

Quantile outputs are retained as diagnostic evidence only. Probabilistic calibration and interval promotion belong to Gate 6E.

## Candidate adapters

### Chronos-2

Chronos-2 runs through `Chronos2Pipeline` on CPU. Cross-learning, covariates, multivariate input, sampling, adaptation, and long-horizon unrolling are disabled. The adapter extracts the model-native 0.10, 0.50, and 0.90 quantiles.

### TimesFM 2.5

TimesFM 2.5 runs through its PyTorch implementation with compilation disabled. Inputs are normalized, the continuous quantile head is enabled, nonnegative inference is enabled for the nonnegative load series, and quantile crossing is corrected. The model-native median is used as the point forecast.

### Moirai 2.0

Moirai 2.0 runs through `Moirai2Module` and `Moirai2Forecast` with direct tensor inference. It receives only the causal univariate target history and emits the model-native required quantiles. No GluonTS evaluation metric or automatic model-selection routine is used.

## Evidence generated

The final package contains:

- `model_provenance_manifest.json`;
- `candidate_results.csv`;
- `outer_fold_results.csv`;
- `out_of_fold_predictions.parquet`;
- `resource_evidence.csv`;
- `failure_records.json`;
- `environment_locks/*.txt`;
- `promotion_recommendation.json`;
- `gate_6d_execution_manifest.json`.

The package records model and source revisions, weight hashes, licenses, prediction counts, aggregate and peak-state metrics, temporal stability, inference latency, memory, runtime, environment identity, deterministic replay, and the nonbinding promotion recommendation.

## Failure behavior

A candidate fails closed when:

- its resolved model revision differs from the contract;
- its weight hash differs;
- its installed source revision is not the pinned commit;
- it exceeds download, memory, or runtime limits;
- a causal window is incomplete;
- a prediction or metric is nonfinite;
- required quantiles cross;
- deterministic replay fails;
- prediction evidence is incomplete;
- the locked boundary is approached or crossed.

No fallback model, alternative revision, reduced context, different batch protocol, adaptation, or post-failure rescue is permitted inside Gate 6D2.

## Governance boundary

Gate 6D2 prohibits:

- locked-test access;
- locked-prediction parsing;
- confirmatory evaluation;
- fine-tuning, LoRA, or other parameter adaptation;
- calibration;
- hyperparameter or context search;
- covariate or multivariate rescue lanes;
- hosted or paid APIs;
- automatic promotion;
- mutation of V1, Gate 6B, or Gate 6C evidence.

A successful execution produces validation evidence only. Gate 6D3 remains mandatory for the human decision and formal Gate 6D closure.
