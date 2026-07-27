# Gate 6D2 governed foundation-model validation results

## Status

Gate 6D2 validation is complete. The evidence is awaiting the mandatory Gate 6D3 human promotion decision. Gate 6D remains open and Gate 6E remains blocked.

## Objective and scope

Gate 6D2 tested whether three pinned public time-series foundation models could improve one-step-ahead industrial energy-demand forecasting relative to the frozen V1 histogram-gradient-boosting incumbent.

The evaluated candidates were:

- Chronos-2;
- TimesFM 2.5;
- Moirai 2.0 as a research-only benchmark.

The execution used zero-shot univariate inference only, a common context of 672 fifteen-minute intervals, a one-interval horizon, four expanding-window outer folds, a four-interval purge, and 7,004 validation origins per candidate.

The locked test was not accessed. No model was fine-tuned, calibrated, searched, ensembled, or adapted.

## Validation results

| Candidate | Mean MAE | Change relative to V1 | Mean peak MAE | Peak change relative to V1 | Positive folds | Promotion eligible |
|---|---:|---:|---:|---:|---:|---|
| Chronos-2 | 4.743581 | 18.60% worse | 21.293473 | 15.69% worse | 0 of 4 | No |
| Moirai 2.0 | 5.184533 | 29.62% worse | 21.603820 | 17.38% worse | 0 of 4 | No |
| TimesFM 2.5 | 5.374107 | 34.36% worse | 21.467849 | 16.64% worse | 0 of 4 | No |

Chronos-2 was the strongest foundation-model challenger, but it underperformed the frozen V1 incumbent materially on aggregate and peak-state MAE. Every candidate lost all four outer folds and exceeded the maximum permitted single-fold degradation.

The formal recommendation contains no eligible candidate.

## Temporal evidence

Chronos-2's relative fold degradations were:

- fold 1: 18.76%;
- fold 2: 14.57%;
- fold 3: 20.60%;
- fold 4: 21.05%.

Moirai 2.0 degraded aggregate MAE by 25.97% to 33.19% across the four folds. TimesFM 2.5 degraded aggregate MAE by 29.60% to 37.91%.

The result is therefore not driven by one isolated temporal segment. The incumbent remained stronger throughout the complete governed validation period.

## Resource and portability evidence

| Candidate | Weight size | Peak memory | p95 latency per 1,000 rows | Total wall clock |
|---|---:|---:|---:|---:|
| Moirai 2.0 | 45.56 MB | 924.32 MB | 5,672.30 ms | 42.80 s |
| Chronos-2 | 477.93 MB | 1,022.97 MB | 45,894.40 ms | 322.78 s |
| TimesFM 2.5 | 925.18 MB | 2,181.29 MB | 97,077.77 ms | 662.33 s |

All three candidates completed inside the frozen memory, download-size, runtime, batch-size, and CPU limits. No hosted or paid API was used. External API cost was zero.

Moirai 2.0 was the smallest and fastest candidate by a substantial margin, but its predictive result was materially weaker than V1 and its CC-BY-NC-4.0 weight license independently prohibits commercial promotion.

TimesFM 2.5 had the largest weight package, highest peak memory, highest latency, longest runtime, and weakest aggregate MAE.

## Diagnostic quantile evidence

Gate 6D uses the model-native median as the authoritative point forecast. The 0.10 and 0.90 quantiles remain diagnostic because probabilistic calibration belongs to Gate 6E.

Native quantile crossing was observed in:

- Chronos-2: 2 of 7,004 origins, or 0.0286%;
- Moirai 2.0: 127 of 7,004 origins, or 1.8132%;
- TimesFM 2.5: 0 of 7,004 origins.

The raw quantiles were preserved. They were not sorted, calibrated, replaced, or used to modify the point forecast. This avoids introducing an unapproved post-execution correction.

Chronos-2 also failed the exact deterministic replay control on the bounded replay sample. Its validation metrics remain admissible benchmark evidence, but this failure independently prevents promotion.

## Provenance verification

All three candidates passed:

- exact model-revision verification;
- exact model-weight SHA-256 verification;
- exact source-revision verification;
- license verification;
- local public-weight execution;
- remote-code-trust prohibition;
- hosted-API prohibition;
- paid-API prohibition;
- CPU execution;
- chronology and locked-boundary checks.

The candidate environments were isolated and recorded separately because the pinned source packages have incompatible Torch, NumPy, and pandas requirements.

## Governance verification

The final manifest records:

- three successful candidates;
- 21,012 total prediction rows;
- 7,004 validation origins per candidate;
- maximum prediction origin: 28,027;
- maximum target dependency: 28,028;
- common context: 672 intervals;
- forecast horizon: one interval;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- fine-tuning: false;
- calibration: false;
- hyperparameter search: false;
- automatic promotion: false;
- V1 immutable: true;
- next required gate: Gate 6D3.

## Technical assessment

The zero-shot foundation-model lane did not add predictive value for the present industrial-energy forecasting objective.

The result is consistent with the evidence from Gates 6B and 6C. The current problem has a short one-step horizon, a moderate sample size, strong engineered lag and operational predictors, and a tabular incumbent specifically validated for this data-generating environment. The foundation models receive only the raw univariate load history and cannot exploit the incumbent's engineered operational structure.

This does not establish that foundation models are generally unsuitable for industrial energy systems. It establishes that the three pinned zero-shot models, under a common seven-day context and one-step objective, do not justify replacing the incumbent.

The result also clarifies the likely boundary of foundation-model value. Their strengths may be more relevant when:

- the horizon is longer;
- multiple related series provide cross-learning value;
- covariates are supported under a prespecified protocol;
- adaptation is justified by a separate training-only design;
- transfer across facilities or domains is the primary research question;
- probabilistic calibration is evaluated directly.

Those are new research questions. They cannot be introduced as rescue mechanisms inside Gate 6D.

## Research and repository value

Gate 6D2 materially strengthens the repository despite the negative promotion result. It demonstrates:

- exact external-weight and source provenance;
- isolated multi-framework execution environments;
- public foundation-model benchmarking without hosted APIs;
- chronological and locked-boundary enforcement;
- complete latency, memory, runtime, and model-size evidence;
- explicit commercial-license controls;
- preservation of native diagnostic defects rather than post-hoc correction;
- evidence-based rejection of fashionable zero-shot models.

A defensible research conclusion is:

> Under a governed one-step industrial-energy forecasting protocol, zero-shot Chronos-2, TimesFM 2.5, and Moirai 2.0 did not overcome a feature-engineered histogram-gradient-boosting incumbent. The gap persisted across all four chronological folds, while the foundation models imposed materially higher inference cost.

## Recommendation

Gate 6D3 should:

1. reject Chronos-2 for promotion;
2. reject TimesFM 2.5 for promotion;
3. retain Moirai 2.0 as a research-only negative benchmark and reject it for promotion;
4. retain the frozen V1 histogram-gradient-boosting model as the reference champion;
5. close Gate 6D;
6. permit Gate 6E probabilistic optimization and uncertainty calibration.

No candidate should be rescued inside Gate 6D through a different context, multivariate inputs, covariates, fine-tuning, calibration, alternative revisions, or additional model families.

## Evidence package

- `outputs/v2/gate_6d/candidate_results.csv`;
- `outputs/v2/gate_6d/outer_fold_results.csv`;
- `outputs/v2/gate_6d/out_of_fold_predictions.parquet`;
- `outputs/v2/gate_6d/resource_evidence.csv`;
- `outputs/v2/gate_6d/model_provenance_manifest.json`;
- `outputs/v2/gate_6d/environment_locks/`;
- `outputs/v2/gate_6d/failure_records.json`;
- `outputs/v2/gate_6d/promotion_recommendation.json`;
- `outputs/v2/gate_6d/gate_6d_execution_manifest.json`.

Gate 6D2 is complete. Gate 6D3 remains mandatory for the human decision and formal closure.
