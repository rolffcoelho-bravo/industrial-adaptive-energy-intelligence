# Gate 6D3 foundation-model closure

## Status

Gate 6D is closed. Chronos-2, TimesFM 2.5, and research-only Moirai 2.0 are rejected for promotion. The frozen V1 histogram-gradient-boosting model remains the reference champion. Gate 6E is the next permitted gate.

## Objective and scope

Gate 6D tested whether three pinned public time-series foundation models could improve one-step-ahead industrial energy-demand forecasting under the frozen V2 governance framework. Gate 6D3 records the required human decision after the completed Gate 6D2 training-and-validation-only execution.

No model was downloaded, loaded, refitted, calibrated, searched, or re-evaluated in Gate 6D3. No prediction was regenerated. No metric was recalculated from the locked test. No locked-test partition or locked prediction artifact was accessed.

## Governed evidence reviewed

The decision uses the complete Gate 6D2 evidence package:

- three exact pinned public model revisions;
- three exact source revisions;
- verified model-weight SHA-256 identities;
- three isolated execution environments;
- zero-shot univariate inference only;
- common context of 672 fifteen-minute intervals;
- one-interval forecast horizon;
- four expanding-window outer folds;
- four-interval purge;
- 7,004 validation origins per candidate;
- 21,012 out-of-fold prediction rows;
- complete aggregate, peak-state, fold, resource, portability, diagnostic-quantile, failure, environment, and lineage evidence.

## Results

| Candidate | Mean MAE | MAE change relative to V1 | Mean peak MAE | Peak change relative to V1 | Positive folds | Decision |
|---|---:|---:|---:|---:|---:|---|
| Chronos-2 | 4.743581 | 18.60% worse | 21.293473 | 15.69% worse | 0 of 4 | Reject |
| Moirai 2.0 | 5.184533 | 29.62% worse | 21.603820 | 17.38% worse | 0 of 4 | Reject |
| TimesFM 2.5 | 5.374107 | 34.36% worse | 21.467849 | 16.64% worse | 0 of 4 | Reject |

Every candidate failed the frozen aggregate-MAE, peak-state, positive-fold, and maximum single-fold degradation requirements. No candidate was promotion eligible.

Chronos-2 was the strongest foundation-model challenger, but it remained materially weaker than V1 and failed the exact deterministic replay control. TimesFM 2.5 had the weakest aggregate result and the highest model-size, memory, latency, and runtime burden. Moirai 2.0 was the smallest and fastest candidate, but its predictive result was materially weaker and its CC-BY-NC-4.0 weight license independently prohibits commercial promotion.

## Human decision

The approved human decision is:

1. reject `chronos_2_zero_shot` for promotion;
2. reject `timesfm_2_5_zero_shot` for promotion;
3. reject `moirai_2_research_zero_shot` for promotion and retain it only as a research negative benchmark;
4. retain `v1_frozen_champion` as the reference champion;
5. close Gate 6D;
6. unblock Gate 6E probabilistic optimization and uncertainty calibration.

The decision is recorded in `outputs/v2/gate_6d/promotion_decision.json`. Automatic promotion remains prohibited.

## Governance verification

The closure preserves the following boundaries:

- maximum prediction origin: 28,027;
- maximum target dependency: 28,028;
- locked-test start: 28,032;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- fine-tuning: false;
- calibration: false;
- hyperparameter search: false;
- rescue lane: false;
- probabilistic authority claimed: false;
- V1 mutation: false;
- automatic promotion: false;
- human decision authority preserved: true;
- Moirai commercial-license boundary preserved: true.

The closure manifest binds the decision to the exact Gate 6D2 candidate, fold, prediction, resource, provenance, failure, recommendation, and environment hashes.

## Technical assessment

The governed zero-shot foundation-model lane did not improve the present forecasting system. The result is temporally consistent because every candidate lost all four outer folds. It is not an isolated bad-period result.

The current objective is a short one-step horizon with a moderate sample and strong engineered lag, rolling, calendar, power-factor, and operational predictors. The V1 histogram-gradient-boosting model can use that structured information directly. The frozen foundation-model lane received only the univariate load history and did not overcome the feature-engineered incumbent.

This conclusion is deliberately narrow. It does not establish that foundation models are generally unsuitable for industrial energy systems. It establishes that the three pinned models, exact revisions, seven-day context, zero-shot univariate mode, and one-step objective evaluated in Gate 6D do not justify replacing V1.

Different contexts, multivariate series, exogenous covariates, fine-tuning, alternative model revisions, longer horizons, or additional foundation-model families are new research questions. They cannot be introduced as rescue mechanisms or used to reinterpret the closed Gate 6D result.

## Research and engineering value

Gate 6D strengthens the repository because it demonstrates evidence-based rejection of high-profile public models rather than presumed improvement. The repository now contains:

- exact external-weight and source provenance;
- isolated multi-framework execution environments;
- public-weight forecasting without hosted APIs;
- chronological and locked-boundary enforcement;
- complete memory, latency, runtime, model-size, and cost evidence;
- explicit commercial-license controls;
- preserved native diagnostic defects without post-hoc repair;
- a human rejection decision tied to immutable evidence hashes.

A defensible conclusion is:

> Under a governed one-step industrial-energy forecasting protocol, zero-shot Chronos-2, TimesFM 2.5, and Moirai 2.0 did not overcome a feature-engineered histogram-gradient-boosting incumbent. The performance gap persisted across all four chronological folds, while the foundation models imposed materially greater inference burden.

## Gate 6E boundary

Gate 6E is unblocked, but no uncertainty model, calibration method, optimization routine, interval claim, or promotion decision is authorized by Gate 6D3 alone.

Gate 6E must begin with a separate prespecified contract that freezes:

- the uncertainty objective;
- admissible probabilistic methods;
- calibration partitions and chronology;
- coverage, width, sharpness, and peak-state diagnostics;
- optimization objectives and hard constraints;
- calibration and selection budgets;
- model and interval promotion rules;
- locked-test exclusion;
- human decision authority.

Gate 6F and Gate 6G remain blocked until their required predecessors close.

## Closure

Gate 6D is closed. The frozen V1 histogram-gradient-boosting model remains the reference champion. Gate 6E is the next permitted gate.
