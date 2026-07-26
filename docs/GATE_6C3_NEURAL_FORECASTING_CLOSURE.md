# Gate 6C3 neural forecasting closure

## Status

Gate 6C is closed. Compact N-HiTS, compact TiDE, and compact PatchTST are rejected for promotion. The frozen V1 histogram-gradient-boosting model remains the reference champion. Gate 6D is the next permitted gate.

## Objective and scope

Gate 6C tested whether three prespecified compact neural architectures could improve one-step-ahead industrial energy-demand forecasting under the frozen V2 governance framework. Gate 6C3 records the required human promotion decision after the Gate 6C2 training-and-validation-only execution.

No model was refitted in Gate 6C3. No metric was recalculated from locked evidence. No locked-test partition or locked prediction artifact was accessed.

## Governed execution reviewed

The decision uses the complete Gate 6C2 evidence package:

- three candidate families;
- one prespecified configuration per family;
- five governed seeds per configuration;
- four expanding-window outer folds;
- four-interval purge;
- 60 candidate-seed-fold evaluations;
- 7,004 validation origins per candidate and seed;
- 105,060 out-of-fold prediction rows;
- CPU-only canonical execution;
- complete performance, stability, resource, portability, failure, and lineage evidence.

## Results

| Candidate | Mean MAE | MAE change relative to V1 | Mean peak MAE | Peak change relative to V1 | Positive folds | Decision |
|---|---:|---:|---:|---:|---:|---|
| Compact TiDE | 4.393078 | 9.84% worse | 20.898733 | 13.55% worse | 0 of 4 | Reject |
| Compact PatchTST | 4.708367 | 17.72% worse | 21.534718 | 17.00% worse | 0 of 4 | Reject |
| Compact N-HiTS | 5.054859 | 26.38% worse | 23.286027 | 26.52% worse | 0 of 4 | Reject |

Every candidate failed the frozen aggregate-MAE, peak-state, positive-fold, and maximum single-fold degradation requirements. No candidate was promotion eligible.

TiDE was the strongest neural challenger. It also had the lowest across-seed MAE standard deviation and the lowest inference latency, but its stable underperformance did not justify promotion. PatchTST had the smallest serialized model but the longest total training time and highest inference latency. N-HiTS had the weakest predictive result and the largest serialized model.

## Human decision

The human decision is:

1. reject `tide_compact`;
2. reject `patchtst_compact`;
3. reject `nhits_compact`;
4. retain `v1_frozen_champion`;
5. close Gate 6C;
6. permit Gate 6D foundation-model benchmarking.

The decision is recorded in `outputs/v2/gate_6c/promotion_decision.json`. Automatic promotion remains prohibited.

## Governance verification

- maximum prediction origin: 28,027;
- maximum target dependency: 28,028;
- locked-test start: 28,032;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- V1 mutation: false;
- quarantined superseded evidence used: false;
- human decision authority preserved: true.

## Technical assessment

The compact neural lane did not solve the current forecasting problem better than the feature-engineered V1 incumbent. The one-step horizon, moderate sample size, and strong lag, rolling, calendar, power-factor, and operational features favor a governed tabular learner that can use structured predictors directly.

The repeated-seed result is especially informative. TiDE's low seed dispersion shows that its underperformance was stable rather than the consequence of one unfavorable initialization. The negative result therefore has stronger evidential value than a single-seed comparison.

This result does not establish that neural forecasting is unsuitable for industrial energy systems in general. It establishes that the three frozen compact architectures, configurations, context length, and one-step objective tested in Gate 6C do not justify replacing the incumbent.

## Research and engineering value

Gate 6C increases the repository's technical value because it demonstrates that advanced models are evaluated and rejected when the evidence does not support promotion. The repository now contains reproducible repeated-seed neural evidence, chronology-safe evaluation, peak-state robustness tests, efficiency measurements, portability controls, and a recorded human decision.

A useful later diagnostic extension could compare raw-sequence and engineered-feature inputs, alternative context lengths, multihorizon objectives, and regime-specific residuals. Such work must be prespecified as a separate extension and cannot rewrite Gate 6C findings.

## Recommendation for Gate 6D

Gate 6D should benchmark a small number of time-series foundation models as zero-shot or strictly governed adaptation challengers. It should freeze model identities, versions, licenses, context rules, external-weight provenance, inference cost, memory, latency, and adaptation permissions before execution.

The foundation-model lane must remain a benchmark rather than a presumed improvement. It must use training and validation evidence only, preserve the same chronological boundary, and require a separate human decision before any later stage.

## Closure

Gate 6C is closed. Gate 6D is unblocked. Gates 6E through 6G remain planned and cannot begin before their required predecessors close.
