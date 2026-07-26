# Gate 6C2 governed neural validation execution

## Status

Validation complete. The evidence is awaiting the mandatory Gate 6C3 human promotion decision. Gate 6D remains blocked.

## 1. Objective and scope

Gate 6C2 executed the prespecified validation-only comparison of compact N-HiTS, compact TiDE, and compact PatchTST against the frozen V1 histogram-gradient-boosting incumbent.

The execution used only training and validation partitions. It did not access the locked test, parse locked predictions, perform a confirmatory evaluation, mutate V1, or create a promotion decision.

## 2. Frozen execution design

- three candidate families;
- one configuration per family;
- five Gate 6A seeds per configuration;
- four expanding-window outer folds;
- four-interval purge;
- context length of 96 intervals;
- one-step forecast horizon;
- CPU-only canonical execution;
- serial execution;
- 40 epochs per candidate-seed-fold fit;
- batch size 256;
- no internal early stopping;
- no configuration search or protocol adaptation.

The completed execution contains 60 candidate-seed-fold evaluations and 7,004 validation origins per candidate and seed.

## 3. Validation results

| Candidate | Mean MAE | Aggregate change versus V1 | Mean peak MAE | Peak change versus V1 | Positive folds | Seed MAE SD | Eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| compact TiDE | 4.393078 | 9.84% degradation | 20.898733 | 13.55% degradation | 0 of 4 | 0.023005 | No |
| compact PatchTST | 4.708367 | 17.72% degradation | 21.534718 | 17.00% degradation | 0 of 4 | 0.124420 | No |
| compact N-HiTS | 5.054859 | 26.38% degradation | 23.286027 | 26.52% degradation | 0 of 4 | 0.093942 | No |

Every candidate failed the four predictive promotion requirements:

- minimum aggregate MAE improvement;
- peak-state MAE boundary;
- positive outer-fold count;
- maximum single-fold degradation.

All three passed chronology, resource, artifact, finite-evidence, and CPU-portability controls. Their rejection signal is therefore a predictive-evidence result rather than an execution or governance failure.

## 4. Resource and portability evidence

| Candidate | Mean model size | Maximum p95 latency per 1,000 rows | Maximum peak memory | Total candidate runtime |
|---|---:|---:|---:|---:|
| compact TiDE | 1,165,366 bytes | 7.617 ms | 1,857.676 MB | 584.218 seconds |
| compact PatchTST | 455,170 bytes | 60.281 ms | 1,857.707 MB | 3,579.967 seconds |
| compact N-HiTS | 2,583,763 bytes | 38.007 ms | 1,812.570 MB | 1,224.920 seconds |

TiDE was the strongest neural candidate on aggregate accuracy, seed stability, latency, and total runtime. PatchTST produced the smallest serialized models but had the longest runtime and highest latency. N-HiTS produced the largest models and the weakest predictive result.

## 5. Evidence and governance verification

- seed-fold evaluations: `60`;
- validation prediction rows: `105060`;
- validation origins per candidate and seed: `7004`;
- maximum prediction origin: `28027`;
- maximum target dependency: `28028`;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- automatic promotion: false;
- V1 immutable: true;
- execution failures: zero;
- next required gate: `6C3`.

The complete package contains seed results, fold aggregations, candidate leaderboard, out-of-fold predictions, trial evidence, failure records, environment lock, evidence hashes, execution manifest, and the deterministic nonbinding promotion recommendation.

## 6. Technical assessment

Gate 6C2 achieved its intended problem-solving purpose. It established whether compact neural architectures could improve the current one-step industrial energy forecast under the same chronology and evidence discipline as the incumbent. They did not.

The result is informative rather than negative repository value. It demonstrates that model sophistication is not treated as evidence of superiority. The frozen tabular incumbent remains materially stronger on both aggregate and peak-state validation performance.

TiDE's low across-seed MAE dispersion shows that its underperformance was stable rather than caused by one unfavorable initialization. PatchTST and N-HiTS also underperformed across every outer fold, so there is no evidence of a hidden temporal segment in which they consistently surpassed V1 under the frozen comparison.

## 7. Recommendation

Gate 6C3 should reject all three neural challengers for promotion and retain the frozen V1 histogram-gradient-boosting model as the reference champion.

No neural architecture should be rescued through post-execution tuning, altered context length, additional configurations, different seeds, or revised thresholds inside Gate 6C. Such changes would invalidate the prespecified comparison.

The next roadmap stage should remain Gate 6D, the governed time-series foundation-model benchmark, only after Gate 6C3 records the human decision and formally closes Gate 6C.

## 8. Research and repository value opportunities

The neural result creates three defensible future opportunities without changing Gate 6C evidence:

1. Add a post-roadmap architecture-context adequacy study that relates receptive field, periodicity, feature redundancy, and sample size to model performance. This could explain why the compact neural hypotheses failed rather than merely reporting that they failed.
2. Preserve the stability-adjusted Pareto surface proposed in Gate 6C1, combining predictive accuracy, peak robustness, seed dispersion, latency, memory, and model size. It should remain diagnostic and must not override frozen promotion constraints.
3. Give higher priority to the later probabilistic and uncertainty gates, where neural or foundation representations may add decision value through interval calibration or tail-risk estimation even when point MAE does not improve.

These extensions would increase methodological and industrial decision value, but none should retroactively modify Gate 6C2.

## 9. Next-gate decision

Gate 6C2 evidence is complete. Gate 6C3 is the next mandatory subgate and requires an explicit human decision to promote, reject, or defer each candidate. Gate 6D remains blocked until Gate 6C3 closes Gate 6C.
