# Target and leakage controls

## Contract status

Decision Gate 2 is locked and effective from 16 July 2026.

The contract fixes the supervised-learning questions before feature engineering or model selection begins. It does not claim predictive performance.

## Operational questions

The analytical system addresses two separate questions:

1. How much electricity will the next 15-minute interval require?
2. Is at least one materially high-load interval likely within the next 60 minutes?

## Regression target

| Field | Locked definition |
|---|---|
| Name | `usage_kwh_t_plus_1` |
| Source | `Usage_kWh` |
| Horizon | One observation, equal to 15 minutes |
| Rule | The next observed interval |
| Boundary | The final origin is unavailable and never imputed |

At prediction origin \(t\), the regression label is the observed `Usage_kWh` value at \(t+1\).

## Peak-risk target

| Field | Locked definition |
|---|---|
| Name | `peak_within_next_60_minutes` |
| Source | `Usage_kWh` |
| Horizon | Four observations, equal to 60 minutes |
| Event | Any of the next four usage values is at or above the locked threshold |
| Boundary | The final four origins are unavailable and never imputed |

The comparison is inclusive. A future observation equal to the threshold is a peak event.

## Peak-threshold governance

The primary threshold is the 90th percentile of `Usage_kWh` in the applicable training partition.

The threshold:

- is estimated from training observations only;
- is refitted independently for each chronological evaluation origin;
- cannot use validation or locked-test outcomes;
- uses linear quantile interpolation;
- is recorded in the fold manifest;
- is never estimated from the full sample.

A 95th-percentile severe-peak measure may be reported later as a secondary diagnostic. It does not replace the primary classification label without a new contract version.

## Prediction origin and information cutoff

The prediction origin is the end of interval \(t\), represented by `effective_timestamp`.

Information available at or before the origin may be used. Information first observed after the origin may not be used as a feature.

Permitted information includes current and historical electrical measurements, past-only statistics, deterministic calendar information, and calendar fields known for the target time.

## Prohibited leakage paths

The following operations are prohibited:

- random train/test splitting;
- future observations used as features;
- centered rolling windows;
- backward filling from future rows;
- full-sample scaling or encoding;
- full-sample peak-threshold estimation;
- validation or test outcomes used in feature selection;
- future load type or future electrical measurements;
- interpolation across a prediction boundary;
- locked-test outcomes used in model or calibration decisions.

## Temporal ownership

The provisional 60% development-training, 20% validation, and 20% locked-test chronological split remains in force.

Every learned transformation, threshold, selector, calibrator, and model must be fit inside the training side of its evaluation origin. The locked final test block cannot influence development decisions.

## Executable enforcement

The repository validates:

- the next-interval regression shift;
- the four-interval forward peak window;
- training-only threshold estimation;
- unavailable boundary targets;
- strict target timestamps after prediction origins;
- chronological row order;
- rejection of a contract that changes the locked 90th-percentile definition;
- prohibition of random splitting and full-sample preprocessing.

The feature-causality test that mutates future rows and verifies earlier features remain unchanged belongs to Decision Gate 3 because no feature universe exists before that gate.

## Gate boundary

Decision Gate 2 creates target definitions, validation rules, schema enforcement, and pure target-construction functions. It does not create Silver features, fit models, report metrics, or generate business-impact claims.

The next stage is Decision Gate 3: the Silver table, feature universe, and availability-at-origin matrix.
