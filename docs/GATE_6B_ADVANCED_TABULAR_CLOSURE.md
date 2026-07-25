# Gate 6B advanced tabular closure

## Decision

Gate 6B is closed.

The human promotion decision rejects all three advanced tabular challengers and retains the immutable V1 histogram-gradient-boosting model as the reference champion.

## Executed candidate families

The governed GitHub-native execution evaluated:

- XGBoost histogram boosting;
- LightGBM L1 boosting;
- CatBoost MAE boosting.

The search used exactly 12 prespecified configurations, four expanding-window outer folds, three chronological inner folds, a four-interval purge, and 7,004 validation origins per candidate.

## Validation evidence

| Candidate | Mean MAE | Relative MAE improvement | Peak-state MAE degradation | Positive outer folds | Promotion eligible |
|---|---:|---:|---:|---:|---|
| LightGBM L1 | 3.968592 | 0.78% | 1.33% | 4 | No |
| XGBoost histogram | 3.982222 | 0.44% | 1.81% | 2 | No |
| CatBoost MAE | 4.190236 | -4.76% | 8.09% | 0 | No |

LightGBM L1 was the strongest challenger. It did not meet the frozen minimum 1% aggregate validation improvement and exceeded the maximum permitted 1% peak-state degradation. XGBoost and CatBoost also failed one or more promotion requirements.

All three candidates passed the hard chronology, leakage, artifact, finite-evidence, and portability constraints. Rejection is therefore a promotion decision, not an execution failure.

## Frozen evidence boundary

- admissible partitions: training and validation only;
- maximum prediction origin: 28,027;
- maximum target dependency: 28,028;
- locked-test start: 28,032;
- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- V1 tag `v1.0.0`: immutable.

## Human authority

The Head of Research approved:

1. rejection of LightGBM L1, XGBoost histogram boosting, and CatBoost MAE for promotion;
2. retention of the frozen V1 incumbent;
3. formal closure of Gate 6B;
4. advancement to Gate 6C neural forecasting.

GenAI remained advisory and had no vote. Deterministic systems executed the prespecified trials and calculated the evidence. Final promotion authority remained human.

## Closure artifacts

- contract: `configs/advanced_tabular_contract.yml`;
- execution manifest: `outputs/v2/gate_6b/gate_6b_execution_manifest.json`;
- candidate leaderboard: `outputs/v2/gate_6b/candidate_leaderboard.csv`;
- trial evidence: `outputs/v2/gate_6b/trial_evidence.json`;
- out-of-fold predictions: `outputs/v2/gate_6b/out_of_fold_predictions.parquet`;
- human decision: `outputs/v2/gate_6b/promotion_decision.json`;
- closure manifest: `outputs/v2/gate_6b/gate_6b_closure_manifest.json`;
- closure schema: `schemas/gate_6b_closure_manifest.schema.json`.

## Claims boundary

Gate 6B provides governed validation evidence only. It does not support a production, savings, drift, causal, optimization-impact, or confirmatory claim.

## Next permitted gate

Gate 6C may evaluate neural forecasting candidates under new prespecified contracts, multiple governed seeds, chronological evidence, variance reporting, resource controls, portability evidence, and final human promotion authority.
