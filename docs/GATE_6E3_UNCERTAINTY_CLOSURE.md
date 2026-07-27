# Gate 6E3 governed uncertainty closure

## Decision

Gate 6E is closed with a human-authorized `no_action` outcome.

No uncertainty configuration is promoted. The frozen V1 histogram-gradient-boosting point model remains the authoritative predictive model. Gate 6E2 interval evidence is retained as validation evidence only and does not establish probabilistic authority for production or confirmatory use.

## Evidence considered

Gate 6E2 evaluated exactly nine frozen configurations over four chronological outer folds and 7,004 validation origins per configuration. The evidence package contains 63,036 interval-prediction rows and 70,935 chronology-safe calibration residual rows.

The strongest aggregate configuration was `aci_672_0p02`:

- weighted interval score: `2.2834847109806047`;
- peak-state weighted interval score: `9.55071623587265`;
- aggregate 90% coverage: `0.9003426613363792`;
- peak-state 90% coverage: `0.6058823529411764`.

Although it improved aggregate and peak-state weighted interval score relative to the expanding reference, it failed the frozen minimum aggregate peak-state 90% coverage requirement. Every other configuration also failed at least one hard constraint.

## Human decision

The final decision is:

- decision outcome: `no_action`;
- promoted uncertainty configuration: none;
- rejected uncertainty configurations: all nine;
- retained predictive model: `v1_frozen_champion`;
- automatic promotion: prohibited;
- probabilistic authority: not claimed.

The expanding configuration remains a validation reference only. The rolling and adaptive configurations remain negative or non-promoted validation evidence only. None may be introduced into a production, confirmatory, optimization, or ensemble layer without a new governed contract and new authorization.

## Immutable boundaries

Gate 6E3 performs no model fitting, calibration, interval generation, optimization execution, metric recalculation, locked-test access, locked-prediction parsing, or confirmatory evaluation.

The closure is bound to the exact Gate 6E2 evidence hashes for:

- calibration lineage;
- calibration residuals;
- configuration results;
- coverage results;
- environment lock;
- failure records;
- interval predictions;
- outer-fold results;
- promotion recommendation;
- resource evidence.

The V1 release remains immutable.

## Gate transition

Gate 6E is formally closed.

Gate 6F becomes the next permitted gate, but this closure does not authorize Gate 6F execution. Gate 6F must begin with a separately approved controlled-ensemble contract. Gate 6G remains blocked.
