# Gate 6C2 governed neural validation execution

## Status

Approved for execution under the frozen Gate 6C1 contract. Gate 6C3 and Gate 6D remain blocked.

## Objective

Execute a prespecified validation-only comparison of compact N-HiTS, compact TiDE, and compact PatchTST against the frozen V1 histogram-gradient-boosting incumbent.

## Frozen execution design

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

The full execution contains 60 candidate-seed-fold evaluations and 7,004 validation origins per candidate and seed.

## Evidence produced

Gate 6C2 records:

- seed-level MAE and peak-state MAE;
- outer-fold means, standard deviations, minima, and maxima across seeds;
- aggregate candidate performance relative to V1;
- across-seed MAE and peak-MAE dispersion;
- outer-fold dispersion;
- model size;
- inference latency;
- peak memory;
- wall-clock time;
- CPU serialization portability;
- complete validation-only out-of-fold predictions;
- execution failures;
- dependency lock and evidence hashes;
- deterministic promotion eligibility and a nonbinding recommendation.

## Frozen promotion requirements

A candidate is eligible for human review only if it:

- improves aggregate validation MAE by at least 1 percent relative to V1;
- remains within 1 percent peak-state MAE degradation;
- improves aggregate MAE in at least three of four outer folds;
- avoids more than 2 percent MAE degradation in any outer fold;
- remains within the frozen across-seed MAE standard-deviation limit;
- passes chronology, locked-boundary, resource, artifact, and CPU-portability controls;
- remains Pareto eligible;
- receives a later explicit human decision in Gate 6C3.

Automatic promotion is prohibited.

## Evidence boundary

- training and validation partitions only;
- maximum prediction origin exclusive: `28028`;
- maximum target dependency exclusive: `28032`;
- locked-test access: prohibited;
- locked-prediction parsing: prohibited;
- confirmatory evaluation: prohibited;
- V1 mutation: prohibited.

## Sequence

Gate 6C2 may create validation evidence and a deterministic promotion recommendation. It must not create a promotion decision or close Gate 6C. Gate 6C3 remains the mandatory human decision and closure stage. Gate 6D remains blocked until Gate 6C3 closes Gate 6C.
