# Gate 6C governed neural forecasting

## Purpose

Gate 6C evaluates whether bounded neural architectures can improve one-step-ahead industrial energy forecasting while preserving the frozen Gate 6A optimization rules, the closed Gate 6B decision, and the immutable V1 confirmatory boundary.

This gate produces validation evidence only. It does not reopen the V1 locked test and does not support a production, savings, drift, causal, optimization-impact, or confirmatory claim.

## Candidate families

Three prespecified neural families are evaluated:

1. **Residual MLP**: a feed-forward neural baseline using the frozen V1 current-origin feature policy.
2. **Causal TCN**: a causal temporal convolutional network using a trailing multivariate sequence and the current-origin feature vector.
3. **GRU sequence model**: a gated recurrent unit network using the same trailing sequence and current-origin feature vector.

Each family has exactly two approved configurations. No parameter may be added, removed, or changed after execution begins.

## Sequence contract

The sequence layer uses 16 trailing 15-minute intervals, equivalent to four hours. The prediction origin is included as the final sequence position. Future rows are prohibited.

Sequence channels are:

- energy usage;
- lagging reactive power;
- leading reactive power;
- carbon dioxide intensity;
- lagging power factor;
- leading power factor.

Sequence-channel standardization is fitted only on rows available to the applicable training partition. Origins without complete history are excluded from model fitting. All governed outer validation origins have complete history.

## Static feature contract

The current-origin feature vector uses the frozen V1 model feature policy. Numeric imputation, scaling, categorical imputation, and one-hot encoding are fitted only on the applicable training partition. Target columns and future timestamps are prohibited.

## Nested chronological selection

The evaluation uses:

- four expanding-window outer folds;
- three chronological inner folds within each outer training partition;
- a four-interval purge;
- one deterministic selection seed for choosing between the two configurations in each family;
- five governed stochastic seeds for final outer-fold evaluation of the selected configuration.

The inner folds select a configuration independently for each outer fold. The selected configuration is then refitted on the full outer training partition for each of the five governed seeds.

## Training controls

All neural training is CPU bounded and GitHub native.

- framework: PyTorch;
- loss: mean absolute error through L1 loss;
- optimizer: AdamW;
- maximum epochs: 6;
- batch size: 1,024;
- internal early stopping: disabled;
- gradient clipping: 1.0;
- CPU threads: 1;
- deterministic PyTorch algorithms: required;
- target normalization: training-partition standardization with inverse transformation before evaluation.

The five governed seeds are `20260721`, `20260722`, `20260723`, `20260724`, and `20260725`. Seed substitution is prohibited.

## Evidence produced

Gate 6C records:

- inner-fold configuration-selection results;
- outer-fold and seed-level metrics;
- across-seed mean, standard deviation, minimum, and maximum;
- aggregate and peak-state MAE;
- temporal-fold dispersion;
- model size;
- p95 inference latency per 1,000 rows;
- peak memory and wall-clock evidence;
- save-and-reload portability checks;
- out-of-fold predictions for every family, fold, seed, and validation origin;
- schema-validated objective and trial records;
- a constrained Pareto classification;
- a machine-readable promotion recommendation;
- a final human promotion, rejection, or deferral decision.

## Promotion boundary

A neural challenger can be recommended for human consideration only when it:

- improves mean validation MAE by at least 1% relative to the frozen V1 champion;
- improves at least three of four outer folds;
- does not degrade any single fold by more than 2%;
- does not degrade peak-state MAE by more than 1%;
- provides all five governed seeds;
- passes all chronology, leakage, artifact, finite-evidence, and portability constraints;
- remains Pareto eligible across accuracy, robustness, seed stability, temporal stability, complexity, and latency.

Automatic promotion is prohibited. Final authority remains human.

## Locked boundary

The maximum admissible Gate 6C prediction origin is 28,027. The maximum admissible target dependency is 28,028. The V1 locked test begins at 28,032.

Gate 6C must not:

- access the locked-test partition;
- parse locked prediction rows;
- invoke the retired evaluator;
- change the target, peak threshold, folds, purge, metrics, or V1 model;
- mutate any V1 or closed Gate 6B artifact;
- describe validation results as confirmatory evidence.

## Execution path

The dedicated GitHub Actions workflow installs the CPU-only PyTorch package, verifies contracts and immutable boundaries, builds the governed Silver layer, executes the neural validation when committed evidence is absent, validates all generated artifacts, publishes a summary, and commits the first complete evidence package to the pull-request branch.

Gate 6D remains blocked until Gate 6C validation is complete, a human decision is recorded, the closure manifest is validated, and all required workflows are green.
