# Gate 6B advanced tabular challengers

**Status:** Approved for execution

## Objective

Evaluate prespecified advanced tabular regression challengers using training and validation evidence only, under the frozen Gate 6A architecture and optimization-governance contracts.

## Candidate families

- XGBoost histogram tree boosting
- LightGBM leaf-wise gradient boosting
- CatBoost ordered boosting

Each family uses a bounded, deterministic four-configuration search space. Every outer fold uses three chronological inner folds with a four-interval purge. Internal early stopping is prohibited so that all iteration counts are explicit and governed.

## Evidence boundary

- admissible partitions: training and validation;
- outer folds: four expanding-window folds inherited from the V1 chronology;
- inner folds: three expanding-window folds;
- purge: four intervals;
- locked-test access: prohibited;
- locked-test prediction parsing: prohibited;
- V1 artifact mutation: prohibited;
- search-space changes after execution begins: prohibited.

## Primary evidence

For each candidate family, Gate 6B records:

- selected configuration by outer fold;
- inner-validation MAE for every prespecified configuration;
- outer-fold MAE and peak-state MAE;
- relative performance against the frozen V1 histogram-gradient-boosting champion;
- temporal-fold dispersion;
- model size and inference latency;
- portability and hard-constraint checks;
- complete out-of-fold predictions limited to validation origins;
- deterministic artifact hashes and execution lineage.

## Promotion boundary

A challenger is promotion-eligible only when all Gate 6A hard constraints pass and the frozen promotion requirements are satisfied. Final promotion remains a human decision. Gate 6B may produce a governed recommendation, but it cannot access a confirmatory period or create a V2 confirmatory claim.

## Outputs

- `configs/advanced_tabular_contract.yml`
- governed search-space instances for all three algorithms
- advanced-tabular execution module and GitHub-native workflow
- inner-search, outer-fold, leaderboard, resource, and out-of-fold evidence
- schema-validated trial and promotion records
- Gate 6B closure manifest and public technical documentation

## Acceptance criteria

- all governed search spaces validate before execution;
- no more than 12 unique configurations are evaluated;
- every outer fold uses exactly three chronological inner folds;
- no prediction origin reaches the locked-test purge boundary;
- all generated metrics are finite and reproducible;
- complete resource and portability evidence is recorded;
- V1 immutability verification passes;
- repository CI and the Gate 6B execution workflow are green;
- no production, savings, drift, optimization-impact, causal, or confirmatory claim is made;
- Gate 6C remains blocked until Gate 6B is formally closed.
