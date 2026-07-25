# Validation gates

The project advances through explicit evidence contracts. Each stage must pass its technical controls before dependent stages are enabled. A later gate cannot reopen or silently mutate a closed earlier gate.

## V1 governed evidence sequence

| Gate | Technical decision | Status | Required evidence |
|---:|---|---|---|
| 0 | Repository identity and contracts | Closed | CI, policies, schemas, public-content controls |
| 1 | Data source and immutable snapshot | Closed | Official UCI source, manifest, hash, license, data-quality checks |
| 2 | Forecast target and leakage boundary | Closed | Target schema, chronology ownership, leakage tests |
| 3 | Governed Silver analytical layer | Closed | Typed table, effective timestamps, feature availability, parity checks |
| 4A-4D | Chronological validation, candidate ladder, and model freeze | Closed | Expanding-window folds, benchmarks, candidate evidence, frozen model contract |
| 4E | Single authorized confirmatory evaluation | Closed | One untouched test execution and immutable terminal evidence |
| 4F | Confirmatory closure | Closed | Closure manifest, retired evaluator, prohibited second evaluation |
| 5A | Reporting alignment | Closed | Evidence-aligned narrative, section contract, model boundaries |
| 5B | Deterministic reporting evidence | Closed | Final tables, lineage, hashes, manifest |
| 5C | Governed visual evidence | Closed | Five approved figures, visual identities, green CI |
| 5D1-5D2 | GitHub-native LaTeX reporting implementation | Closed | Governed source, build workflow, structural validation |
| 5D3 | Human visual approval | Closed | Five-page 200 DPI review and approval record |
| 5D4 | Formal reporting closure | Closed | Canonical PDF identity, visual fingerprints, closure schema and manifest |
| 5E | Frozen V1 release closure | Pending | Release manifest, immutable tag, GitHub Release, canonical PDF asset, final verification |

## Closed confirmatory boundary

The V1 model and confirmatory evidence are immutable. The following actions are prohibited:

- model re-estimation against the locked period;
- a second locked-test evaluation;
- threshold, temporal-block, subgroup, or metric changes;
- parsing locked prediction rows for new exploratory analysis;
- reactivation of the retired evaluator;
- unsupported structural-drift, optimization, savings, causal, or live-production claims.

## Reporting closure boundary

Gate 5D4 freezes:

- the LaTeX source;
- the exact approved PDF identity;
- five rendered-page fingerprints;
- the report payload;
- five figures and five reporting tables;
- section order, conclusions, model boundaries, and approval evidence.

The exact approved PDF remains the only admissible V1 release asset. A later timestamped LaTeX rebuild may be accepted only as a visual-equivalence check and cannot replace the frozen canonical binary.

## Next permitted decision

The only open V1 gate is **Gate 5E: frozen V1 release closure**.

V2 Gate 6A is defined but blocked until Gate 5E is closed. No V2 architecture, optimization, uncertainty, GenAI, cloud-adapter, or human-decision work may be represented as active V1 evidence.
