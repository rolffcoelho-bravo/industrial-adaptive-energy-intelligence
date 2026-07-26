# Gate 6C superseded execution quarantine

## Status

The closed branch `gate-6c-neural-forecasting`, Issue #11, and PR #12 are superseded and inadmissible for the approved Gate 6C protocol.

## What occurred

The superseded branch fitted residual MLP, causal TCN, and GRU challengers under a six-configuration, five-seed design before the approved Gate 6C1 implementation-only boundary was completed. It produced validation artifacts on the branch.

## Boundary verification

The superseded execution manifest records:

- locked-test access: false;
- locked-prediction parsing: false;
- confirmatory evaluation: false;
- maximum prediction origin: `28027`;
- maximum target dependency: `28028`;
- V1 immutable: true.

The failure is therefore procedural and methodological, not a locked-test or V1-integrity breach.

## Quarantine rule

The superseded artifacts must not:

- be merged into `main`;
- be used as Gate 6C evidence;
- influence N-HiTS, TiDE, or PatchTST configurations;
- influence seed selection, context length, resource limits, or promotion thresholds;
- be used for model promotion, reporting, ensemble design, or later confirmatory decisions;
- be represented as part of the approved research findings.

They remain only as an auditable record of a rejected implementation path.

## Approved replacement

Gate 6C now follows Issue #13 and PR #14:

1. Gate 6C1: implementation-only contracts, schemas, tests, and workflow boundaries, with no fitting;
2. Gate 6C2: compact N-HiTS, compact TiDE, and compact PatchTST using exactly the fixed seeds `20260725`, `20260726`, and `20260727`;
3. Gate 6C3: human promotion decision and formal closure.

Gate 6D remains blocked until Gate 6C3 closes Gate 6C.
