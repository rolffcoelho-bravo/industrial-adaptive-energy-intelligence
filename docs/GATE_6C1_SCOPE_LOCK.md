# Gate 6C1 scope lock

## Approved lineage

Gate 6C1 implements the neural-forecasting protocol tracked by Issue #13, subject to reconciliation with the frozen Gate 6A optimization-governance contract.

The only approved candidate families are:

- compact N-HiTS;
- compact TiDE;
- compact PatchTST.

Exactly one configuration per family and exactly three configurations in total are permitted.

## Seed-governance block

The Gate 6C proposal specifies:

- `20260725`;
- `20260726`;
- `20260727`.

The frozen Gate 6A contract specifies five different governed seeds and prohibits substitution without versioned change control. Gate 6C1 therefore remains blocked pending an explicit seed-governance decision. The authoritative conflict record is `docs/GATE_6C1_SEED_GOVERNANCE_RECONCILIATION.md`.

## Required sequence

1. resolve the seed-governance conflict before any fitting;
2. Gate 6C1 freezes contracts, schemas, implementation interfaces, tests, resource controls, and GitHub-native validation without fitting;
3. Gate 6C2 executes governed training-and-validation-only evidence after Gate 6C1 is merged, green, and formally closed;
4. Gate 6C3 records the human promotion decision and closes Gate 6C;
5. Gate 6D remains blocked until Gate 6C3 closes Gate 6C.

## Superseded design

PR #12 and Issue #11 are superseded and closed. Their residual MLP, causal TCN, GRU, six-configuration, and five-seed design is not part of the approved candidate-family framework and must not be revived without formal change control.

## Gate 6C1 prohibitions

Gate 6C1 must not fit a neural model, generate validation predictions, calculate candidate performance, access the locked test, parse locked predictions, perform a confirmatory evaluation, mutate V1, or promote a model.

## Closure requirement

Gate 6C1 closes only after the seed conflict is resolved and its contract, schemas, tests, workflow, V1 immutability verification, Gate 6B closure verification, public-content audit, dedicated workflow, and repository CI are green. The closure report must include detailed implementation results, governance evidence, technical assessment, recommendations, originality opportunities, and the exact Gate 6C2 approval request.
