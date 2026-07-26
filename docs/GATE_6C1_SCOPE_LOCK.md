# Gate 6C1 scope lock

## Approved lineage

Gate 6C1 implements the neural-forecasting protocol tracked by Issue #13.

The only approved candidate families are:

- compact N-HiTS;
- compact TiDE;
- compact PatchTST.

The only approved seeds are:

- `20260721`;
- `20260722`;
- `20260723`;
- `20260724`;
- `20260725`.

Exactly one configuration per family and exactly three configurations in total are permitted.

## Required sequence

1. Gate 6C1 freezes contracts, schemas, implementation interfaces, tests, resource controls, and GitHub-native validation without fitting.
2. Gate 6C2 executes governed training-and-validation-only evidence after Gate 6C1 is merged, green, formally closed, and explicitly approved.
3. Gate 6C3 records the human promotion decision and closes Gate 6C.
4. Gate 6D remains blocked until Gate 6C3 closes Gate 6C.

## Superseded design

PR #12 and Issue #11 are superseded and closed. Their residual MLP, causal TCN, GRU, six-configuration, and alternative seed design is not part of the approved Gate 6C framework and must not be revived without formal change control.

## Gate 6C1 prohibitions

Gate 6C1 must not fit a neural model, generate validation predictions, calculate candidate performance, access the locked test, parse locked predictions, perform a confirmatory evaluation, mutate V1, or promote a model.

## Closure requirement

Gate 6C1 closes only after its contract, schemas, tests, workflow, V1 immutability verification, Gate 6B closure verification, public-content audit, dedicated workflow, and repository CI are green. The closure report must include detailed implementation results, governance evidence, technical assessment, recommendations, research-value opportunities, and the exact Gate 6C2 approval request.
