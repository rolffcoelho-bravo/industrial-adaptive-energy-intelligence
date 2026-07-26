# Gate 6C1 seed-governance reconciliation

## Status

Blocked pending an explicit methodological decision before any neural fitting.

## Conflict detected

The frozen Gate 6A optimization-governance contract requires:

- five stochastic seeds per neural configuration;
- the fixed seed set `20260721`, `20260722`, `20260723`, `20260724`, `20260725`;
- a minimum stochastic seed count of five;
- seed substitution prohibited;
- a new contract version and human approval for any pre-execution change.

The later Gate 6C protocol currently specifies three seeds:

- `20260725`;
- `20260726`;
- `20260727`.

The two contracts therefore disagree on seed count and seed identity.

## Governance consequence

Gate 6C1 cannot close and Gate 6C2 cannot begin until one of the following is explicitly approved:

1. restore the Gate 6A five-seed set for Gate 6C; or
2. approve a versioned Gate 6C change-control record that replaces the parent seed rule before execution.

No neural model has been fitted under PR #14. No validation predictions or metrics have been produced. The locked test remains excluded and V1 remains immutable.

## Technical assessment

The five-seed parent design is methodologically stronger. It provides more reliable across-seed variance estimates, preserves direct consistency with Gate 6A, and avoids introducing a special governance exception immediately before neural execution. The three-seed proposal reduces compute but weakens the stability evidence and requires an explicit versioned exception.

## Technical recommendation

Restore the five frozen Gate 6A seeds. The additional compute cost remains inside the existing Gate 6A neural budget, which already permits five seeds per configuration and up to 720 wall-clock minutes.

## Sequence protection

- Gate 6C1 remains open;
- Gate 6C2 remains blocked;
- Gate 6C3 remains blocked;
- Gate 6D remains blocked.
