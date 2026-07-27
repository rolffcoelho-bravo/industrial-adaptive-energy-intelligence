# Gate 6D1 to Gate 6D2 transition validation

## Purpose

Gate 6D1 prohibited foundation-model execution before explicit authorization. After Gate 6D2 produced its complete governed evidence package, the Gate 6D1 predecessor validator required a transition-safe artifact rule.

## Transition rule

Before Gate 6D2 execution:

- `outputs/v2/gate_6d/` must contain no execution artifact;
- any partial or unauthorized artifact causes Gate 6D1 validation to fail.

After Gate 6D2 execution:

- `gate_6d_execution_manifest.json` must exist;
- the manifest must validate against the Gate 6D2 schema;
- status must be `validation_complete_pending_human_decision`;
- the candidate set must remain Chronos-2, TimesFM 2.5, and research-only Moirai 2.0;
- locked-test access, locked-prediction parsing, confirmatory evaluation, fine-tuning, calibration, hyperparameter search, and automatic promotion must remain false;
- V1 must remain immutable;
- the next gate must be Gate 6D3.

A directory containing partial execution artifacts without the complete manifest remains prohibited.

## Governance effect

This transition does not weaken Gate 6D1. It preserves the original implementation-only prohibition before authorization while allowing the closed predecessor validator to recognize complete, schema-validated Gate 6D2 evidence after authorized execution.

The same pattern prevents a closed design gate from incorrectly rejecting the evidence package produced by its authorized successor.
