# Visualization standard

The project uses Matplotlib as an evidence-governance layer. Every figure must answer a specific forecasting, validation, data, or model-risk question.

## Visual doctrine

- Light background only.
- Editorial, institutional composition suitable for a five-page technical brief.
- One dominant message per figure.
- Real observations and final machine-readable evidence only.
- No synthetic demonstrations, placeholder curves, or visually implied results.
- Direct labels are preferred over large legends.
- Units, sample period, source, figure identifier, and evidence identity are mandatory.
- Candidate and reference values must remain visually distinguishable.
- Locked metrics must reconcile exactly with the committed Gate 5B tables.
- No personal name or publisher label appears inside figures.
- Same-environment repeated rendering must reproduce identical PNG bytes.
- Cross-platform byte identity is not claimed.

## Permitted evidence

The renderer may read only the five committed Gate 5B tables and `outputs/reporting_evidence_manifest.json`.

It must not read locked prediction rows, reconstruct forecasts, fit models, estimate thresholds, define new subgroups, or invoke the retired evaluator.

## Required figures

### Confirmatory forecasting verdict

Candidate and persistence MAE, aggregate and peak-state improvement, and the four prespecified temporal blocks.

### Governed data architecture

Official source provenance, immutable raw evidence, source-aware chronology, governed Silver construction, and deterministic reporting evidence.

### Model ladder and chronological validation

The formal reference, rejected linear candidates, promoted nonlinear candidate, chronological validation, and validation-only selection.

### Locked-test temporal stability

Candidate and persistence MAE across four equal prespecified blocks, peak-state performance, and the exact evaluation boundary.

### Evidence governance and model boundaries

Gate lineage, artifact identity, single-evaluation controls, and claims outside the validated scope.

## Publication controls

A figure passes only when its inputs are committed Gate 5B artifacts, its annotations reconcile with those inputs, its dimensions satisfy the contract, and it has been inspected at full size and PDF size.

Figures must not imply structural drift conclusions, optimization recommendations, savings, causal effects, proprietary operations, or live production performance.
