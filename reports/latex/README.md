# Governed technical brief

This directory contains the publication source for the five-page Industrial Adaptive Energy Intelligence confirmatory evidence brief.

## Source of truth

The numerical evidence remains governed by:

- `outputs/report_payload.json`
- `outputs/modeling/locked_test_results.json`
- `outputs/modeling/locked_test_closure_manifest.json`
- the five approved figures under `outputs/charts/`
- `outputs/reporting_closure_manifest.json`

The LaTeX file is a presentation layer only. It must not recalculate locked metrics, parse locked prediction rows, fit models, invoke the retired evaluator, or introduce unsupported claims.

## GitHub-native build

The workflow `.github/workflows/build-brief.yml` compiles the LaTeX source in GitHub Actions using a containerized TeXLive environment. No local LaTeX installation is required.

The workflow:

1. verifies the governed payload and approved figures;
2. validates the Gate 5D4 closure manifest;
3. checks the five-page source contract and prohibited public claims;
4. compiles `industrial_adaptive_energy_intelligence_technical_brief.tex`;
5. verifies PDF page count and project-level metadata;
6. retrieves the exact Gate 5D3 approved canonical PDF;
7. verifies visual equivalence between the rebuild and canonical PDF at 200 DPI;
8. uploads the canonical PDF, rebuild, LaTeX source, payload, figures, tables, and closure evidence as a GitHub Actions artifact;
9. attaches the exact approved canonical PDF to the GitHub Release when Gate 5E publishes V1.

## Report files

- Source: `reports/latex/industrial_adaptive_energy_intelligence_technical_brief.tex`
- Visual approval: `reports/latex/GATE_5D3_VISUAL_APPROVAL.md`
- Formal closure: `docs/GATE_5D4_REPORTING_CLOSURE.md`
- Closure manifest: `outputs/reporting_closure_manifest.json`
- Canonical release asset name: `industrial_adaptive_energy_intelligence_technical_brief.pdf`

## Canonical PDF identity

The approved canonical PDF is frozen with SHA-256:

```text
35e331e0349e0afca4aa8695a3f4aeafeffa18f83cdd9420876662bc6c782ba3
```

The PDF is retained in the approved GitHub Actions artifact during Gate 5D closure. Gate 5E publishes the exact artifact as a permanent GitHub Release asset. A later timestamped LaTeX rebuild cannot silently replace it, even when the rendered pages are visually identical.
