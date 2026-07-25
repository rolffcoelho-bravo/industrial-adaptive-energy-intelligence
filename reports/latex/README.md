# Governed technical brief

This directory contains the publication source for the five-page Industrial Adaptive Energy Intelligence confirmatory evidence brief.

## Source of truth

The numerical evidence remains governed by:

- `outputs/report_payload.json`
- `outputs/modeling/locked_test_results.json`
- `outputs/modeling/locked_test_closure_manifest.json`
- the five approved figures under `outputs/charts/`

The LaTeX file is a presentation layer only. It must not recalculate locked metrics, parse locked prediction rows, fit models, invoke the retired evaluator, or introduce unsupported claims.

## GitHub-native build

The workflow `.github/workflows/build-brief.yml` compiles the LaTeX source in GitHub Actions using a containerized TeXLive environment. No local LaTeX installation is required.

The workflow:

1. verifies the governed payload and approved figures;
2. checks the five-page source contract and prohibited public claims;
3. compiles `industrial_adaptive_energy_intelligence_technical_brief.tex`;
4. verifies the PDF page count and metadata;
5. uploads the PDF, LaTeX source, payload, figures, and reporting tables as a GitHub Actions artifact;
6. attaches the PDF to a GitHub Release when the workflow is triggered by a published release.

## Report files

- Source: `reports/latex/industrial_adaptive_energy_intelligence_technical_brief.tex`
- Generated PDF: `reports/latex/industrial_adaptive_energy_intelligence_technical_brief.pdf`

The generated PDF is not committed during design review. It becomes a GitHub Actions artifact and, after release approval, a permanent GitHub Release asset.
