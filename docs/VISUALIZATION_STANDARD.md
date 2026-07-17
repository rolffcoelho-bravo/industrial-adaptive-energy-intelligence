# Visualization standard

The project uses Matplotlib as a decision-support and evidence-governance layer. Every chart must answer a specific forecasting, validation, data, or model-risk question.

## Visual doctrine

- Light background only.
- Editorial, institutional composition suitable for a five-page technical brief.
- One dominant message per figure.
- Real observations and final machine-readable model outputs only.
- No synthetic demonstrations, placeholder curves, or visually implied results.
- Direct labels are preferred over large legends.
- Units, sample period, source, figure identifier, and generation timestamp are mandatory.
- Candidate and reference values must remain visually distinguishable.
- Locked metrics must reconcile exactly with the terminal evidence.
- No personal name or project publisher label appears inside figures.

## Required decision figures

### Confirmatory forecasting verdict

A compact executive figure comparing candidate and persistence MAE, aggregate and peak-state improvement, and the four positive temporal-block results. It reports the closed verdict without implying operational intervention.

### Governed data architecture

A provenance and analytical-flow figure presenting the immutable raw snapshot, source-aware chronology, Silver layer, quality controls, modeling evidence, and portable and Databricks execution paths.

### Model ladder and chronological validation

A validation figure showing the benchmark and candidate ladder, fold-level performance, promotion decisions, and the boundary between validation-only selection and untouched confirmatory evaluation.

### Locked-test temporal stability

A confirmatory figure showing candidate and persistence MAE across the four prespecified temporal blocks, peak-state performance, and exact test boundaries.

### Evidence governance and model boundaries

A governance figure showing decision gates, execution identity, immutable artifact hashes, single-evaluation controls, and claims that remain outside the validated scope.

## Quality controls

A chart passes the render gate only when:

1. its input file is final and machine-readable;
2. its labels contain no placeholder language;
3. its dimensions and resolution satisfy `configs/visualization_contract.yml`;
4. every numerical annotation reconciles with the source artifact;
5. its source note and sample period are present;
6. its rendering is visually inspected at PDF size;
7. it does not imply access to proprietary company information;
8. it does not present drift, optimization, savings, or production claims without approved evidence.
