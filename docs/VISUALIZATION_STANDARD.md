# Visualization standard

The project uses Matplotlib as a decision-support layer, not as decoration. Every chart must answer a specific operating, modelling, governance, or business question.

## Visual doctrine

- Light background only.
- Editorial, institutional composition suitable for a technical decision brief.
- Complex figures are permitted only when multiple panels form one coherent evidence chain.
- One dominant message per figure.
- Real observations and validated model outputs only.
- No synthetic demonstrations, placeholder curves, or visually implied results.
- Direct labels are preferred over large legends.
- Units, sample period, source, figure identifier, and generation timestamp are mandatory.
- Uncertainty must be visible whenever a point prediction or scenario estimate is shown.
- Thresholds and intervention states must be explicitly labelled.
- A candidate model is highlighted only after it passes the locked validation contract.

## Required decision figures

### Executive decision timeline

A three-layer figure combining actual demand, forecast uncertainty, peak-risk probability, and the resulting no-action, watch, or adaptation state. The operating decision must be understandable within seconds.

### Industrial load profile

A temporal load heatmap with intraday and weekday summaries, plus concise data-quality evidence. The figure exposes structure, concentration, seasonality, and coverage rather than showing a generic time-series plot.

### Model validation dashboard

A chronological-fold performance matrix, benchmark comparison, peak-state metric panel, and calibration evidence. Average accuracy, peak performance, and worst-fold robustness must be visible together.

### Drift and optimization dashboard

A synchronized drift-score timeline and global-versus-local disagreement panel, followed by the constrained cost, peak, and disruption frontier. The selected operating point must be distinguished from feasible alternatives and the no-action state.

### Business impact and governance

An assumption-bounded value bridge or scenario range combined with evidence lineage and authorization boundaries. The figure separates measured results, derived estimates, assumptions, and decisions requiring human authorization.

## Quality controls

A chart passes the render gate only when:

1. its input file is final and machine-readable;
2. its labels contain no placeholder language;
3. its dimensions and resolution satisfy `configs/visualization_contract.yml`;
4. its numerical annotations reconcile with the source output table;
5. its source note and sample period are present;
6. its rendering is visually inspected at PDF size;
7. it does not imply access to proprietary company information.
