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
- The most attractive model is never highlighted unless it passes the locked validation contract.

## Five strategic brief figures

### Page 1: Executive decision timeline

A three-layer figure combining actual demand, forecast and interval, peak-risk probability, and the resulting no-action/watch/adaptation state. It should let a senior reviewer understand the decision within seconds.

### Page 2: Industrial load profile

A temporal load heatmap with marginal intraday and weekday summaries, plus concise data-quality evidence. The objective is to expose structure, concentration, seasonality, and coverage rather than show a generic time-series plot.

### Page 3: Model validation dashboard

A chronological-fold performance matrix, benchmark comparison, peak-state metric panel, and calibration evidence. Average accuracy, peak performance, and worst-fold robustness must be visible together.

### Page 4: Drift and optimization dashboard

A synchronized drift-score timeline and global-versus-local disagreement panel, followed by the constrained cost/peak/disruption frontier. The selected operating point must be distinguished from feasible alternatives and from the no-action state.

### Page 5: Business impact and governance

An assumption-bounded value bridge or scenario range combined with evidence lineage and approval boundaries. It must separate measured results, derived estimates, assumptions, and decisions requiring human authorization.

## Quality controls

A chart is publication-ready only when:

1. its input file is final and machine-readable;
2. its labels contain no placeholder language;
3. its dimensions and resolution satisfy `configs/visualization_contract.yml`;
4. its numerical annotations reconcile with the source output table;
5. its source note and sample period are present;
6. its rendering is visually inspected at PDF size;
7. it does not imply access to proprietary company information.
