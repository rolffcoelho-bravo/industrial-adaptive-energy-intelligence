# Five-page technical brief specification

## Page 1 - Executive decision

Business problem, system decision, three validated results, operational meaning, and boundaries.

## Page 2 - Data and Databricks architecture

Data provenance, quality results, Bronze/Silver/Gold flow, Python/SQL split, workflow evidence.

## Page 3 - Machine learning and chronological validation

Regression, classification, ensemble logic, benchmark table, peak-state performance, locked test.

## Page 4 - Structural drift and constrained optimization

Champion/challenger gate, drift states, promotion rule, objective function, constraints, no-action state.

## Page 5 - Governed agents and business impact

Agent roles, supervisor, permissions, evidence traceability, scenario-bounded value, production boundaries.

## Publication controls

- exactly five pages;
- no appendix inside the PDF;
- no placeholder language;
- no unsupported company-specific conclusion;
- every result linked to a machine-readable output;
- every value estimate accompanied by assumptions;
- visual render checked before release.

## Visual structure

Each page contains one strategically selected, publication-quality Matplotlib figure. The five required figures and their evidence roles are controlled by `configs/visualization_contract.yml`. Complexity is allowed only when aligned panels form a single decision chain; decorative complexity is prohibited.
