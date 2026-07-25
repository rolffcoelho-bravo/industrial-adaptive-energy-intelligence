# Gate 6A V2 architecture and optimization governance closure

**Status:** CLOSED

Gate 6A establishes the V2 architecture and optimization-governance boundary. It preserves the immutable V1 release and authorizes only future development under separately versioned contracts.

## Closed deliverables

- [`configs/v2_architecture_contract.yml`](../configs/v2_architecture_contract.yml)
- [`configs/optimization_governance.yml`](../configs/optimization_governance.yml)
- [`docs/V2_ARCHITECTURE.md`](V2_ARCHITECTURE.md)
- [`docs/OPTIMIZATION_GOVERNANCE.md`](OPTIMIZATION_GOVERNANCE.md)
- [`schemas/v2_architecture_contract.schema.json`](../schemas/v2_architecture_contract.schema.json)
- [`schemas/optimization_governance.schema.json`](../schemas/optimization_governance.schema.json)
- [`schemas/governed_search_space.schema.json`](../schemas/governed_search_space.schema.json)
- [`schemas/objective_record.schema.json`](../schemas/objective_record.schema.json)
- [`schemas/trial_evidence.schema.json`](../schemas/trial_evidence.schema.json)
- [`schemas/promotion_decision.schema.json`](../schemas/promotion_decision.schema.json)
- [`outputs/v2/gate_6a_architecture_manifest.json`](../outputs/v2/gate_6a_architecture_manifest.json)

## V1 baseline

```text
Release: v1.0.0
Commit: 9e8cfe567d2174639675cbc784f21fe968dafe92
State: immutable
```

CI verifies the frozen V1 paths against the release tag and validates governed hashes from the V1 release manifest.

## Architecture decision

The frozen V2 sequence is:

```text
Data Evidence
    -> Predictive Models
    -> Governed Model Optimization
    -> Uncertainty
    -> Robust Selection
    -> Generative Interpretation
    -> Human Decision
```

The analytical core is provider-neutral. GCP is the first planned cloud execution adapter, Google Workspace is restricted to collaboration and approval, and Databricks remains a separate enterprise execution path.

## Optimization decision

Gate 6A prespecifies:

- nested expanding-window validation;
- training and validation as the only admissible partitions;
- complete locked-test exclusion;
- objective and hard-constraint definitions;
- search budgets;
- fixed random seeds;
- stochastic variance reporting;
- out-of-fold-only ensemble evidence;
- separate uncertainty calibration;
- constrained Pareto selection;
- final human promotion authority;
- machine-readable search-space, objective, trial, and decision records.

## Boundary evidence

Gate 6A performed no model fitting, optimization trial, uncertainty calibration, ensemble search, locked-test access, or confirmatory evaluation.

It makes no claim of:

- improved forecasting performance;
- optimization impact;
- production operation;
- savings;
- structural drift;
- causality.

## Governance conclusion

V1 remains unchanged. Gate 6A is closed as a design and governance gate.

The next permitted stage is **Gate 6B: advanced tabular candidates**. Gate 6B must operate only inside the Gate 6A contracts and cannot access or reuse the V1 locked test.
