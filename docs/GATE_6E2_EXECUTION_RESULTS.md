# Gate 6E2 governed uncertainty execution results

## Status

Gate 6E2 is complete as validation-only uncertainty evidence. Gate 6E3 human authority remains required.

## Configuration evidence

| Configuration | Method | WIS | Peak WIS | 90% coverage | 90% width | Hard constraints | Human-selection eligible |
|---|---|---:|---:|---:|---:|---|---|
| `aci_672_0p02` | `adaptive_absolute_conformal` | 2.283485 | 9.550716 | 0.9003 | 20.572056 | False | False |
| `aci_2688_0p02` | `adaptive_absolute_conformal` | 2.292192 | 9.782142 | 0.9009 | 20.312280 | False | False |
| `aci_672_0p01` | `adaptive_absolute_conformal` | 2.325175 | 9.692876 | 0.9008 | 20.791914 | False | False |
| `aci_2688_0p01` | `adaptive_absolute_conformal` | 2.328511 | 9.898191 | 0.9012 | 20.289719 | False | False |
| `aci_672_0p005` | `adaptive_absolute_conformal` | 2.346516 | 9.679160 | 0.9011 | 20.819507 | False | False |
| `aci_2688_0p005` | `adaptive_absolute_conformal` | 2.357254 | 9.957693 | 0.9021 | 20.416157 | False | False |
| `rolling_672` | `rolling_absolute_conformal` | 2.424463 | 10.216608 | 0.8939 | 19.881408 | False | False |
| `expanding_all` | `expanding_absolute_conformal` | 2.460686 | 10.842203 | 0.8978 | 19.452028 | False | False |
| `rolling_2688` | `rolling_absolute_conformal` | 2.481674 | 11.033722 | 0.8942 | 19.053885 | False | False |

## Recommendation boundary

- Outcome: `no_action`.
- Recommended configuration: `None`.
- Eligible configurations: `[]`.
- Pareto-eligible configurations: `[]`.
- Automatic promotion: `false`.
- Human decision required: `true`.

## Evidence boundary

- Validation origins per configuration: `7004`.
- Total interval-prediction rows: `63036`.
- Maximum prediction origin: `28027`.
- Maximum target dependency: `28028`.
- Locked-test access: `false`.
- Locked-prediction parsing: `false`.
- Confirmatory evaluation: `false`.
- V1 mutation: `false`.

Gate 6E2 does not establish production, savings, causal, drift, optimization-impact, or confirmatory evidence.
