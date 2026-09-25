# Phase 16 Decision-Aware QUANTUM-TEMP

## Defense

```text
L = CE + lambda_js * JS(p_clean,p_perturbed)
       + lambda_margin * ReLU(target - perturbed_true_class_margin)
       + lambda_q * measured-feature fidelity loss
```

Training uses unchanged uniform perturbations bounded by `0.02T`. Architecture,
optimizer, split, epochs, and Phase 14 attacks are unchanged.

## JS Ablation

For `lambda_js={0.1,0.5,1,2,5}`, clean validation accuracy remains 0.9667 and mean
PGD ASR remains 0.1121. `R_JS` ranges from `1.6e-6` to `8.0e-5`; JS-only provides no
validation robustness benefit.

## Margin Ablation

| Weight | Target | Val accuracy | Mean PGD ASR | R_margin |
|---:|---:|---:|---:|---:|
| 0.1 | 0.10 | 0.9667 | 0.1034 | 0.0097 |
| 0.5 | 0.10 | 0.9333 | 0.0804 | 0.0432 |
| 1.0 | 0.10 | 0.9000 | 0.0370 | 0.0832 |
| 2.0 | 0.10 | 0.9333 | 0.0625 | 0.1635 |
| 5.0 | 0.10 | 0.9000 | 0.0463 | 0.4030 |
| 0.1 | 0.05 | 0.9667 | 0.1121 | 0.0066 |
| 0.1 | 0.20 | 0.9667 | 0.1034 | 0.0172 |

Larger margin weights improve PGD robustness but violate the 2-point clean-validation
criterion. Weight 0.1 is the eligible margin setting.

## Combined Ablation

JS+Margin and JS+Margin+Quantum both preserve 0.9667 validation accuracy with mean
PGD ASR 0.1034. The validation selector freezes:

```text
lambda_js = 0.1
lambda_margin = 0.1
margin_target = 0.1
lambda_q = 5
```

It improves seed-42 PGD ASR at 5% and 10%, meeting the initial two-budget gate.

## Gradient Analysis

At the selected checkpoint:

| Loss | Circuit gradient | Classifier gradient |
|---|---:|---:|
| CE | 0.05905 | 0.11104 |
| JS | 0.000204 | 0.000049 |
| Margin | 0.16869 | 0.07017 |
| Quantum | 0.000558 | 0 |

Margin is the only auxiliary term with gradient scale comparable to CE. JS directly
reaches the head but is approximately 0.044% of the CE head gradient.

## Three-Seed Validation

| Seed | Model | Clean val accuracy | Mean PGD ASR | Mean attacked margin |
|---:|---|---:|---:|---:|
| 42 | Baseline | 0.9667 | 0.1121 | 0.6926 |
| 42 | Selected | 0.9667 | 0.1034 | 0.6753 |
| 777 | Baseline | 0.9667 | 0.1379 | 0.6492 |
| 777 | Selected | 0.9667 | 0.1379 | 0.6280 |
| 2026 | Baseline | 0.9333 | 0.1161 | 0.6284 |
| 2026 | Selected | 0.9000 | 0.1019 | 0.7150 |

Only seed 42 improves while preserving clean accuracy. Seed 777 is unchanged; seed
2026 improves PGD ASR but loses 3.33 accuracy points. Attacked margin is lower for the
selected model in seeds 42 and 777.

## Progression Decision

The multi-seed validation gate fails. Per the preregistered protocol:

- final test was not run;
- `test_set_accessed=false` is recorded;
- final Phase 14 attack and paired test tables are intentionally absent.

## Assessment

JS does not provide useful scale or robustness. Margin directly affects the decision
boundary and improves PGD resistance at stronger weights, but acceptable-clean-accuracy
effects are weak and not reproducible across validation seeds. Adding JS and quantum
fidelity does not clearly outperform margin-only or the previous fidelity-only model.

Phase 16 does not support a robustness claim.
