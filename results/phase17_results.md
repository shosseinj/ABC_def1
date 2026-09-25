# 1. Interpreter

The requested path without the separator before `.venv` does not exist. The verified
existing interpreter was used without environment changes:
`C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`.

# 2. Files Added

`attacks/training_timing.py`, `scripts/run_phase17_adversarial_training.py`,
`tests/test_phase17_adversarial_training.py`, the selected config, Phase 17 CSV/JSON
artifacts, and this report.

# 3. Files Modified

`experiments/iris/training.py`, `phase_runner.py`, both READMEs, the phase-status
document, and `MANIFEST.txt`. Phase 14 attack source was not modified.

# 4. Regression Status

Phase 17 gate: 9 passed. Full regression suite: 53 passed in 13.86 seconds. The Phase 14
attack source hash and the held-out-test access invariant are both verified by tests.

# 5. Phase 17 Defense Definition

Training PGD maximizes true-label CE subject to `||t_adv-t||_inf <= 0.02T` and
`t_adv in [0,T]`. It uses clean initialization, sign ascent, no random start, and
`alpha=epsilon/steps`. The selected outer loss is:

```text
CE(clean) + CE(adversarial) + 0.5 * margin_loss(adversarial, target=0.1)
```

# 6. Training-Time PGD Verification

Tests cover zero epsilon, budget, bounds, local CE ascent, unchanged weights/gradients
during generation, outer-step weight updates, no clean-input mutation, and unchanged
Phase 14 PGD. Parameter gradients are disabled and restored inside generation. Each
inner step and the final timing tensor are detached, preventing second-order graphs.

# 7. PGD-Step Ablation

| Steps | Clean accuracy | Mean evaluation-PGD ASR |
|---:|---:|---:|
| 1 | 0.9000 | 0.0278 |
| 3 | 0.9000 | 0.0370 |
| 5 | 0.9000 | 0.0370 |

Baseline is 0.9667 clean accuracy and 0.1121 mean PGD ASR. Every step setting violates
the 2-point clean-accuracy criterion.

# 8. Selected PGD Strength

PGD-1 is retained as a diagnostic choice because it is strongest, cheapest, and stable.
It is explicitly marked ineligible on clean accuracy when used with Adv-CE alone.

# 9. Objective Ablation

| Objective | Clean accuracy | Random ASR | Mean PGD ASR |
|---|---:|---:|---:|
| Adv-CE | 0.9000 | 0.0000 | 0.0278 |
| Adv-CE + JS | 0.9000 | 0.0000 | 0.0278 |
| Adv-CE + Margin | 0.9667 | 0.0086 | 0.0776 |
| Adv-CE + JS + Margin | 0.9667 | 0.0086 | 0.0776 |

# 10. Loss-Scale Analysis

Selected epoch means: clean CE 0.8817, adversarial CE 0.8957, raw margin 0.0955,
weighted margin 0.0478, total 1.8252. In the combined variant weighted JS is only
`4.5e-5` and changes no reported validation metric.

# 11. Gradient-Norm Analysis

At the selected checkpoint, circuit/head norms are 0.0618/0.1251 for clean CE,
0.0642/0.1297 for adversarial CE, 0.0199/0.0251 for margin, and
0.000281/0.000143 for JS. Adversarial CE is comparable to clean CE and substantially
stronger than previous fidelity regularization.

# 12. Selected Validation Objective

Adv-CE+Margin is preferred over the tied JS combination because it uses fewer terms.
For seed 42 it preserves clean accuracy and improves PGD ASR at 2%, 5%, and 10%.

# 13. Multi-Seed Validation

| Seed | Model | Clean acc. | Macro F1 | Random ASR | PGD ASR |
|---:|---|---:|---:|---:|---:|
| 42 | Baseline | 0.9667 | 0.9666 | 0.0000 | 0.1121 |
| 42 | Selected | 0.9667 | 0.9666 | 0.0086 | 0.0776 |
| 777 | Baseline | 0.9667 | 0.9666 | 0.0345 | 0.1379 |
| 777 | Selected | 0.9333 | 0.9333 | 0.0000 | 0.0982 |
| 2026 | Baseline | 0.9333 | 0.9333 | 0.0357 | 0.1161 |
| 2026 | Selected | 0.9000 | 0.8997 | 0.0185 | 0.0833 |

# 14. Validation Paired Robustness

Net gains at 1/2/5/10% are `0/1/1/2` for seed 42, `0/0/1/0` for seed 777, and
`-2/-2/1/3` for seed 2026. Seed 2026 improves large-epsilon attacks while introducing
two new failures at both small epsilons. No paired result is statistically compelling
on the 30-sample validation split.

# 15. Test-Access Gate

**NO-GO.** Seeds 777 and 2026 each lose 3.33 clean-accuracy points; seed 2026 also has
a clear small-epsilon paired collapse. `test_set_accessed=false`.

# 16. Frozen Final Configuration

PGD-1, `epsilon_train=0.02T`, `alpha=0.02T`, clean start, no random start,
`lambda_adv=1`, `lambda_js=0`, `lambda_margin=0.5`, margin target 0.1. This is frozen
for reporting but is not authorized for held-out testing.

# 17. Final Clean Test Performance

Not run because the validation gate failed.

# 18. Final Classical PGD Results

Not run because the validation gate failed.

# 19. Final TEMP-DRIFT Results

Not run because the validation gate failed.

# 20. Final Paired Analysis

Not run because the validation gate failed.

# 21. Class-Wise Robustness

Class 0 is fully robust in these validation comparisons. At seed 42, 5%/10% gains are
entirely class 2 (ASR 0.30 to 0.20 and 0.60 to 0.40). Seeds 777 and 2026 gain in classes
1 and 2, but selected class-2 clean accuracy drops by 0.10 in each. Gains are class-specific
and partly confounded by clean degradation.

# 22. Robustness/Clean Trade-Off

Direct PGD exposure sharply lowers aggregate PGD ASR. Adv-CE alone costs 6.67 clean
points; margin restores seed-42 clean accuracy but not reproducibly. Attacked margin is
lower in seeds 42 and 2026 despite lower ASR. Seed-42 random ASR rises from 0 to 0.0086,
and seed 2026 worsens at small PGD epsilon.

# 23. Scientific Assessment

1. PGD timing training improves mean PGD ASR more consistently than random-jitter training, but not clean-preserving paired robustness.
2. One step is sufficient; 3 and 5 do not improve the trade-off.
3. Adv-CE alone lowers PGD ASR but loses 6.67 clean points.
4. JS adds no measurable benefit.
5. Margin restores seed-42 clean accuracy but is seed-sensitive.
6. Adv-CE+Margin gives the best seed-42 trade-off.
7. Aggregate PGD means reproduce, but clean preservation and small-epsilon paired gains do not.
8. The gate does not allow held-out test access.
9. Final paired PGD robustness is unknown because test was withheld.
10. TEMP-DRIFT transfer is unknown because test was withheld.
11. Gains are concentrated in classes 1 and 2.
12. Evidence is not strong enough for a robustness claim.

The Phase 17 result is negative under the preregistered gate.

# 24. Recommended Next Phase

Diagnose class-2 clean degradation before test access. A narrow next experiment could
validation-select class-balanced adversarial CE or the clean/adversarial weight while
retaining PGD-1, the unchanged evaluation attacks, and the same gate.
