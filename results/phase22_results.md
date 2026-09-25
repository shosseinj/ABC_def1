# Phase 22 — Model-seed sensitivity

Compared seeds 42, 777, and 2026 on the same fixed 90/30/30 development split (split seed 271), using existing Phase-21 validation observations only. The frozen Classical PGD attack was evaluated at epsilon/T = 0.02 and 0.10; no test features were accessed.

Failure rates are denominator-aware: attack failures divided by clean-correct validation samples within each split/seed/epsilon cell. Clean-incorrect samples are excluded, not relabeled as robust.

## epsilon/T = 0.02
- seed 42: mean failure rate 0.037; mean clean-correct denominator 27.0/30; split rates 0.037
- seed 777: mean failure rate 0.034; mean clean-correct denominator 29.0/30; split rates 0.034
- seed 2026: mean failure rate 0.000; mean clean-correct denominator 20.0/30; split rates 0.000

## epsilon/T = 0.1
- seed 42: mean failure rate 0.407; mean clean-correct denominator 27.0/30; split rates 0.407
- seed 777: mean failure rate 0.345; mean clean-correct denominator 29.0/30; split rates 0.345
- seed 2026: mean failure rate 0.000; mean clean-correct denominator 20.0/30; split rates 0.000

## Compact comparison

| Seed | epsilon 2% failure | epsilon 10% failure | most sensitive TTFS coordinate |
|---:|---:|---:|---:|
| 42 | 0.037 | 0.407 | TTFS 2 |
| 777 | 0.034 | 0.345 | TTFS 2 |
| 2026 | 0.000 | 0.000 | TTFS 2 |

## Answers
1. On the common split, the same samples are not attacked successfully across seeds; overlap is partial and seed-dependent.
2. This fixed-split sample is too small for a class-wide concentration claim; labels and outcomes are in the CSV.
3. The most sensitive coordinate is computed from the largest absolute TTFS update per sample and summarized above.
4. On this split, the dominant coordinate is consistent across seeds: TTFS 2.
5. The CSV permits direct clean-margin comparison for failed versus robust clean-correct samples; no causal claim is made here.
6. Quantum-feature and logit gradient norms were not recorded in Phase 21, so PGD's mechanism cannot be resolved from this artifact. Movement fields are not gradient norms.
7. Simplest supported explanation: seed-specific decision boundaries and margins make the same timing perturbation cross different samples at different seeds.

## Classification
**B. DECISION_BOUNDARY_SEED_DEPENDENCE**

This is a limited descriptive classification for one fixed development split, not a robustness claim. No architecture, TTFS, circuit, training, attack, or test-set conclusions were changed.
