# 1. Interpreter

Used `C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`. The requested
path without the separator before `.venv` does not exist. No environment was changed.

# 2. Files Added

Added the Phase 17.1 runner, analysis helpers, focused tests, selected-candidate config,
seven required CSV/JSON artifacts, gate artifact, and this report.

# 3. Files Modified

Updated training to expose a test-free validation loader and a testable adversarial-loss
function. Updated `README.md`, `README_FA.md`, phase status, manifest, and phase runner.
The QSNN architecture, training PGD generator, and Phase 14 attacks are unchanged.

# 4. Regression Status

Phase 17.1 gate: 6 passed. Full regression suite: 59 passed in 6.15 seconds. The Phase 14
source hash and the validation-only/no-final-test invariants pass.

# 5. Frozen PGD Configuration

Every run uses one clean-start sign-PGD step, `epsilon_train=0.02T`, `alpha=0.02T`, and
no random start. Margin remains `lambda_margin=0.5`, target 0.1. JS and quantum fidelity
are disabled.

# 6. Lambda-Adv Ablation

| lambda_adv | Clean | Macro F1 | C0 | C1 | C2 | PGD 1 | PGD 2 | PGD 5 | PGD 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | .9667 | .9666 | 1.0 | .9 | 1.0 | .0690 | .0690 | .1034 | .1724 |
| 0.25 | .9667 | .9666 | 1.0 | .9 | 1.0 | .0345 | .0690 | .0690 | .1724 |
| 0.50 | .9667 | .9666 | 1.0 | .9 | 1.0 | 0 | 0 | .1034 | .2069 |
| 0.75 | .9667 | .9666 | 1.0 | .9 | 1.0 | 0 | 0 | .1034 | .2069 |
| 1.00 | .9667 | .9666 | 1.0 | .9 | 1.0 | 0 | 0 | .1034 | .2069 |

# 7. Loss-Contribution Analysis

`R_adv` rises approximately linearly: 0.102, 0.254, 0.509, 0.764, and 1.018.
`R_margin` remains 0.047-0.049. Total auxiliary ratios are 0.149, 0.302, 0.557,
0.812, and 1.067. Lower adversarial weight does not dominate clean CE, but values below
0.5 lose small-epsilon stability.

# 8. Class-2 Clean Stability

All seed-42 candidates retain class-2 clean accuracy 1.0. In multi-seed replication,
0.5 and 0.75 both obtain class-2 accuracies 1.0, 0.9, and 0.8 for seeds 42, 777, and
2026. This is the same one-sample class-2 degradation pattern as Phase 17.

# 9. Class-2 Attack Robustness

At lambda 0.5, class-2 mean clean/attacked margins at 1/2/5/10% are:

| Seed | Clean | 1% | 2% | 5% | 10% |
|---:|---:|---:|---:|---:|---:|
| 42 | .1807 | .1676 | .1543 | .1140 | .0451 |
| 777 | .6161 | .5756 | .5345 | .4083 | .1781 |
| 2026 | .1098 | .1030 | .0959 | .0743 | .0416 |

Positive attacked-margin fractions at 10% are 0.6, 0.7, and 0.7 respectively.

# 10. Clean-Degradation Sample Analysis

Sample ID 119 is the only class-2 baseline-correct/candidate-wrong sample and repeats for
both candidates in seeds 777 and 2026. Baseline margins are only 0.00496 and 0.00103;
candidate margins range from -0.0973 to -0.0320. Baseline true-class confidence is
0.4721/0.4631 and candidate confidence is 0.4552-0.3840. The repeated sample is clearly
near the baseline decision boundary; measured features are saved in the class-2 artifact.

# 11. Candidate Selection

Seed-42 eligibility rejects 0.10 and 0.25 because both worsen 1% and/or 2% paired PGD.
Weights 0.50, 0.75, and 1.00 tie on predictions; the two smaller candidates, 0.50 and
0.75, are replicated. Smaller-weight tie preference excludes 1.00.

# 12. Multi-Seed Validation

Both candidates produce identical paired counts. Clean deltas for seeds 42/777/2026 are
0, -0.0333, and -0.0333. Net gains at 1/2/5/10% are `0/1/1/2`, `0/0/1/0`, and
`-2/-2/1/3`. Common-clean counts are 29, 28, and 27.

# 13. Small-Epsilon Robustness

Weights 0.10 and 0.25 already degrade small-epsilon behavior on seed 42. Weights 0.50
and 0.75 avoid this at seeds 42 and 777 but reproduce two broken samples at both 1%
and 2% for seed 2026. Small-epsilon failures are not eliminated.

# 14. Class-Wise Margin Analysis

All class-wise clean and attacked margins, margin drops, and positive-margin fractions
are stored in `iris_phase171_class2_analysis.json`. Class 0 remains strongly separated.
Classes 1 and 2 carry the attack failures. Class-2 separation is weakest at seed 2026,
with only 0.8 positive clean-margin fraction for both candidates.

# 15. Class-Conditional Gradient Analysis

For lambda 0.5, adversarial-CE circuit/head norms are 0.821/0.364 for class 0,
0.474/0.774 for class 1, and 0.532/0.817 for class 2. Class 2 has the largest head norm,
but only slightly above class 1; it is not uniquely extreme. Margin gradients are zero
for class 0 and substantial for classes 1 and 2. Lambda 0.75 gives the same qualitative
pattern.

# 16. Class-Balance Diagnostic

Training counts are exactly `[30,30,30]`. Inverse-frequency weights would all equal one,
so class imbalance is not responsible and no class-balanced diagnostic run is justified.

# 17. Multi-Seed Gate

**FAILED.** Both candidates violate the 2-point clean criterion in seeds 777 and 2026,
mean clean delta is -2.22 points, and seed 2026 has negative net gain at both small
epsilons. No carry-forward candidate is frozen.

# 18. Test-Access Status

`test_set_accessed=false`. No final-test artifact was created. Phase 17.1 uses a loader
that does not transform or return held-out features.

# 19. Scientific Assessment

1. Lambda 1 is not uniquely too aggressive; 0.5 and 0.75 reproduce its predictions.
2. Every tested weight preserves seed-42 clean accuracy; no weight preserves it across all seeds.
3. Reduced weights retain some 5/10% gains, but values below 0.5 damage small-epsilon robustness.
4. Class 2 recovers on seed 42 but not on seeds 777/2026.
5. Sample 119 is repeatedly degraded and is extremely close to the baseline boundary.
6. The clean/robustness trade-off does not improve across all seeds.
7. No candidate satisfies the multi-seed gate.
8. Small-epsilon failures persist.
9. Class-2 head gradients are large but comparable to class 1, not uniquely pathological.
10. Class imbalance is not responsible because the training split is exactly balanced.
11. There is no validated candidate worth carrying forward.

Phase 17.1 is negative. The evidence does not support a robustness claim.

# 20. Recommended Next Phase

Do not continue scalar `lambda_adv` tuning. Diagnose sample 119 and the class-1/class-2
boundary across seeds, preferably through checkpoint trajectories or validation-only
boundary calibration, while retaining the no-test rule.
