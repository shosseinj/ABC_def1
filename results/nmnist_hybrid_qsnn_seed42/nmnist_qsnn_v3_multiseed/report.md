# N-MNIST QSNN-v3 Frozen Five-Seed Validation Campaign

Architecture and hyperparameters were frozen to the selected `wide4 + project_measure2` configuration. No architecture tuning was performed between seeds. The official test partition and attacks were not used.

| Seed | Validation Accuracy | Macro-F1 | Best Epoch | Runtime (s) |
|---:|---:|---:|---:|---:|
| 42 | 97.06% | 97.06% | 37 | 1200.00 |
| 123 | 97.28% | 97.28% | 24 | 938.80 |
| 777 | 97.52% | 97.52% | 40 | 1245.30 |
| 2026 | 97.44% | 97.44% | 38 | 1251.55 |
| 6543 | 97.06% | 97.06% | 40 | 1237.35 |

Validation accuracy mean +/- sample SD: 97.2720% +/- 0.2119%
Macro-F1 mean +/- sample SD: 97.2736% +/- 0.2119%

SNN reference validation accuracy mean +/- sample SD: 98.1320% +/- 0.2407%
SNN reference Macro-F1 validation aggregate: unavailable in the preserved SNN multi-seed artifacts. The existing SNN `macro_f1` aggregate is for the held-out test partition and is not compared here.

QSNN minus SNN validation accuracy: -0.8600 percentage points.
QSNN versus SNN Macro-F1: not computed because the available SNN Macro-F1 aggregate is test-partition-only.
