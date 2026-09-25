# 1. Interpreter

`C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`

# 2. Files Added

The required Phase 17.3 CSV, JSON, configuration, gate, report, and selected-checkpoint artifacts were generated.

# 3. Files Modified

`experiments/iris/phase173.py`, `scripts/run_phase173_checkpoint_selection.py`, `tests/test_phase173_checkpoint_selection.py`, and `phase_runner.py`.

# 4. Regression Status

Phase 17.3 gate: 9 passed. Full pytest suite: 74 passed. No failures or warnings were reported.

# 5. Current Checkpoint Rule

Primary metric: highest validation accuracy. Equal accuracy is broken by lower validation cross-entropy loss; earliest epoch is the deterministic final tie-break. Thus lower validation loss overrides equal accuracy.

Selected epochs: `[80, 80, 39]` for seeds `[42, 777, 2026]`.

# 6. Epoch-Level Validation Dynamics

All 240 epoch rows use validation data only. Low margin means true-class margin `< 0.10`; negative margins are included and also reported separately.

# 7. RULE_A Current Results

| rule | seed | selected_epoch | clean_accuracy | macro_F1 | min_class_acc | class1_acc | class2_acc | mean_margin | PGD2_ASR | sample119_pred | sample119_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_A_CURRENT | 42 | 80 | 0.966667 | 0.966583 | 0.900000 | 0.900000 | 1.000000 | 0.735401 | 0.000000 | 2 | 0.020163 |
| RULE_A_CURRENT | 777 | 80 | 0.933333 | 0.933333 | 0.900000 | 0.900000 | 0.900000 | 0.926538 | 0.035714 | 1 | -0.095990 |
| RULE_A_CURRENT | 2026 | 39 | 0.900000 | 0.899749 | 0.800000 | 0.900000 | 0.800000 | 0.202405 | 0.074074 | 1 | -0.040910 |

# 8. RULE_B Accuracy-Stable Results

| rule | seed | selected_epoch | clean_accuracy | macro_F1 | min_class_acc | class1_acc | class2_acc | mean_margin | PGD2_ASR | sample119_pred | sample119_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_B_ACCURACY_STABLE | 42 | 80 | 0.966667 | 0.966583 | 0.900000 | 0.900000 | 1.000000 | 0.735401 | 0.000000 | 2 | 0.020163 |
| RULE_B_ACCURACY_STABLE | 777 | 80 | 0.933333 | 0.933333 | 0.900000 | 0.900000 | 0.900000 | 0.926538 | 0.035714 | 1 | -0.095990 |
| RULE_B_ACCURACY_STABLE | 2026 | 39 | 0.900000 | 0.899749 | 0.800000 | 0.900000 | 0.800000 | 0.202405 | 0.074074 | 1 | -0.040910 |

# 9. RULE_C Class-Stability Results

| rule | seed | selected_epoch | clean_accuracy | macro_F1 | min_class_acc | class1_acc | class2_acc | mean_margin | PGD2_ASR | sample119_pred | sample119_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_C_CLASS_STABILITY | 42 | 78 | 0.966667 | 0.966583 | 0.900000 | 0.900000 | 1.000000 | 0.718380 | 0.068966 | 2 | 0.016179 |
| RULE_C_CLASS_STABILITY | 777 | 80 | 0.933333 | 0.933333 | 0.900000 | 0.900000 | 0.900000 | 0.926538 | 0.035714 | 1 | -0.095990 |
| RULE_C_CLASS_STABILITY | 2026 | 35 | 0.900000 | 0.899749 | 0.800000 | 0.900000 | 0.800000 | 0.157623 | 0.074074 | 1 | -0.054760 |

# 10. RULE_D Robust-Validation Results

| rule | seed | selected_epoch | clean_accuracy | macro_F1 | min_class_acc | class1_acc | class2_acc | mean_margin | PGD2_ASR | sample119_pred | sample119_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_D_ROBUST_VALIDATION | 42 | 80 | 0.966667 | 0.966583 | 0.900000 | 0.900000 | 1.000000 | 0.735401 | 0.000000 | 2 | 0.020163 |
| RULE_D_ROBUST_VALIDATION | 777 | 58 | 0.933333 | 0.932660 | 0.800000 | 0.800000 | 1.000000 | 0.690236 | 0.000000 | 2 | 0.057795 |
| RULE_D_ROBUST_VALIDATION | 2026 | 29 | 0.900000 | 0.899749 | 0.800000 | 0.900000 | 0.800000 | 0.104715 | 0.000000 | 1 | -0.065863 |

# 11. Sample 119 Diagnostic

| rule | seed | selected_epoch | sample119_pred | sample119_margin | sample119_confidence |
| --- | --- | --- | --- | --- | --- |
| RULE_A_CURRENT | 42 | 80 | 2 | 0.020163 | 0.471245 |
| RULE_B_ACCURACY_STABLE | 42 | 80 | 2 | 0.020163 | 0.471245 |
| RULE_C_CLASS_STABILITY | 42 | 78 | 2 | 0.016179 | 0.467374 |
| RULE_D_ROBUST_VALIDATION | 42 | 80 | 2 | 0.020163 | 0.471245 |
| RULE_A_CURRENT | 777 | 80 | 1 | -0.095990 | 0.455220 |
| RULE_B_ACCURACY_STABLE | 777 | 80 | 1 | -0.095990 | 0.455220 |
| RULE_C_CLASS_STABILITY | 777 | 80 | 1 | -0.095990 | 0.455220 |
| RULE_D_ROBUST_VALIDATION | 777 | 58 | 2 | 0.057795 | 0.460353 |
| RULE_A_CURRENT | 2026 | 39 | 1 | -0.040910 | 0.380721 |
| RULE_B_ACCURACY_STABLE | 2026 | 39 | 1 | -0.040910 | 0.380721 |
| RULE_C_CLASS_STABILITY | 2026 | 35 | 1 | -0.054760 | 0.370450 |
| RULE_D_ROBUST_VALIDATION | 2026 | 29 | 1 | -0.065863 | 0.357989 |

Sample 119 was diagnostic only and is absent from every selection rule.

# 12. Class 1/2 Stability

| rule | seed | class1_acc | class2_acc | class1_mean_margin | class2_mean_margin | class1_low_margin_count | class2_low_margin_count | class1_misclassified | class2_misclassified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_A_CURRENT | 42 | 0.900000 | 1.000000 | 0.071763 | 0.180709 | 5 | 3 | 1 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | 0.900000 | 1.000000 | 0.071763 | 0.180709 | 5 | 3 | 1 | 0 |
| RULE_C_CLASS_STABILITY | 42 | 0.900000 | 1.000000 | 0.073350 | 0.164772 | 5 | 3 | 1 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | 0.900000 | 1.000000 | 0.071763 | 0.180709 | 5 | 3 | 1 | 0 |
| RULE_A_CURRENT | 777 | 0.900000 | 0.900000 | 0.411412 | 0.616105 | 2 | 1 | 1 | 1 |
| RULE_B_ACCURACY_STABLE | 777 | 0.900000 | 0.900000 | 0.411412 | 0.616105 | 2 | 1 | 1 | 1 |
| RULE_C_CLASS_STABILITY | 777 | 0.900000 | 0.900000 | 0.411412 | 0.616105 | 2 | 1 | 1 | 1 |
| RULE_D_ROBUST_VALIDATION | 777 | 0.800000 | 1.000000 | 0.215033 | 0.485811 | 2 | 1 | 2 | 0 |
| RULE_A_CURRENT | 2026 | 0.900000 | 0.800000 | 0.056234 | 0.109837 | 5 | 3 | 1 | 2 |
| RULE_B_ACCURACY_STABLE | 2026 | 0.900000 | 0.800000 | 0.056234 | 0.109837 | 5 | 3 | 1 | 2 |
| RULE_C_CLASS_STABILITY | 2026 | 0.900000 | 0.800000 | 0.062647 | 0.083045 | 5 | 3 | 1 | 2 |
| RULE_D_ROBUST_VALIDATION | 2026 | 0.900000 | 0.800000 | 0.060392 | 0.050487 | 9 | 8 | 1 | 2 |

# 13. Validation Attack Comparison

| rule | seed | attack | epsilon | N_common | failures | ASR | rescued_vs_current | broken_vs_current | net_gain_vs_current |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_A_CURRENT | 42 | random_jitter | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | random_jitter | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | random_jitter | 0.050000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | random_jitter | 0.100000 | 29 | 2 | 0.068966 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | classical_timing | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | classical_timing | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | classical_timing | 0.050000 | 29 | 3 | 0.103448 | 0 | 0 | 0 |
| RULE_A_CURRENT | 42 | classical_timing | 0.100000 | 29 | 6 | 0.206897 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | random_jitter | 0.010000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | random_jitter | 0.020000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | random_jitter | 0.050000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | random_jitter | 0.100000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | classical_timing | 0.010000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | classical_timing | 0.020000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | classical_timing | 0.050000 | 28 | 3 | 0.107143 | 0 | 0 | 0 |
| RULE_A_CURRENT | 777 | classical_timing | 0.100000 | 28 | 6 | 0.214286 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | random_jitter | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | random_jitter | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | random_jitter | 0.050000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | random_jitter | 0.100000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | classical_timing | 0.010000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | classical_timing | 0.020000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | classical_timing | 0.050000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_A_CURRENT | 2026 | classical_timing | 0.100000 | 27 | 3 | 0.111111 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | random_jitter | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | random_jitter | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | random_jitter | 0.050000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | random_jitter | 0.100000 | 29 | 2 | 0.068966 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | classical_timing | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | classical_timing | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | classical_timing | 0.050000 | 29 | 3 | 0.103448 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 42 | classical_timing | 0.100000 | 29 | 6 | 0.206897 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | random_jitter | 0.010000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | random_jitter | 0.020000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | random_jitter | 0.050000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | random_jitter | 0.100000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | classical_timing | 0.010000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | classical_timing | 0.020000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | classical_timing | 0.050000 | 28 | 3 | 0.107143 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 777 | classical_timing | 0.100000 | 28 | 6 | 0.214286 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | random_jitter | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | random_jitter | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | random_jitter | 0.050000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | random_jitter | 0.100000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | classical_timing | 0.010000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | classical_timing | 0.020000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | classical_timing | 0.050000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_B_ACCURACY_STABLE | 2026 | classical_timing | 0.100000 | 27 | 3 | 0.111111 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | random_jitter | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | random_jitter | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | random_jitter | 0.050000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | random_jitter | 0.100000 | 29 | 2 | 0.068966 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | classical_timing | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 42 | classical_timing | 0.020000 | 29 | 2 | 0.068966 | 0 | 2 | -2 |
| RULE_C_CLASS_STABILITY | 42 | classical_timing | 0.050000 | 29 | 2 | 0.068966 | 1 | 0 | 1 |
| RULE_C_CLASS_STABILITY | 42 | classical_timing | 0.100000 | 29 | 5 | 0.172414 | 1 | 0 | 1 |
| RULE_C_CLASS_STABILITY | 777 | random_jitter | 0.010000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | random_jitter | 0.020000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | random_jitter | 0.050000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | random_jitter | 0.100000 | 28 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | classical_timing | 0.010000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | classical_timing | 0.020000 | 28 | 1 | 0.035714 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | classical_timing | 0.050000 | 28 | 3 | 0.107143 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 777 | classical_timing | 0.100000 | 28 | 6 | 0.214286 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | random_jitter | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | random_jitter | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | random_jitter | 0.050000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | random_jitter | 0.100000 | 27 | 1 | 0.037037 | 0 | 1 | -1 |
| RULE_C_CLASS_STABILITY | 2026 | classical_timing | 0.010000 | 27 | 0 | 0.000000 | 2 | 0 | 2 |
| RULE_C_CLASS_STABILITY | 2026 | classical_timing | 0.020000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | classical_timing | 0.050000 | 27 | 2 | 0.074074 | 0 | 0 | 0 |
| RULE_C_CLASS_STABILITY | 2026 | classical_timing | 0.100000 | 27 | 3 | 0.111111 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | random_jitter | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | random_jitter | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | random_jitter | 0.050000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | random_jitter | 0.100000 | 29 | 2 | 0.068966 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | classical_timing | 0.010000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | classical_timing | 0.020000 | 29 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | classical_timing | 0.050000 | 29 | 3 | 0.103448 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 42 | classical_timing | 0.100000 | 29 | 6 | 0.206897 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | random_jitter | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | random_jitter | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | random_jitter | 0.050000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | random_jitter | 0.100000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | classical_timing | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | classical_timing | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | classical_timing | 0.050000 | 27 | 2 | 0.107143 | 1 | 1 | 0 |
| RULE_D_ROBUST_VALIDATION | 777 | classical_timing | 0.100000 | 27 | 5 | 0.214286 | 1 | 1 | 0 |
| RULE_D_ROBUST_VALIDATION | 2026 | random_jitter | 0.010000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 2026 | random_jitter | 0.020000 | 27 | 0 | 0.000000 | 0 | 0 | 0 |
| RULE_D_ROBUST_VALIDATION | 2026 | random_jitter | 0.050000 | 27 | 1 | 0.037037 | 0 | 1 | -1 |
| RULE_D_ROBUST_VALIDATION | 2026 | random_jitter | 0.100000 | 27 | 1 | 0.037037 | 0 | 1 | -1 |
| RULE_D_ROBUST_VALIDATION | 2026 | classical_timing | 0.010000 | 27 | 0 | 0.000000 | 2 | 0 | 2 |
| RULE_D_ROBUST_VALIDATION | 2026 | classical_timing | 0.020000 | 27 | 0 | 0.000000 | 2 | 0 | 2 |
| RULE_D_ROBUST_VALIDATION | 2026 | classical_timing | 0.050000 | 27 | 3 | 0.111111 | 0 | 1 | -1 |
| RULE_D_ROBUST_VALIDATION | 2026 | classical_timing | 0.100000 | 27 | 7 | 0.259259 | 0 | 4 | -4 |

ASR uses each checkpoint's clean-correct denominator; rescued/broken/net counts use samples clean-correct under both the alternative and current checkpoints.

# 14. Pareto Analysis

| seed | epoch | val_accuracy | min_class_acc | PGD2_ASR | mean_margin |
| --- | --- | --- | --- | --- | --- |
| 42 | 80 | 0.966667 | 0.900000 | 0.000000 | 0.735401 |
| 777 | 58 | 0.933333 | 0.800000 | 0.000000 | 0.690236 |
| 777 | 65 | 0.900000 | 0.800000 | 0.000000 | 0.766726 |
| 777 | 80 | 0.933333 | 0.900000 | 0.035714 | 0.926538 |
| 2026 | 29 | 0.900000 | 0.800000 | 0.000000 | 0.104715 |
| 2026 | 39 | 0.900000 | 0.800000 | 0.074074 | 0.202405 |
| 2026 | 40 | 0.866667 | 0.800000 | 0.038462 | 0.215003 |
| 2026 | 79 | 0.833333 | 0.700000 | 0.000000 | 0.777966 |
| 2026 | 80 | 0.833333 | 0.700000 | 0.040000 | 0.788436 |

Pareto membership is diagnostic and did not define an additional rule.

# 15. Multi-Seed Rule Comparison

| rule | mean_accuracy | mean_macro_F1 | mean_min_class_acc | mean_PGD1_ASR | mean_PGD2_ASR | mean_PGD5_ASR | mean_PGD10_ASR | improved_seeds | unchanged_seeds | worsened_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RULE_A_CURRENT | 0.933333 | 0.933222 | 0.866667 | 0.036596 | 0.036596 | 0.094888 | 0.177431 | 0 | 3 | 0 |
| RULE_B_ACCURACY_STABLE | 0.933333 | 0.933222 | 0.866667 | 0.036596 | 0.036596 | 0.094888 | 0.177431 | 0 | 3 | 0 |
| RULE_C_CLASS_STABILITY | 0.933333 | 0.933222 | 0.866667 | 0.011905 | 0.059585 | 0.083394 | 0.165937 | 1 | 2 | 0 |
| RULE_D_ROBUST_VALIDATION | 0.933333 | 0.932997 | 0.833333 | 0.000000 | 0.000000 | 0.107234 | 0.226814 | 0 | 2 | 1 |

# 16. Checkpoint-Only Gate

Outcome: **NEGATIVE**. Gate status: `negative_checkpoint_gate_failed`. Passing generic rules: `[]`.

# 17. Test-Access Status

Held-out test set accessed: **No**. Test loader invoked: **No**. Final-test artifacts: **None**.

# 18. Root-Cause Update

The current loss tie-break and accuracy-stable rule both selected seed-777 epoch 80. Robust-validation selected epoch 58 and preserved sample 119, but reduced seed-777 minimum class accuracy. Checkpoint selection therefore exposes a seed-specific trade-off rather than a generic cause or solution.

# 19. Scientific Assessment

1. Current checkpoint selection contributes materially across seeds: **not established**.
2. An earlier tied seed-777 checkpoint preserves sample 119 without harming global metrics: **no**.
3. Accuracy-stable tie-breaking improves seed 777: **no**.
4. Class-stability helps seed 2026: **no**.
5. Robustness-aware selection improves PGD without clean degradation: **no**.
6. Best generic rule: **none passed the prespecified gate**.
7. Improvement is general rather than sample-specific: **not demonstrated**.
8. Checkpoint selection alone solves Phase 17/17.1 instability: **no**.
9. Class-1/class-2 compression remains: **yes**.
10. Generic rule worth freezing: **none**.

# 20. Recommended Next Phase

Treat Phase 17.3 as a negative checkpoint-only result. Preserve the frozen objective and investigate the remaining class-1/class-2 boundary mechanism in a separately prespecified validation-only phase.
