# Final Frozen-Checkpoint Attack Comparison

Held-out evaluation using only the five SHA-256-verified checkpoints in `results/final_clean_protocol.json`.

## Per-Seed Results

| Seed | epsilon | Attack | Clean-correct | Successes | ASR | 1-Fidelity | Trace Distance | Exact feasibility |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | 2% | classical_timing | 29 | 3 | 0.1034 | 0.000987 | 0.031409 | 1.0000 |
| 42 | 2% | temp_drift_adaptive | 29 | 3 | 0.1034 | 0.000986 | 0.031395 | 1.0000 |
| 42 | 5% | classical_timing | 29 | 8 | 0.2759 | 0.006151 | 0.078429 | 1.0000 |
| 42 | 5% | temp_drift_adaptive | 29 | 8 | 0.2759 | 0.005855 | 0.076436 | 1.0000 |
| 42 | 10% | classical_timing | 29 | 11 | 0.3793 | 0.024397 | 0.156195 | 1.0000 |
| 42 | 10% | temp_drift_adaptive | 29 | 12 | 0.4138 | 0.021882 | 0.147617 | 1.0000 |
| 123 | 2% | classical_timing | 29 | 2 | 0.0690 | 0.000987 | 0.031409 | 1.0000 |
| 123 | 2% | temp_drift_adaptive | 29 | 2 | 0.0690 | 0.000972 | 0.031166 | 1.0000 |
| 123 | 5% | classical_timing | 29 | 5 | 0.1724 | 0.006151 | 0.078429 | 1.0000 |
| 123 | 5% | temp_drift_adaptive | 29 | 5 | 0.1724 | 0.005683 | 0.075188 | 1.0000 |
| 123 | 10% | classical_timing | 29 | 12 | 0.4138 | 0.024324 | 0.155958 | 1.0000 |
| 123 | 10% | temp_drift_adaptive | 29 | 12 | 0.4138 | 0.021423 | 0.145864 | 1.0000 |
| 777 | 2% | classical_timing | 29 | 5 | 0.1724 | 0.000987 | 0.031409 | 1.0000 |
| 777 | 2% | temp_drift_adaptive | 29 | 5 | 0.1724 | 0.000984 | 0.031363 | 1.0000 |
| 777 | 5% | classical_timing | 29 | 8 | 0.2759 | 0.006080 | 0.077957 | 1.0000 |
| 777 | 5% | temp_drift_adaptive | 29 | 8 | 0.2759 | 0.005799 | 0.076050 | 1.0000 |
| 777 | 10% | classical_timing | 29 | 11 | 0.3793 | 0.023891 | 0.154490 | 1.0000 |
| 777 | 10% | temp_drift_adaptive | 29 | 11 | 0.3793 | 0.022352 | 0.149145 | 1.0000 |
| 2026 | 2% | classical_timing | 23 | 2 | 0.0870 | 0.000987 | 0.031409 | 1.0000 |
| 2026 | 2% | temp_drift_adaptive | 23 | 2 | 0.0870 | 0.000978 | 0.031262 | 1.0000 |
| 2026 | 5% | classical_timing | 23 | 2 | 0.0870 | 0.006080 | 0.077957 | 1.0000 |
| 2026 | 5% | temp_drift_adaptive | 23 | 2 | 0.0870 | 0.005633 | 0.074906 | 1.0000 |
| 2026 | 10% | classical_timing | 23 | 4 | 0.1739 | 0.023964 | 0.154727 | 1.0000 |
| 2026 | 10% | temp_drift_adaptive | 23 | 4 | 0.1739 | 0.021592 | 0.146424 | 1.0000 |
| 6543 | 2% | classical_timing | 27 | 0 | 0.0000 | 0.000987 | 0.031409 | 1.0000 |
| 6543 | 2% | temp_drift_adaptive | 27 | 0 | 0.0000 | 0.000983 | 0.031359 | 1.0000 |
| 6543 | 5% | classical_timing | 27 | 4 | 0.1481 | 0.006080 | 0.077957 | 1.0000 |
| 6543 | 5% | temp_drift_adaptive | 27 | 4 | 0.1481 | 0.005862 | 0.076488 | 1.0000 |
| 6543 | 10% | classical_timing | 27 | 8 | 0.2963 | 0.023770 | 0.154067 | 1.0000 |
| 6543 | 10% | temp_drift_adaptive | 27 | 8 | 0.2963 | 0.022517 | 0.149777 | 1.0000 |

## Paired Common-Clean-Correct Outcomes

| epsilon | Common clean-correct | PGD only | TEMP-DRIFT only | Both successful | Both robust |
|---:|---:|---:|---:|---:|---:|
| 2% | 137 | 0 | 0 | 12 | 125 |
| 5% | 137 | 0 | 0 | 27 | 110 |
| 10% | 137 | 0 | 1 | 46 | 90 |

## Compact Summary

| epsilon | PGD ASR mean±SD | TEMP-DRIFT ASR mean±SD | PGD 1-F mean±SD | TEMP-DRIFT 1-F mean±SD | ASR Winner | Drift Winner |
|---:|---:|---:|---:|---:|---|---|
| 2% | 0.0864±0.0622 | 0.0864±0.0622 | 0.000987±0.000000 | 0.000980±0.000006 | TIE | PGD |
| 5% | 0.1918±0.0828 | 0.1918±0.0828 | 0.006109±0.000039 | 0.005766±0.000103 | TIE | PGD |
| 10% | 0.3285±0.0967 | 0.3354±0.1023 | 0.024069±0.000276 | 0.021953±0.000473 | TEMP-DRIFT | PGD |

**TEMP-DRIFT COMPETITIVE**
