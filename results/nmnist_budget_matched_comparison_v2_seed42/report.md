# Corrected N-MNIST Budget-Matched Comparison v2 (Seed 42)

Status: **complete**. Seed-42 evidence only; this is not a multi-seed robustness claim.

## Design

- Original fixed-budget rows and full event arrays are reused from the hash-verified, fully audited v3 artifact.
- Access 21: TEMP 13 + 4x2 candidates; matched to PGD's 21 candidate evaluations.
- Access 41: TEMP 13 + 4x7 candidates; matched to PGD's 41 attack-internal forwards.
- PGD has 41 attack forwards, 20 backward evaluations, and 21 candidates. TEMP has Q forwards/candidates and zero backwards.
- No backward-pass conversion factor or backward equivalence is asserted.
- Verification uses two additional forwards per new record and is excluded from attack timing/accounting.

## Wall-Clock Calibration

- QSNN: matched; Q=4000, median TEMP/PGD ratio=0.978.
- SNN: matched; Q=1600, median TEMP/PGD ratio=1.083.

## Paired Outcomes

| Condition | Model | epsilon | PGD ASR | TEMP ASR | Delta | PGD only | TEMP only | Both succeed | Neither succeeds | Exact p | Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_v3_fixed_budget | SNN | 0.05 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| original_v3_fixed_budget | SNN | 0.10 | 0.000 | 0.010 | +0.010 | 0 | 1 | 0 | 99 | 1 | [+0.000, +0.030] |
| original_v3_fixed_budget | QSNN | 0.05 | 0.030 | 0.040 | +0.010 | 0 | 1 | 3 | 96 | 1 | [+0.000, +0.030] |
| original_v3_fixed_budget | QSNN | 0.10 | 0.030 | 0.050 | +0.020 | 0 | 2 | 3 | 95 | 0.5 | [+0.000, +0.050] |
| access_candidates_21 | SNN | 0.05 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| access_candidates_21 | SNN | 0.10 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| access_candidates_21 | QSNN | 0.05 | 0.030 | 0.040 | +0.010 | 0 | 1 | 3 | 96 | 1 | [+0.000, +0.030] |
| access_candidates_21 | QSNN | 0.10 | 0.030 | 0.030 | +0.000 | 0 | 0 | 3 | 97 | 1 | [+0.000, +0.000] |
| access_forwards_41 | SNN | 0.05 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| access_forwards_41 | SNN | 0.10 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| access_forwards_41 | QSNN | 0.05 | 0.030 | 0.040 | +0.010 | 0 | 1 | 3 | 96 | 1 | [+0.000, +0.030] |
| access_forwards_41 | QSNN | 0.10 | 0.030 | 0.030 | +0.000 | 0 | 0 | 3 | 97 | 1 | [+0.000, +0.000] |
| wallclock_snn | SNN | 0.05 | 0.000 | 0.000 | +0.000 | 0 | 0 | 0 | 100 | 1 | [+0.000, +0.000] |
| wallclock_snn | SNN | 0.10 | 0.000 | 0.010 | +0.010 | 0 | 1 | 0 | 99 | 1 | [+0.000, +0.030] |
| wallclock_qsnn | QSNN | 0.05 | 0.030 | 0.040 | +0.010 | 0 | 1 | 3 | 96 | 1 | [+0.000, +0.030] |
| wallclock_qsnn | QSNN | 0.10 | 0.030 | 0.050 | +0.020 | 0 | 2 | 3 | 95 | 0.5 | [+0.000, +0.050] |

Complete sample IDs, runtime means/medians, distortions, and audit fields are in the CSV/JSON outputs.
The original, access-21, access-41, and model-specific wall-clock conditions are reported separately.
