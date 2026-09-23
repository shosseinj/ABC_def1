# Current Research Status — TEMP-DRIFT Benchmark / Prior QSNN Work

> **Active autonomous workflow status (2026-09-22):** Phase 0, 0.5, and 0.75 independently `PASS`. Phase 1 uses indivisible amplitude-bearing cell-packets and its contract/regression suite passes 18/18. Seed-42 binary `B_inf=1` is complete (834/1000 attacks, ASR 83.4%, independent audit PASS). Fixed attack/audit batch 64 GPU vectorization passed 192/192 exact comparisons, independent audit, profiler evidence, and 19.81x–51.16x measured speedup. The resumable `B_inf=2` condition preserves its valid 175-sample scalar prefix and continues from sample 176 without rerunning completed work.

## Benchmark phase status (updated 2026-09-21)

The active roadmap is now the **SNN + TEMP-DRIFT reference-paper benchmark** over N-MNIST, DVS-Gesture, and CIFAR10-DVS. QSNN is outside this benchmark scope. No new experiment was run during this Phase 0 audit.

| Phase | Dataset / output | Status | Evidence and main gap |
|---|---|---|---|
| 0 | Framework preparation | **PASS** | Repository, checkpoints, saved results, attack code, and audit framework inspected; see `Reports/phase_0_report.md`. |
| 0.5 | Benchmark contract | **PASS** | `B∞/B1/B0` projectors, independent auditor, result schema, deterministic subsets, hashing, and atomic manifests implemented; corrected suite passes 17/17. |
| 0.75 | Specification freeze | **PASS** | Exactly seeds 42/123/777; 9/9 checkpoints validated, 13/13 tests passed, and the independent gate returned PASS. |
| 1 | N-MNIST | **RUNNING** | Obsolete blocker resolved; 1000/1000 frozen samples per seed are feasible. Seed-42 dry run passed 4/4 independent audits; real benchmark runs resumably in seed order. |
| 2 | DVS-Gesture | **Training remediation complete** | Five new hash-recorded checkpoints and official-test evaluations exist; attack benchmark awaits the specification gate. |
| 3 | CIFAR10-DVS | **Waiting for prior phases** | Required seeds 42, 123, and 777 are complete and validated. Seeds 2026/6543 are outside the benchmark. |
| 4 | Reference comparison | **Not started** | `benchmark_comparison.csv` and `benchmark_comparison.md` do not exist. Reference-paper values have not been extracted or verified. |

### Required benchmark budgets

- N-MNIST: `B∞={1,2,3}`, `B1={500,750,1000,1500}`, `B0={200,300,400,600}`.
- DVS-Gesture and CIFAR10-DVS: `B∞={1,2,3}`, `B1={2000,4000,8000,16000}`, `B0={1000,2000,4000,8000}`.
- ASR must use clean-correct samples as its denominator, and every result must retain the realized `B∞`, `B1`, and `B0` values.

### Phase 0.75 remediation evidence

- **N-MNIST:** downloaded through Tonic 1.6.0 in the repository's native event format. Verification records 60,000 training and 10,000 test samples, sensor size 34×34×2, and fields `x,y,t,p`. Archive hashes are in `Reports/results/nmnist_dataset_verification.json`.
- **PIL-PGD:** a clean-room implementation derived from arXiv:2602.03284v1 equations 8–14, Algorithm 1, and Appendix D is in `attacks/pil_pgd.py`. It uses strict projected forward inputs, differentiable soft-retiming gradients, capacity and normalized budget penalties, and exact `B∞/B1/B0` projection. The upstream code commit used for cross-checking was `19f42e63d31cbdc4ef2d5ef7b6e40716f92531c3`; no unlicensed source was copied wholesale.
- **Contract tests:** 12/12 pass, including all three strict grid budgets; see `Reports/logs/phase_0_75_pil_contract_tests.log`.
- **DVS-Gesture:** `models/dvs_gesture_snn.py` and `scripts/train_dvs_gesture_snn_multiseed.py` produced five 242,315-parameter SNN checkpoints. Official-test accuracies were 84.09%, 78.03%, 87.12%, 73.86%, and 80.30% for seeds 42, 123, 777, 2026, and 6543. Per-seed hashes and cache/split provenance are in `Reports/results/dvs_gesture_training_summary.json`.
- **CIFAR10-DVS:** the valid seed-42 checkpoint remains unchanged at SHA-256 `7f0c6a4af012082a79c15296191520146ad612f4d60a9478f593a500d9a5d7d7`. Seeds 123 and 777 have hash-verified completion markers under `Reports/checkpoints/`. Seed 2026 has an unvalidated partial checkpoint and seed 6543 has not completed; neither is counted as complete.
- **Phase 0.75 gate:** `PASS`; see `Reports/logs/phase_0_75_gate_attempt_3.log`. Seeds 2026 and 6543 are not required by the frozen three-seed benchmark.

Phase 1 now uses the amended amplitude-bearing-cell contract. Integer amplitudes move whole, capacity-1 applies between cell-packets, collisions/merging are forbidden, and B∞/B1/B0 are unweighted packet-displacement measures. Prior epsilon/query-budget results are not relabeled as locked-protocol results.

### Full-test clean accuracy (2026-09-23)

> **Representation audit failure (2026-09-23):** The reported Binary-grid accuracies below are diagnostic OOD evaluations, not valid Binary-trained-model accuracies. All nine checkpoints were trained on count or per-sample-normalized-count inputs; no Binary-trained checkpoint exists, and the same checkpoint was reused after thresholding input to occupancy. Mark all Binary rows `REPRESENTATION_MISMATCH / NON_COMPARABLE` and remove them from paper benchmark tables. DVS-Gesture/CIFAR10-DVS “Integer” rows are actually fractional normalized-count rows and are `NON_COMPARABLE` to strict Integer-grid amplitudes; CIFAR10-DVS seed 42 additionally has a float32-training versus float16-cache evaluation mismatch. See `Reports/clean_accuracy_audit.md`.

> **Corrected Phase A active:** New N-MNIST strict Integer-only manifests are complete for seeds 42, 123, and 777. Each contains 1,000 samples selected deterministically only from that seed's Integer-clean-correct official-test predictions, with no Binary dependency. Corrected attacks write only under `Reports/results/nmnist_integer_corrected/`; the first condition (seed 42, $B_\infty=1$) independently passed at 84.70% ASR, and the remaining resumable conditions are running. Legacy intersection-based Integer rows are excluded from final tables.

> **True N-MNIST Binary clean training PASS (2026-09-23):** Independently initialized Binary-grid checkpoints were trained with occupancy inputs in both training and evaluation. Full official-test accuracies are 98.73% (seed 42), 98.29% (seed 123), and 98.53% (seed 777), giving **98.52 ± 0.22%** (sample SD; 95% t CI 97.97–99.06%). All three use all classes and pass the class-collapse check. These results supersede the invalid 85.48%, 95.65%, and 80.15% count-checkpoint-on-Binary evaluations. Evidence: `Reports/results/nmnist_binary_true/`.

> **True DVS-Gesture Binary clean training PASS (2026-09-23):** Representation-matched full official-test accuracies are 78.79% (seed 42), 80.68% (seed 123), and 84.47% (seed 777), giving **81.31 ± 2.89%**. All class-collapse checks pass. These results supersede the invalid 12.50%, 19.32%, and 9.47% normalized-count-checkpoint-on-Binary evaluations. Evidence: `Reports/results/dvs_gesture_binary_true/`.

Clean-only evaluation of all requested seeds and both frozen representations is complete. It used all 10,000 official N-MNIST test samples, all 264 official DVS-Gesture test samples, and all 1,000 samples in the frozen CIFAR10-DVS stratified test split (CIFAR10-DVS has no official train/test partition). No attack manifest, clean-correct filtering, balancing, or subsampling was used. All sample-count, checkpoint seed/hash, representation-difference, eval-mode, and repeated-prefix determinism checks passed. See `Reports/clean_accuracy_report.md`, `Reports/results/clean_accuracy_by_seed.csv`, and `Reports/results/clean_accuracy_summary.csv`.

| Dataset | Binary-grid mean ± SD | Integer-grid mean ± SD |
|---|---:|---:|
| N-MNIST | **98.5167% ± 0.2203% (true Binary-trained)** | 98.3867% ± 0.2146% |
| DVS-Gesture | **81.3131% ± 2.8930% (true Binary-trained)** | 83.0808% ± 4.6289% |
| CIFAR10-DVS | 16.0333% ± 3.3828% | 46.2000% ± 6.1733% |

### N-MNIST benchmark tables with seed-matched clean accuracy

While ASR is reported in seed-specific rows, each row uses the clean full-test accuracy from the same seed. This avoids mixing a three-seed mean accuracy with a single-seed ASR. The prior Binary ASRs were generated against count-trained checkpoints and are excluded; true Binary attack cells remain pending. After representation-matched attacks finish, the final paper table should replace the three seed-specific rows with one **SNN (Ours)** row containing the three-seed mean accuracy and three-seed aggregated ASR.

#### Table 1 — Binary-grid DVS, N-MNIST

| Dataset | Model | Acc. (%) | $B_\infty$ 1 | 2 | 3 | $B_1$ 500 | 750 | 1k | $B_0$ 200 | 300 | 400 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N-MNIST | ConvNet | 99.06 | 100 | 100 | 100 | 58.9 | 99.9 | 100 | 13.0 | 53.1 | 98.5 |
|  | ResNet18 | 99.62 | 100 | 100 | 100 | 69.2 | 97.4 | 100 | 78.9 | 100 | 100 |
|  | VGGSNN | 99.64 | 98.9 | 100 | 100 | 26.4 | 65.5 | 94.7 | 18.3 | 81.8 | 99.8 |
|  | **SNN (Ours), seed 42** | **98.73** | pending | pending | pending | pending | pending | pending | pending | pending | pending |
|  | **SNN (Ours), seed 123** | **98.29** | pending | pending | pending | pending | pending | pending | pending | pending | pending |
|  | **SNN (Ours), seed 777** | **98.53** | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Final representation-matched clean accuracy across seeds 42, 123, and 777: **98.52 ± 0.22%** (95% t CI 97.97–99.06%).

#### Table 2 — Integer-grid DVS, N-MNIST

| Dataset | Model | Acc. (%) | $B_\infty$ 1 | 2 | 3 | $B_1$ 500 | 750 | 1k | 1.5k | $B_0$ 200 | 300 | 400 | 600 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N-MNIST | ConvNet | 99.19 | 100 | 100 | 100 | 61.8 | 99.2 | 100 | 100 | 13.0 | 56.1 | 99.1 | 100 |
|  | ResNet18 | 99.62 | 100 | 100 | 100 | 53.9 | 93.6 | 99.8 | 100 | 86.1 | 99.8 | 100 | 100 |
|  | VGGSNN | 99.71 | 46.3 | 100 | 100 | 8.3 | 18.5 | 39.9 | 76.7 | 5.8 | 11.2 | 16.1 | 49.8 |
|  | **SNN (Ours), seed 42** | **98.49** | **83.3** | **100** | **100** | **100** | **100** | **100** | **100** | **99.8** | **100** | **100** | **100** |
|  | **SNN (Ours), seed 123** | **98.14** | **77.9** | **100** | **100** | **99.9** | pending | — | — | — | — | — | — |
|  | **SNN (Ours), seed 777** | **98.53** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Final clean accuracy across seeds 42, 123, and 777: **98.39 ± 0.21%**.

#### Final aggregated clean-accuracy values

| Dataset | Binary-grid Acc. | Integer-grid Acc. |
|---|---:|---:|
| N-MNIST | **98.52 ± 0.22%** | **98.39 ± 0.21%** |
| DVS-Gesture | **81.31 ± 2.89%** | **83.08 ± 4.63%** |
| CIFAR10-DVS | **16.03 ± 3.38%** | **46.20 ± 6.17%** |

---

## Preserved prior research record

**Repository state audited:** 2026-09-20. This README is an evidence-bounded status record based on the saved N-MNIST artifacts and completed experiments. The official N-MNIST test partition was not accessed for QSNN-v3 or the attack studies.

## 1. Project status

The N-MNIST QSNN-v3 validation campaign is complete. Its architecture and training protocol were frozen across five seeds: 42, 123, 777, 2026, and 6543.

| Model | Accuracy | Macro-F1 | Evaluation |
|---|---:|---:|---|
| QSNN-v3 | **97.272% ± 0.212%** | **97.274% ± 0.212%** | five-seed validation |
| SNN reference | **98.132% ± 0.241%** | — | five-seed validation reference |

The SNN reference Macro-F1 validation aggregate is not available in the preserved SNN artifacts. The QSNN values are validation results, not official-test results.

The canonical seed-42 PGD/TEMP-DRIFT-v2 protocol is independently audited with 2,400/2,400 passing records. The completed budget-matched extension contains 1,600/1,600 valid records across 100 frozen samples.

## 2. Final QSNN-v3 architecture

The frozen configuration is `wide4 + project_measure2`:

- `wide4` Conv/LIF frontend with 16 then 32 channels and a 4×4 spatial summary;
- 32-dimensional latent representation;
- learned **32→8 angle projection**;
- `project_measure2` quantum circuit;
- RY/RZ two-axis data encoding;
- two variational re-upload blocks;
- CNOT-ring entanglement;
- learned measurement basis and 256-state probability readout;
- linear classifier from the quantum probabilities;
- **no classical bypass** from the latent representation to logits.

The promoted end-to-end model has 40,762 parameters. The five-seed campaign was validation-only and used frozen architecture/configuration across seeds.

## 3. Attack protocol

The attack studies use frozen SNN and QSNN models and a common clean-correct manifest of **100 validation samples**, stratified at 10 samples per class. The attacks perturb event timestamps in timestamp space; event coordinates and polarity are preserved.

### Canonical representation and audit

- Event/frame indexing is `[time, polarity, y, x]`.
- Temporal bins use polarity-major channels on the 34×34 sensor grid.
- Canonical reconstruction uses 10 temporal bins and 2 polarity channels.
- Timestamp projection enforces monotonicity, the clean time window, and the per-event epsilon bound.
- Each serialized record is checked for event-count, coordinate, polarity, timestamp, canonical-frame, prediction, objective, success, distortion, and query-accounting invariants.
- Atomic NPZ writing, reload validation, record hashes, and independent audit status are retained in the result artifacts.

### PGD

- White-box timestamp PGD with 20 sign steps.
- 21 candidate evaluations.
- Accounting: 41 attack forwards and 20 backward evaluations.
- True-label cross-entropy objective.

### TEMP-DRIFT-v2

- Derivative-free timestamp search.
- Zero gradients and zero backward evaluations.
- Budget-scaled population/refinement search under the frozen access and wall-clock definitions.

## 4. Audit status

```text
1600/1600 records complete
100/100 samples complete
FULLY AUDITED: YES
All records passed independent audit: YES
Serialization reconstruction verified: YES
```

The original canonical v3 artifact also remains complete and independently audited: 2,400/2,400 records passed with zero failures. The budget-matched artifact reuses the hash-verified frozen v3 source records where specified and records all newly generated conditions separately.

## 5. Budget-matched results

All comparisons below use paired outcomes on the same 100 clean-correct samples. ASR is the fraction of samples for which the attack changes the predicted class.

### Access/query matched

TEMP access was matched to PGD using either 21 candidate evaluations or 41 attack-internal forwards.

| Model | Epsilon | PGD ASR | TEMP-DRIFT-v2 ASR |
|---|---:|---:|---:|
| SNN | 5% | 0% | 0% |
| SNN | 10% | 0% | 0% |
| QSNN | 5% | 3% | 4% |
| QSNN | 10% | 3% | 3% |

### Wall-clock matched

Frozen calibration selected TEMP Q=1600 for SNN and Q=4000 for QSNN. The median TEMP/PGD runtime ratios were 1.083 and 0.978, respectively.

| Model | Epsilon | PGD ASR | TEMP-DRIFT-v2 ASR |
|---|---:|---:|---:|
| SNN | 5% | 0% | 0% |
| SNN | 10% | 0% | 1% |
| QSNN | 5% | 3% | 4% |
| QSNN | 10% | 3% | 5% |

## 6. Distortion comparison

QSNN, epsilon=10%, wall-clock matched:

| Attack | Normalized timestamp shift | Events with bin changes | Frame L2 |
|---|---:|---:|---:|
| PGD | 0.0667 | 0.6463 | 105.65 |
| TEMP-DRIFT-v2 | 0.0925 | 0.8907 | 137.85 |

## 7. Interpretation and limitations

**Observed ASR differences were small and not statistically significant under paired tests. TEMP-DRIFT occasionally achieved slightly higher ASR, accompanied by larger timestamp/frame distortion.**

The attack evidence is single-seed evidence based on 100 paired samples. It does not support a claim that TEMP-DRIFT is superior, that QSNN is more robust, or that either method produces a statistically significant improvement. No multi-seed robustness claim is made.

The paired budget-matched analysis reports PGD-only, TEMP-only, both-success, and neither-success outcomes, together with exact paired tests and class-stratified bootstrap intervals. These results are descriptive and evidence-bounded by the frozen seed-42 attack protocol.

## 8. Key artifacts

### Clean QSNN-v3 validation

- `results/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/summary.json`
- `results/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/report.md`

### Canonical audited attack protocol

- `results/nmnist_attack_protocol_v3_auditable_seed42/STATUS.json`
- `results/nmnist_attack_protocol_v3_auditable_seed42/audit.json`
- `results/nmnist_attack_protocol_v3_auditable_seed42/per_sample_results.csv`
- `results/nmnist_attack_protocol_v3_auditable_seed42/records_manifest.json`

### Completed budget-matched comparison

- `results/nmnist_budget_matched_comparison_v2_seed42/STATUS.json`
- `results/nmnist_budget_matched_comparison_v2_seed42/audit.json`
- `results/nmnist_budget_matched_comparison_v2_seed42/budget_definitions.json`
- `results/nmnist_budget_matched_comparison_v2_seed42/wallclock_calibration.json`
- `results/nmnist_budget_matched_comparison_v2_seed42/condition_summaries.csv`
- `results/nmnist_budget_matched_comparison_v2_seed42/paired_comparisons.json`
- `results/nmnist_budget_matched_comparison_v2_seed42/report.md`

### Runners and tests

- `scripts/run_nmnist_qsnn_v3_multiseed.py`
- `scripts/run_nmnist_attack_protocol_v3_auditable_seed42.py`
- `scripts/run_nmnist_budget_matched_comparison_v2_seed42.py`
- `scripts/audit_nmnist_attack_protocol_v3_auditable_seed42.py`
- `tests/test_nmnist_canonical_attack_preprocessing.py`
- `tests/test_nmnist_attack_auditor.py`
- `tests/test_nmnist_budget_matched_atomic_writer.py`

## Bottom line

The repository contains completed five-seed QSNN-v3 validation evidence, a completed five-seed SNN reference, an independently audited canonical seed-42 attack protocol, and a completed 1,600-record budget-matched PGD/TEMP-DRIFT-v2 comparison. The evidence shows small, non-significant ASR differences and higher TEMP timestamp/frame distortion in the reported QSNN wall-clock cell. It does not establish attack superiority, QSNN robustness superiority, or a multi-seed robustness claim.
