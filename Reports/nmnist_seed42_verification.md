# N-MNIST seed-42 independent verification

**Date:** 2026-09-25  
**Scope:** seed 42 only; Binary occupancy representation; existing custom SNN and frozen attack artifacts  
**Reference:** Yu et al., *Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks*, arXiv:2602.03284v1

## Verdicts

| Gate | Verdict | Meaning |
|---|---|---|
| Clean-model validity | **PASS** | Independent full-test accuracy is **98.73%**, above the 95% gate; architecture, checkpoint selection, preprocessing, labels, splits, duplicate/overlap checks, and evaluation checks passed. |
| Attack validity | **PASS — local frozen protocol** | All nine seed-42 Binary cells passed independent artifact reconstruction, conservation, exact active-budget, prediction, denominator, and ASR checks. |
| Paper comparability | **NON_COMPARABLE** | The local temporal binning, victim architecture/checkpoint, and attacked-subset selection differ from the paper. Local ASRs must not be represented as a replication or direct performance comparison. |

No retraining or attack rerun was needed. Existing checkpoints and historical results were not modified.

## 1. Model, checkpoint, and training provenance

The verified model is `NMNISTConvSNN(decay=0.5, n_classes=10)`:

1. bias-free `Conv2d(2,16,3,padding=1)` + `BatchNorm2d(16)` + subtractive-reset LIF + 2×2 average pooling;
2. bias-free `Conv2d(16,32,3,padding=1)` + `BatchNorm2d(32)` + subtractive-reset LIF + 2×2 average pooling;
3. `Linear(32×8×8,10)` applied to second-layer spikes and averaged logits over 10 time steps.

The LIF decay is 0.5, threshold is 1.0, and the surrogate derivative is \((1+5|x|)^{-2}\). Programmatic trainable-parameter counting produced:

| Component | Parameters |
|---|---:|
| Conv1 | 288 |
| BatchNorm1 affine parameters | 32 |
| Conv2 | 4,608 |
| BatchNorm2 affine parameters | 64 |
| Classifier | 20,490 |
| **Total** | **25,482** |

- Checkpoint: `checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt`
- Checkpoint SHA-256: `367cbb76c7dd690c0efafa254d180aa7b4fb28de4f5b21722afa6f20c97eccc6`
- Strict state-dict loading: PASS
- Embedded seed: 42
- Original training: Adam, learning rate 0.001, weight decay 0.0001, gradient clipping 1.0, maximum 15 epochs, historical batch size **128**.
- Selection rule: highest validation accuracy, then lowest validation loss. Epoch 15 was independently recovered as the selected epoch from the 15-row history and matches the checkpoint (`validation accuracy=98.20%`, loss 0.06293208).
- Original epoch-15 training accuracy: 98.7745%. Fresh evaluation of the frozen selected checkpoint on the complete 55,000-sample training subset gives 98.9055%; this is expected to differ from the online training-mode epoch metric.

**Provenance defect:** `configs/nmnist_snn_clean_seed42.json` describes count frames and batch size 128, but the Binary training runner did not load it. The effective Binary configuration was hard-coded and embedded in the checkpoint. The checkpoint itself is sufficiently specific and reproducible for this verification, but the standalone config must not be cited as the effective Binary config.

## 2. Dataset and preprocessing audit

- Dataset loader: Tonic N-MNIST native events.
- Official partitions: 60,000 training and 10,000 test samples.
- Training/validation split: seed-42 stratification of the official training partition into 55,000 training and 5,000 validation samples, exactly 500 validation samples per class.
- Split integrity: training/validation index intersection 0; their union covers all 60,000 official training samples. The official test partition is separate and was not used for checkpoint selection.
- Event layout: `[time, polarity, y, x]`, 10×2×34×34.
- Temporal binning: ten equal-duration timestamp bins, `floor((t-t0)*10/(t_last-t0+1))`, clipped to bins 0–9.
- Representation: Binary occupancy after count binning (`count != 0`), cast to float32 at model input.
- Normalization: none.
- Polarity: preserved as two channels. Labels are the native Tonic targets.
- Independent preprocessing check: 1,000 deterministic samples in each official partition matched an independently coded binning implementation.
- Label check: 0 target mismatches over all 70,000 samples.
- Input-only duplicate hashes (labels deliberately excluded): no raw-event or Binary-grid duplicates within either official partition.
- Train/test overlap: 0 raw-event hash intersections and 0 Binary-grid hash intersections.

These checks found no leakage, duplicate-sample issue, train/test overlap, label error, or invalid checkpoint selection.

## 3. Independent clean evaluation

All new primary evaluations used batch size 64 and evaluation mode. The complete 10,000-sample official test set was evaluated without filtering.

| Split | Samples | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|---:|
| Frozen training subset | 55,000 | 98.9055% | 98.8963% | 0.0393581 |
| Validation subset | 5,000 | 98.2000% | 98.2019% | 0.0629321 |
| **Official test** | **10,000** | **98.7300% (9,873/10,000)** | **98.7183%** | **0.0446116** |

### Per-class test accuracy

| Class | Correct / total | Accuracy |
|---:|---:|---:|
| 0 | 974 / 980 | 99.3878% |
| 1 | 1,129 / 1,135 | 99.4714% |
| 2 | 1,030 / 1,032 | 99.8062% |
| 3 | 1,001 / 1,010 | 99.1089% |
| 4 | 970 / 982 | 98.7780% |
| 5 | 883 / 892 | 98.9910% |
| 6 | 939 / 958 | 98.0167% |
| 7 | 1,014 / 1,028 | 98.6381% |
| 8 | 940 / 974 | 96.5092% |
| 9 | 993 / 1,009 | 98.4143% |

### Test confusion matrix

Rows are true classes and columns are predicted classes, ordered 0–9.

```text
[[ 974,   0,   3,   1,   0,   1,   0,   0,   1,   0],
 [   0,1129,   3,   1,   0,   0,   0,   2,   0,   0],
 [   1,   0,1030,   0,   0,   0,   0,   1,   0,   0],
 [   0,   0,   2,1001,   0,   4,   0,   2,   1,   0],
 [   0,   0,   1,   0, 970,   0,   1,   1,   2,   7],
 [   2,   0,   0,   4,   0, 883,   1,   0,   1,   1],
 [   5,   3,   1,   0,   4,   3, 939,   0,   3,   0],
 [   0,   2,   5,   3,   0,   0,   0,1014,   1,   3],
 [   3,   0,   2,   5,   3,   4,   2,   3, 940,  12],
 [   0,   2,   1,   0,   4,   1,   0,   7,   1, 993]]
```

### Batch-size consistency

Batch-size-1 evaluation was used only for the required consistency check. Batch sizes 1 and 64 both produced 9,873/10,000 correct, with **0 prediction mismatches**. The maximum absolute logit difference was `5.7220459e-06`, which did not alter any argmax. There is no batch-dependent classification inconsistency.

## 4. PIL-PGD verification

The attack is untargeted and white-box. Preserved settings are temperature 1, sign-step size 1, logit clipping ±10, capacity penalty 20, normalized budget penalty 10, 20 iterations for \(B_\infty\), and 40 iterations for \(B_1/B_0\). The historical runner and auditor recorded batch size 64.

The fixed manifest has 1,000 distinct samples selected from all 9,873 independently confirmed clean-correct test samples by a seed/sample-ID SHA-256 order. The exact selected ordering was independently regenerated. One frozen manifest is reused across all nine cells; every denominator is exactly 1,000 clean-correct samples.

| Budget | Successes / denominator | Verified ASR | Realized \(B_\infty\) range | Realized \(B_1\) range | Realized \(B_0\) range |
|---|---:|---:|---:|---:|---:|
| \(B_\infty=1\) | 829 / 1,000 | 82.9% | 1–1 | 568–2,604 | 568–2,604 |
| \(B_\infty=2\) | 1,000 / 1,000 | 100.0% | 2–2 | 840–4,231 | 573–2,766 |
| \(B_\infty=3\) | 1,000 / 1,000 | 100.0% | 3–3 | 1,092–5,545 | 576–2,841 |
| \(B_1=500\) | 1,000 / 1,000 | 100.0% | 3–9 | 500–500 | 268–463 |
| \(B_1=750\) | 1,000 / 1,000 | 100.0% | 4–9 | 750–750 | 364–615 |
| \(B_1=1,000\) | 1,000 / 1,000 | 100.0% | 6–9 | 1,000–1,000 | 438–790 |
| \(B_0=200\) | 1,000 / 1,000 | 100.0% | 8–9 | 683–955 | 200–200 |
| \(B_0=300\) | 1,000 / 1,000 | 100.0% | 8–9 | 1,051–1,403 | 300–300 |
| \(B_0=400\) | 1,000 / 1,000 | 100.0% | 9–9 | 1,369–1,913 | 400–400 |

For every one of the 9,000 records, the independent audit verified source packet identity, unchanged amplitude, packet count, total mass, fixed spatial/polarity event line, temporal domain, collision-free destinations, exact reconstruction, all three realized budget values, equality of stored and recomputed budgets, clean correctness, adversarial prediction, and success. Every active requested budget was realized exactly. No label leakage was found: labels are used only in the expected untargeted white-box loss and clean-correct eligibility; hash ordering does not use label identity or attack outcome.

The suspicious cells are genuine local results: \(B_1=500\) and \(B_0=200\) each independently re-evaluated at 1,000/1,000 successes. In addition, five fixed sample IDs (`5059, 6471, 7404, 862, 5403`) were freshly attacked under both cells; all ten new adversarial tensors and predictions exactly matched their saved counterparts.

## 5. Fair comparison with paper Table 1

| Source / model | Trainable parameters | Clean accuracy | \(B_\infty\) 1 / 2 / 3 | \(B_1\) 500 / 750 / 1k | \(B_0\) 200 / 300 / 400 |
|---|---:|---:|---:|---:|---:|
| Paper ConvNet (`simplenet_v2`) | 299,264 | 99.06% | 100 / 100 / 100 | 58.9 / 99.9 / 100 | 13.0 / 53.1 / 98.5 |
| Paper Spiking ResNet18 | 11,173,386 | 99.62% | 100 / 100 / 100 | 69.2 / 97.4 / 100 | 78.9 / 100 / 100 |
| Paper VGGSNN | 9,227,786 | 99.64% | 98.9 / 100 / 100 | 26.4 / 65.5 / 94.7 | 18.3 / 81.8 / 99.8 |
| **Custom SNN, seed 42 (independently verified; NON_COMPARABLE)** | **25,482** | **98.73%** | **82.9 / 100 / 100** | **100 / 100 / 100** | **100 / 100 / 100** |

The table places the values together for transparent context, not for a controlled model ranking. Three decisive conditions differ:

1. paper preprocessing uses `split_by='number'` (equal-event-count slices), while the local pipeline uses equal-duration timestamp bins;
2. the local 25,482-parameter victim and checkpoint differ from every paper victim/checkpoint;
3. the paper attacks the first 1,000 clean-correct samples in test order, while the local protocol uses deterministic SHA-256 ordering.

Accordingly, architecture alone cannot explain the ASR gaps, and no direct replication, superiority, or inferiority claim is supported.

## 6. Problems and corrective actions

1. **Configuration provenance:** make the Binary runner consume a dedicated immutable config rather than hard-coding it; do not reuse the count-frame config name.
2. **Historical attack metadata:** it lacks an attack-source/config hash and stores an abbreviated command. Future runs should serialize complete interpreter/command/environment, source hash, and canonical config hash. Current results remain locally validated because artifacts were independently reconstructed and the two suspicious cells also have exact fresh reproduction evidence.
3. **Paper comparability:** a genuine replication requires equal-event-count preprocessing, the exact paper model/checkpoint, and the paper's first-1,000 clean-correct selection policy. Do not relabel these local results.

## 7. Evidence

- Clean structured result: `Reports/results/nmnist_seed42_verification/seed42_clean_verification.json`
- Frozen predictions: `Reports/results/nmnist_seed42_verification/seed42_clean_predictions.npz`
- Attack structured result: `Reports/results/nmnist_seed42_verification/seed42_attack_verification.json`
- Fresh suspicious-cell reproduction: `Reports/results/nmnist_table1_audit/small_subset_reproduction.json`
- Clean verifier: `scripts/verify_nmnist_seed42_clean.py`
- Attack verifier: `scripts/verify_nmnist_seed42_attacks.py`
- Timestamped logs: `Reports/logs/nmnist_seed42_verification/`
- Original training history: `Reports/logs/nmnist_binary_true/seed42_history.csv`
- Broader protocol audit: `Reports/nmnist_table1_audit.md`

Methodological audit procedure informed by: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. <https://doi.org/10.48550/arXiv.2609.00065>
