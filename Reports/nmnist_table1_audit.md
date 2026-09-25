# Audit of the N-MNIST Binary Table 1 claims

**Audit date:** 2026-09-25  
**Reference:** Yu et al., *Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks*, arXiv:2602.03284v1  
**Scope:** the seed-42 Binary N-MNIST rows in `readme_jafar.md`, with emphasis on the reported 100% ASRs at \(B_1\in\{500,750,1000\}\) and \(B_0\in\{200,300,400\}\).

## Verdict

The seed-42 100% values are **internally reproducible under this repository's local protocol**. I found no Binary-grid packet-conservation, collision, active-budget, ASR-denominator, or label-leakage error that would invalidate those local measurements.

They are nevertheless **`NON_COMPARABLE` to the paper's Table 1 ConvNet row**. The decisive mismatches are:

1. **Temporal preprocessing:** the official implementation uses ten equal-event-count slices (`split_by='number'`); the local benchmark uses ten equal-duration timestamp bins.
2. **Victim model:** the local 25,482-parameter `NMNISTConvSNN` is not the official repository's much larger N-MNIST `SimpleNet`/ConvNet implementation.
3. **Attacked subset:** the official runner attacks the first 1,000 correctly classified samples encountered in test order; the local runner selects 1,000 clean-correct samples by a seed-dependent SHA-256 ordering.
4. **Checkpoint/training provenance:** the local seed-42 checkpoint is repository-trained and is not the paper's checkpoint.

Consequently, the gap between local \(100\%\) and the paper's \(58.9\%\) at \(B_1=500\), or between local \(100\%\) and paper \(13.0\%\) at \(B_0=200\), is **not evidence of a paper replication, an improvement over the paper, or an architecture-only effect**. Architecture is a plausible contributor, but preprocessing, model training, checkpoint, and sample selection all change simultaneously.

## What matches the paper

| Item | Paper / official implementation | Local seed-42 run | Assessment |
|---|---|---|---|
| Dataset | N-MNIST official test set | N-MNIST official test set | Match |
| Time steps | \(T=10\) | \(T=10\) | Match |
| Attack | Untargeted, white-box PIL-PGD | Untargeted, white-box PIL-PGD | Match at method level |
| Hyperparameters | \(\kappa=1\), \(\alpha=1\), logit clip 10, 40 iterations for \(B_1/B_0\), \(\lambda_{cap}=20\), \(\lambda_{budget}=10\) | Same values | Match |
| Binary packet | One occupied event-line/time cell | One occupied event-line/time cell | Match |
| Capacity | At most one packet per line/time bin | Enforced by strict projection | Match |
| \(B_1\) | Unweighted sum of packet displacements | Same | Match |
| \(B_0\) | Number of moved packets | Same | Match |
| ASR denominator | Correctly classified attacked samples | Frozen 1,000 clean-correct samples | Match in definition |

For Binary grids, an occupied cell has value one, so the paper's packet wording and the repository's nonzero-cell packet contract coincide. The suspicious realized counts are therefore not, by themselves, budget violations.

## Decisive protocol mismatches

### 1. Different temporal grids

The paper states \(T=10\), while the official code makes the missing preprocessing detail explicit:

```python
NMNIST(..., data_type='frame', frames_number=T, split_by='number')
```

SpikingJelly's `split_by='number'` divides the ordered raw events into ten equal-event-count segments (with the remainder in the last segment). In contrast, local `events_to_frames` maps timestamps into ten equal-duration bins:

```python
bins = ((t - t[0]) * temporal_bins) // (t[-1] - t[0] + 1)
```

This changes packet locations and packet counts after Binary occupancy thresholding. On the five fixed audit samples, the Jaccard overlap between the local grid and an independently constructed equal-event-count grid was only 0.515–0.683:

| Sample | Local occupied cells | Event-count occupied cells | XOR cells | Jaccard |
|---:|---:|---:|---:|---:|
| 5059 | 2,038 | 2,152 | 788 | 0.683 |
| 6471 | 1,549 | 1,607 | 694 | 0.639 |
| 7404 | 1,131 | 1,158 | 493 | 0.646 |
| 862 | 2,294 | 2,469 | 1,115 | 0.621 |
| 5403 | 1,992 | 2,134 | 1,322 | 0.515 |

Across the local seed-42 manifest, the mean packet count is 1,878.6. Thus \(B_0=400\) can touch up to 21.3% of the local sample's mean packet count, whereas the paper reports that \(B_0=400\) corresponds to 14.2% of N-MNIST spikes. This confirms a materially different perturbation scale relative to the represented input.

### 2. Different model

Local `models/nmnist_snn.py` uses two convolutional LIF stages with 16/32 channels, average pooling, and 25,482 trainable parameters. The paper's configured N-MNIST ConvNet is `simplenet_v2`, with exactly 299,264 trainable parameters; the separate generic `simplenet` v1 definition would incorrectly yield the older approximately 2.40 million estimate. No evidence links the local checkpoint to the paper checkpoint.

### 3. Different attacked sample set

Official `test.py` iterates the unshuffled test loader, skips clean errors, and stops after 1,000 attacked N-MNIST samples. The local manifest instead chooses the first 1,000 under:

```text
SHA-256("nmnist_binary_true_attacks:<seed>:<sample_id>")
```

among all 9,873 seed-42 Binary-clean-correct test samples. This is deterministic and outcome-independent, but it is not the paper/official-code subset.

## Independent artifact and reproduction checks

### Full saved cells

The existing artifacts contain 1,000 records in each suspicious cell:

- `B1=500`: 1,000/1,000 attacks succeed; every record realizes exactly \(B_1=500\). Realized \(B_0\) ranges 268–463 (median 381), and realized \(B_\infty\) ranges 3–9 (median 8).
- `B0=200`: 1,000/1,000 attacks succeed; every record realizes exactly \(B_0=200\). Realized \(B_1\) ranges 683–955 (median 825), and realized \(B_\infty\) ranges 8–9 (median 9).

The large \(B_1\) values in a \(B_0\)-bounded run are legal because no \(B_1\) constraint is active there. Likewise, moving hundreds of packets in a \(B_1=500\) run is legal when many moves are one bin.

### Fresh fixed-subset reproduction

I reran \(B_1=500\) and \(B_0=200\) from clean inputs for fixed sample IDs `5059, 6471, 7404, 862, 5403` (ten fresh attacks total). All ten:

- reproduced the exact saved adversarial tensor and prediction;
- remained clean-correct before attack and misclassified after attack;
- preserved packet count, total mass, and packet amplitudes;
- preserved event-line identity and stayed within bins 0–9;
- were collision-free;
- satisfied the requested active budget.

Structured evidence: `Reports/results/nmnist_table1_audit/small_subset_reproduction.json`  
SHA-256: `e400f724e72714824aae18d122c06b1c59206e2618346d9eaf153ccd4f696ad3`

Reproduction command:

```powershell
& 'C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe' scripts/reproduce_nmnist_table1_audit_subset.py
```

## Projection, conservation, ASR, and leakage findings

### Projection and conservation

`strict_project_grid` moves complete nonzero Binary cells on the same flattened spatial/polarity line. It reserves unmoved origins, prevents duplicate destinations, restricts targets to the ten-bin domain, and charges global per-sample \(B_1\) or \(B_0\) counters. The runner reconstructs each output from source identity and displacement before serialization. The independent auditor reconstructs it again and re-evaluates the model.

No projection or conservation failure was found in the inspected cells or fresh reproductions.

### ASR denominator and sample selection

The seed-42 manifest is frozen and reused across budget cells. It contains exactly 1,000 samples, each recorded and re-evaluated as clean-correct. ASR is computed as adversarial misclassifications divided by these 1,000 samples. Thus the reported 100% cells correspond to 1,000/1,000, not to a changing successful-only denominator.

### Label leakage

No label leakage was found. Ground-truth labels are used in cross-entropy to construct an untargeted white-box attack, which is expected. Manifest selection depends only on clean correctness and a hash of seed/sample ID; it does not depend on adversarial outcomes or target labels. Labels, clean predictions, and adversarial predictions are independently checked during audit.

## Limits of the existing `PASS` audits

The current `.audit.json` files are strong **internal structural/prediction audits**, but their `PASS` status does not establish paper comparability. They do not independently verify:

- raw-event sample identity by reloading the original dataset;
- equivalence to the paper's `split_by='number'` preprocessing;
- an attack/configuration hash (the run metadata has no configuration hash);
- an attack-source hash;
- equivalence of model architecture/training/checkpoint to the paper;
- equality of stored `reported_b_*` arrays to independently recomputed values (the auditor recomputes budgets but does not compare those arrays);
- a complete executable command/environment record (`exact_command` is only `scripts/run_nmnist_binary_true_attacks.py --all`).

These omissions do not overturn the local results, but they make the existing `PASS` narrower than the repository's full provenance contract.

## Classification of the discrepancy

| Category | Finding |
|---|---|
| Confirmed attack/projection implementation error | **None found for the audited Binary cells** |
| Confirmed protocol mismatch | **Yes:** preprocessing, model/checkpoint, and attacked subset |
| Unsupported claim | Any interpretation of the local row as a Table 1 replication or directly comparable improvement |
| Plausible architecture effect | Yes, but not identifiable separately from the other mismatches |
| Locally valid high ASR | **Yes**, for the frozen local protocol and seed-42 checkpoint |

## Corrections and required follow-up

1. **Completed:** local N-MNIST attack rows in README Table 1 are labeled **`NON_COMPARABLE`**, with paper values treated as contextual references only.
2. Do not describe the local 100% cells as reproducing or outperforming the paper.
3. For a true replication, regenerate N-MNIST frames with `split_by='number'`, use the paper/official ConvNet architecture and matching training/checkpoint, and freeze the first 1,000 clean-correct samples in official test order before any budget evaluation.
4. Add configuration and source hashes, raw-dataset/sample verification, stored-vs-recomputed budget checks, and a complete command/environment record to the independent auditor.
5. Keep the current artifacts unchanged as valid results for the explicitly named **local equal-duration, compact-SNN protocol**.

## Evidence reviewed

- Paper HTML, sections 4.1–4.3, section 5, Table 1, and Appendix D: <https://arxiv.org/html/2602.03284v1>
- Official implementation: <https://github.com/yuyi-sd/Spike-Retiming-Attacks>, commit `19f42e63d31cbdc4ef2d5ef7b6e40716f92531c3`
- `attacks/pil_pgd.py`
- `scripts/run_nmnist_binary_true_attacks.py`
- `scripts/audit_nmnist_binary_true_attacks.py`
- `experiments/nmnist/snn_baseline.py`
- `models/nmnist_snn.py`
- seed-42 manifest, cache, run NPZ files, metadata, and full audit JSON files for `B1=500` and `B0=200`
- `Reports/results/nmnist_binary_true_attacks/asr_by_seed.csv`

Validation: `tests/test_benchmark_contract.py` passed 6/6; the reproduction script compiled successfully.
