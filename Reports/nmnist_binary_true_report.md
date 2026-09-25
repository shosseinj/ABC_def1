# True N-MNIST Binary-grid clean accuracy

Status: **PASS**

The three checkpoints were independently initialized and trained from scratch using Binary occupancy during both training and evaluation. No count-trained weights were loaded.

| Seed | Checkpoint | Full test correct | Full test samples | Accuracy | Collapse check |
|---:|---|---:|---:|---:|---|
| 42 | `checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt` | 9,873 | 10,000 | 98.73% | PASS |
| 123 | `checkpoints/nmnist_binary_true/nmnist_binary_seed123_best.pt` | 9,829 | 10,000 | 98.29% | PASS |
| 777 | `checkpoints/nmnist_binary_true/nmnist_binary_seed777_best.pt` | 9,853 | 10,000 | 98.53% | PASS |

Three-seed clean accuracy: **98.52 ± 0.22%** (sample SD). Two-sided 95% Student-t CI: **97.97–99.06%**.

## Reference-table context

| Model | Binary-grid accuracy |
|---|---:|
| ConvNet | 99.06% |
| ResNet18 | 99.62% |
| VGGSNN | 99.64% |
| **SNN (Ours)** | **98.52 ± 0.22%** |

The corrected mean is 0.54 percentage points below ConvNet, 1.10 points below ResNet18, and 1.12 points below VGGSNN. Architecture and other paper-comparability conditions still need to be considered; accuracy proximity alone does not establish replication.

The old 85.48%, 95.65%, and 80.15% values are preserved as legacy `REPRESENTATION_MISMATCH` evaluations. Their attack ASRs must not be reused with these new checkpoints. True Binary attack manifests and attack results remain pending.
