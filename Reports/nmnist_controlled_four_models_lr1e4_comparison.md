# Controlled local four-model N-MNIST comparison

All values below are new local measurements from the same seed-42 data split, training recipe, equal-event-count T=10 representation, validation-selected checkpoints, and the first 1,000 official test examples jointly classified correctly by all four models. Every attack cell passed independent raw-data reconstruction and prediction audit. The paper's published numbers are not included in this controlled table.

## Binary

| Model | Clean accuracy (%) | B_inf=1 ASR (%) | B0=200 ASR (%) |
|---|---:|---:|---:|
| custom | 97.99 | 100.0 | 99.5 |
| convnet | 96.90 | 1.9 | 1.2 |
| resnet18 | 98.61 | 0.1 | 0.0 |
| vggsnn | 99.21 | 0.0 | 0.0 |

Joint clean-correct pool: 9522 samples; first 1,000 frozen. Manifest SHA-256: `ecb866405e191304f36d3de3f8447d08b02913ffee7079f3b0f3b8a63f2b46a5`.

## Integer

| Model | Clean accuracy (%) | B_inf=1 ASR (%) | B0=200 ASR (%) |
|---|---:|---:|---:|
| custom | 98.14 | 99.9 | 98.0 |
| convnet | 96.58 | 0.6 | 1.1 |
| resnet18 | 98.12 | 0.2 | 0.1 |
| vggsnn | 99.30 | 0.0 | 0.0 |

Joint clean-correct pool: 9495 samples; first 1,000 frozen. Manifest SHA-256: `b784862be9f353691cd1902466331ab5e53f9dde4f3dfa39de5a0d5932587a34`.

The four architectures differ by design; the dataset, train/validation/test split, preprocessing, seed, optimizer settings, checkpoint rule, attacked sample IDs, attack implementation, and budgets are held constant. Lower ASR means greater robustness under this local attack. This is one seed; it does not establish multi-seed uncertainty. The official source hard-codes the B0 penalty at 5, whereas the paper's general text states 10. This source/paper ambiguity is shared across all local models and is not resolved by these measurements.

Evidence: `Reports/results/nmnist_controlled_four_models_lr1e4/`, `Reports/checkpoints/nmnist_controlled_four_models_lr1e4/`, and `Reports/logs/nmnist_controlled_four_models_lr1e4/`.
