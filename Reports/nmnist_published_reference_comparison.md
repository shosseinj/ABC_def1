# N-MNIST Binary: custom SNN beside published reference values

**Reference:** Yu et al., *Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks*, arXiv:2602.03284v1, Table 1, https://arxiv.org/pdf/2602.03284v1. The first three rows below are transcribed reference results. The fourth row is measured in this repository. No reference-model checkpoint was run locally.

| Source | Model | Clean accuracy | B∞ 1 | 2 | 3 | B1 500 | 750 | 1000 | B0 200 | 300 | 400 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper Table 1 | ConvNet | 99.06 | 100 | 100 | 100 | 58.9 | 99.9 | 100 | 13.0 | 53.1 | 98.5 |
| Paper Table 1 | Spiking ResNet18 | 99.62 | 100 | 100 | 100 | 69.2 | 97.4 | 100 | 78.9 | 100 | 100 |
| Paper Table 1 | VGGSNN | 99.64 | 98.9 | 100 | 100 | 26.4 | 65.5 | 94.7 | 18.3 | 81.8 | 99.8 |
| Local measured, seed 42 | Custom SNN | 98.56 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 99.4 | 100.0 | 100.0 |

All attack entries are ASR percentages. The local row uses the official N-MNIST test set; Binary occupancy; ten equal-event-count temporal bins; first 1,000 clean-correct test examples in official order; untargeted white-box attack; the paper's nine Binary budgets and step counts; and an official-source PIL-PGD adapter. The one frozen local manifest is reused across all nine budgets. Each cell has a clean-correct denominator of 1,000, a stored numerator, and an independent reconstruction/prediction audit with `PASS`.

**Interpretation:** This is a published-reference comparison with substantial evaluation-protocol alignment, not a controlled architecture experiment or a replication of the paper's training. The custom SNN was trained and selected locally (seed 42, 25,482 parameters, 98.56% clean accuracy); the paper's models and checkpoints have different architectures and training provenance. Different victim models are the subject of comparison, but training and clean-accuracy differences can also affect ASR. The paper does not supply per-sample predictions here, so a paired confidence interval or matched-sample four-model test cannot be calculated from these published summary values. The high local ASRs are valid for the audited local model; they do not isolate an architectural cause.

**Excluded:** The older 98.73%-accuracy Binary row in `readme_jafar.md` used equal-duration bins and a SHA-ordered manifest, and must not be used for this published comparison. The existing local Integer rows also use equal-duration bins and remain `NON_COMPARABLE` with paper Table 2. A separate paper-aligned Integer model/evaluation is needed for that table.

Local evidence: `Reports/results/nmnist_seed42_paper_aligned/seed42_clean_result.json`, `seed42_attack_manifest.json`, and the nine `attacks/seed42_paper_aligned_*.json`, `.npz`, and `.audit.json` artifacts with completion markers under `Reports/checkpoints/nmnist_seed42_paper_aligned/`. Projection equivalence and mixed-prefix provenance are described in `Reports/nmnist_paper_aligned_projection_speedup.md`.
