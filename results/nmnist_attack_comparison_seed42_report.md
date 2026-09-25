# N-MNIST Seed-42 Validation Attack Comparison

Validation-only comparison on the same 100 clean-correct samples. The official test partition was not instantiated.

- Checkpoint: `checkpoints/nmnist_qsnn_ablation/T4_S8_Q12_B6_best.pt` (`c68065369bdad33af3c98b9b0a9045722f05cb95f00e2649479990b8e8318e75`)
- Split: `results/nmnist_snn_multiseed_split.json` (`a12176ce117ab9a85dd29d277901f8617ad2b5427dd3f976dbba436942f70198`)
- Temporal span: inclusive clean duration `tlast - t0 + 1`; each epsilon is a fraction of this span.
- Margin change: clean true-class margin minus attacked true-class margin.
- Trace metric: `sqrt(1-F)` for clean/attacked final pure premeasurement circuit states.

| Epsilon | Attack | ASR | Attacked Acc | 1-Fidelity | Trace Distance | Feasibility | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| 2% | PGD | 0.0800 | 0.9200 | 0.042797 | 0.177274 | 1.0000 | 973.34 s |
| 2% | TEMP-DRIFT | 0.2200 | 0.7800 | 0.073987 | 0.263669 | 1.0000 | 662.88 s |
| 5% | PGD | 0.3200 | 0.6800 | 0.274971 | 0.479222 | 1.0000 | 971.15 s |
| 5% | TEMP-DRIFT | 0.7600 | 0.2400 | 0.422699 | 0.641635 | 1.0000 | 663.78 s |
| 10% | PGD | 0.6300 | 0.3700 | 0.663569 | 0.792713 | 1.0000 | 972.49 s |
| 10% | TEMP-DRIFT | 0.9900 | 0.0100 | 0.890414 | 0.942616 | 1.0000 | 663.52 s |

| Epsilon | PGD-only | TEMP-only | Both Success | Both Robust |
|---|---:|---:|---:|---:|
| 2% | 0 | 14 | 8 | 78 |
| 5% | 0 | 44 | 32 | 24 |
| 10% | 0 | 36 | 63 | 1 |

Successful-attack-only drift and exact evaluation counts are included in the JSON and comparison CSV.

**TEMP-DRIFT: STRONGER**

Single-seed exploratory validation evidence; no robustness or superiority claim.
