# N-MNIST QSNN Controlled Ablation

Validation-only, seed 42, frozen official-training split. The official test partition was not instantiated.

| Run | Temporal Bins | Spatial-Polarity Channels | Qubits | Blocks | Input Features | Params | Best Val Acc | Best Val Loss | Best Epoch | Runtime (s) | Peak GPU Memory (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T4_S8_Q8_B4 | 4 | 8 | 8 | 4 | 32 | 154 | 0.6956 | 1.018668 | 15 | 877.15 | 103738368 |
| T8_S8_Q8_B4 | 8 | 8 | 8 | 4 | 64 | 154 | 0.4832 | 1.587339 | 15 | 993.70 | 103823360 |
| T10_S8_Q8_B4 | 10 | 8 | 8 | 4 | 80 | 154 | 0.3684 | 1.806725 | 15 | 996.27 | 103864320 |
| T4_S16_Q8_B4 | 4 | 16 | 8 | 4 | 64 | 154 | 0.3676 | 1.820386 | 15 | 869.14 | 103823360 |
| T4_S8_Q8_B6 | 4 | 8 | 8 | 6 | 32 | 186 | 0.7588 | 0.850372 | 15 | 957.08 | 120625152 |
| T4_S8_Q12_B6 | 4 | 8 | 12 | 6 | 32 | 274 | 0.7672 | 0.833559 | 15 | 1343.42 | 1321157120 |

- Temporal winner: `T4_S8_Q8_B4`
- Representation winner: `T4_S8_Q8_B4`
- Quantum-capacity winner: `T4_S8_Q12_B6`
- Final selected configuration: `T4_S8_Q12_B6`

This single-seed adaptive development study does not establish statistical superiority. The selected architecture requires prespecified multi-seed validation before any official-test evaluation.
