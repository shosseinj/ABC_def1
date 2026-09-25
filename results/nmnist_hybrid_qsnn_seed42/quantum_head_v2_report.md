# N-MNIST Quantum Head V2, Seed 42

Validation-only development on the frozen 55,000/5,000 official-training split. The official test partition was not instantiated. The Conv/LIF frontend and its 8D latent definition were unchanged. Batch size was 256 for every run.

| Variant | Best validation accuracy | Macro-F1 | Best epoch | Parameters | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Preserved v1 joint-probability baseline | 86.16% | 86.10% | 20 | 7,026 | 577.08 |
| V2 optimization-only screen | 64.10% | 64.12% | 11 | 7,026 | 283.26 |
| V2 learned-measurement screen, unstable BN protocol | 63.48% | 63.56% | 12 | 7,042 | 318.74 |
| V2 projected/two-axis/learned-measurement screen | 89.48% | 89.55% | 12 | 7,130 | 369.96 |
| V2 projected/two-axis/learned-measurement full | 92.14% | 92.16% | 30 | 7,130 | 923.07 |
| V2 projected/two-axis/learned-measurement continued (selected) | **92.60%** | **92.61%** | **37** | **7,130** | **1,327.36 total** |
| V2 three-block alternating-topology screen | 88.12% | 88.21% | 12 | 7,154 | 456.63 |

The selected head uses an identity-initialized learned 8-to-8 angle projection, RY/RZ two-axis data upload, two variational re-upload blocks, CNOT-ring entanglement, a learned local measurement basis, all 256 computational-basis probabilities, and a linear classifier. There is no latent-to-logit classical bypass. A two-epoch head-alignment warm-up and fixed pretrained BatchNorm statistics prevented destructive frontend drift.

The selected checkpoint improves over the preserved 86.16% baseline by 6.44 percentage points and remains 1.10 points below the 93.70% classical latent control. The three-block alternating candidate was not promoted because it underperformed the two-block screen.

## Reproduction

Run both commands from the repository root with the required existing environment:

```powershell
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" scripts/run_nmnist_quantum_head_v2_seed42.py --variant project_measure2 --stage full --epochs 30
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" scripts/run_nmnist_quantum_head_v2_seed42.py --variant project_measure2 --stage extend --epochs 15
```
