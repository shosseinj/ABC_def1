# N-MNIST Frontend V3 Seed-42 Development

All experiments use the frozen 55,000/5,000 official-training split. Only `train=True` was instantiated. The timestamp-facing adapter remains separate from frame construction for TEMP-DRIFT compatibility.

| Frontend | Classical screen accuracy | Macro-F1 | Parameters | Runtime (s) |
|---|---:|---:|---:|---:|
| Current 8D `spatial2` control | 93.70% | 93.67% | 4,514 | 284.11 full |
| `spatial4`: 12/24 channels, 4x4 summaries, 16D latent | 97.26% | 97.26% | 15,354 | 144.65 |
| `wide4`: 16/32 channels, 4x4 summaries, 32D latent | 97.42% | 97.42% | 38,122 | 125.91 |
| `spatial8`: 16/32 channels, 8x8 summaries, 32D latent | 97.46% | 97.46% | 136,426 | 123.83 |

The promoted `wide4` frontend was selected because it is within 0.04 points of `spatial8` with 72% fewer parameters. Its full classical run reached **97.80% validation accuracy / 97.80% Macro-F1** at epoch 17, with 38,122 parameters and 273.87 seconds.

## Resulting QSNN

`wide4` feeds the preserved `project_measure2` quantum architecture through an explicit learned 32-to-8 angle projection. The circuit is the only route from latent to logits: two-block RY/RZ data upload, learned measurement basis, full 256-state probability readout, and linear classifier. No classical bypass was added.

| QSNN | Validation accuracy | Macro-F1 | Best epoch | Parameters | Runtime |
|---|---:|---:|---:|---:|---:|
| `wide4` + `project_measure2` | **97.06%** | **97.06%** | **37** | **40,762** | 1,200 s observed; external timeout during epoch 38 |

The best checkpoint was saved at epoch 37 and independently reloaded/evaluated with the same result. Its SHA-256 is `fbc8978785678f9d84b9a26507f59f638809f7e2ef961f889600c45807ca4237`. The process reached epoch 38 before external termination, so the final result JSON was finalized from the saved checkpoint. The official test partition was not accessed.

## Reproduction

```powershell
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" scripts/run_nmnist_frontend_v3_seed42.py --frontend wide4 --model classical --stage full --epochs 30
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" scripts/run_nmnist_frontend_v3_seed42.py --frontend wide4 --model quantum --stage full --epochs 40
```
