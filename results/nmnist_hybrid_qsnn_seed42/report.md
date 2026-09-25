# N-MNIST Hybrid QSNN Seed-42 Development

Validation-only experiment on the frozen 55,000/5,000 official-training split. The official test partition was not instantiated.

## Selected Results

| Model | Validation accuracy | Macro-F1 | Best epoch | Parameters | Runtime (s) | Peak GPU bytes |
|---|---:|---:|---:|---:|---:|---:|
| Classical latent control | 0.9370 | 0.9367 | 18 | 4,514 | 284.11 | 1,561,866,752 |
| Two-block hybrid QSNN, joint probability readout | 0.8616 | 0.8610 | 20 | 7,026 | 577.08 | 1,561,922,048 |

The hybrid replaces T4-S8 with 10 full-resolution polarity frames, a two-stage Conv/LIF extractor, an 8-value bounded latent, one latent coordinate per qubit, two data re-upload blocks, trainable RY/RZ gates, CNOT-ring entanglement, full computational-basis probability measurement, and a linear classifier. There is no classical bypass around the quantum circuit.

## Negative Results

The local-Z plus adjacent-ZZ readout peaked at 65.20% validation accuracy. A one-block circuit trained with the extractor frozen peaked at 62.62%. These results isolate quantum encoding/readout and extractor-circuit co-adaptation as remaining bottlenecks.

## Runtime

Batch sizes 32, 64, 128, and 256 were benchmarked with real training steps. Batch 256 was fastest. Selected-QSNN throughput was 3,756.9 samples/s during the benchmark, with 1,837,398,528 peak allocated bytes. Conv/LIF operations used float16 AMP; quantum state evolution used float32/complex64. The environment was PyTorch 2.5.1+cu124 on an RTX 4090.

## Reproduction

```powershell
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" scripts/run_nmnist_hybrid_qsnn_seed42.py --mode all --classical-epochs 25 --quantum-epochs 25 --quantum-variant joint
```
