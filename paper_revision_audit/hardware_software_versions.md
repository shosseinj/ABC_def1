# Hardware and software evidence

Read-only inspection on 2026-09-28 used the repository's specified interpreter. These are **current environment** values; absent a saved run-time environment snapshot they are not proof that every historical run used identical drivers or packages.

| Item | Observed value | Source and qualification |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090; 25,756,696,576 bytes addressable VRAM | `torch.cuda.get_device_name(0)` and device properties, current environment |
| CPU | AMD64 Family 25 Model 97 Stepping 2; 24 logical processors | `platform.processor()` and `os.cpu_count()`; marketing name and physical core count NOT VERIFIED |
| RAM | NOT VERIFIED | CIM access denied; no reliable saved run-time total located |
| OS | Windows build 26200, 64-bit Python environment | `platform.platform()` returned `Windows-10-10.0.26200-SP0`; marketed Windows edition NOT VERIFIED |
| Python | 3.10.4, 64-bit | specified `.venv/Scripts/python.exe`; `results/nmnist_budget_matched_comparison_v2_seed42/STATUS.json` agrees |
| PyTorch | 2.5.1+cu124 | current interpreter import |
| PyTorch CUDA runtime | 12.4 | `torch.version.cuda`; distinct from driver and toolkit |
| cuDNN | 9.1.0 (`90100`) | `torch.backends.cudnn.version()` |
| CUDA toolkit | 12.4, nvcc V12.4.99 | `nvcc --version`, current PATH |
| NVIDIA driver | NOT VERIFIED | `nvidia-smi` unavailable on PATH; CUDA availability does not identify driver version |
| NumPy / SciPy / scikit-learn | 1.26.4 / 1.15.3 / 1.7.2 | current interpreter imports |
| tonic / Numba | 1.6.0 / 0.67.0 | current interpreter imports |
| Quantum backend | batched analytic PyTorch state-vector, complex64 | `models/nmnist_hybrid_qsnn.py:246-264`; no quantum hardware or shot-noise claim |

The newer four-model training was logged on CUDA; exact environment and checkpoint hashes belong to the per-run JSON under `Reports/results/nmnist_controlled_four_models_lr1e4/`. The QSNN campaign's seed-42 checkpoint reports a 1200-second timeout and best epoch 37, so total training runtime should not be inferred from a single artifact (`results/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/summary.json`).
