# Phase 0 — Repository inspection

**Inspection time:** 2026-09-20T10:50:53.165097+00:00  
**Outcome:** Repository inventory completed; no benchmark experiment was run.

## Environment

- Configured interpreter: `C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe` (Python 3.10.4)
- PyTorch: 2.5.1+cu124; CUDA available: True; CUDA runtime: 12.4
- GPU(s): NVIDIA GeForce RTX 4090
- Free disk: 266.7 GB
- Dependency versions are recorded in `Reports/repository_inventory.json` and `Reports/logs/phase_0_environment.log`.

## Repository inventory

- Models: 8 Python modules, including classical SNN and PennyLane QSNN paths.
- Checkpoints: 79 files; every path, size, and SHA-256 is recorded in the machine-readable inventory.
- Configurations: 22 JSON files, each hashed.
- Entry points: 83 Python scripts; 45 Python test files.
- Attack code includes legacy TEMP-DRIFT variants plus the new independent `ResearchLoop/core/projector.py` and `audit.py` contract utilities.
- Defenses currently consist primarily of `defenses/quantum_temp.py`; locked-protocol defense evaluation has not run.

## Dataset storage

| Dataset | Path | Files | Bytes |
|---|---|---:|---:|
| N-MNIST | `data/mnist/NMNIST` | 0 | 0 |
| DVS-Gesture | `data/dvs_gesture/DVSGesture` | 1343 | 19118389738 |
| CIFAR10-DVS | `data/cifar10_dvs/CIFAR10DVS` | 10012 | 37297164205 |
| SHD (prior work; outside benchmark) | `data/shd/SHD` | 4 | 516550398 |
| MNIST (prior work; outside benchmark) | `data/MNIST` | 8 | 66544770 |

Presence does not imply completeness. N-MNIST, DVS-Gesture, and CIFAR10-DVS directories contain archives/extracted material, but only N-MNIST has five repository SNN checkpoints. DVS-Gesture has no benchmark checkpoint; CIFAR10-DVS has seed-42 checkpoints only.

## Representation and provenance findings

Prior N-MNIST code uses Tonic event data and `[time, polarity, y, x]` framed representations, commonly with 10 temporal bins. Existing attack results use earlier epsilon/query-matched protocols and are **NON_COMPARABLE** to the locked paper benchmark unless every matching field is independently established. The amended benchmark contract defines one indivisible amplitude-bearing packet per nonzero temporal cell and requires fixed event lines, discrete bins `[0,10)`, collision-free cell-packet capacity-1, unweighted displacement budgets, and clean-correct ASR.

## Gaps that affect later phases

- No TEMP-DRIFT benchmark run under the locked paper protocol is complete.
- DVS-Gesture has no repository benchmark model/checkpoint or dataset-specific attack runner.
- CIFAR10-DVS has only seed-42 baseline checkpoints, not five frozen seeds.
- N-MNIST prior attack outputs use a different epsilon/query protocol and cannot be relabeled as paper-comparable.
- Raw-event preprocessing must be reconciled with the locked T=10 capacity-1 packet representation before specification freeze.
- requirements.txt pins torch 2.9.1/torchvision 0.24.1, while the configured environment has torch 2.5.1+cu124/torchvision 0.20.1+cu124.

Phase 0 is an inspection gate only. These gaps do not invalidate this inventory, but they must be resolved by the benchmark-contract and specification-freeze phases before dataset experiments.
