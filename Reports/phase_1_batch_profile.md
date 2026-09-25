# Phase 1 GPU batch profile

## Baseline runner

The stopped legacy worker used effective attack batch size **1**. Its completed seed-42 binary `B_inf=1` condition measured 15.253 seconds/sample over 1,000 samples. A controlled representative `B_inf=2` profile measured 13.790 seconds/sample.

| Metric | Baseline |
|---|---:|
| Effective attack batch size | 1 |
| GPU utilization, mean | 18.02% |
| GPU utilization, maximum | 44.0% |
| Peak VRAM | 2,377 MiB / 24,564 MiB |
| Process CPU utilization, mean | 99.94% |
| Representative B∞=2 wall time | 13.790 s/sample |

Low GPU utilization was caused primarily by Python/CPU greedy-projection bookkeeping and thousands of synchronous scalar GPU reads/writes, not by VRAM pressure.

## Component profile

The instrumented pre-optimization B∞=1 sample profile measured:

| Component | Seconds/sample |
|---|---:|
| Dataset access + T=10 preprocessing | 0.0559 |
| Model forwards | 0.2387 |
| Optimization excluding forwards/projection | 0.8320 |
| Strict budget projection | 8.8594 |
| Tensor serialization | 0.0049 |
| Per-sample logging with forced flush | 0.0032 |
| Standalone independent audit | 3.1944 |
| Attack total | 9.9300 |

The exact-equivalent bulk-transfer CPU projector reduced the representative projection time to 2.8363 seconds and attack total to 4.1093 seconds. Full five-sample equivalence profiling measured a balanced 4.52× attack speedup across B∞, B1, and B0.

## I/O and synchronization

- The legacy condition wrote no resumable per-sample structured checkpoint; B∞=2 samples 1–52 existed only in RAM and cannot pass reconstruction audit after process termination.
- The optimized runner writes an atomic partial checkpoint every 25 samples and logs progress every 25 samples.
- Frozen T=10 manifest-ordered inputs are cached once per seed and transferred from pinned host memory.
- Model/checkpoint initialization occurs once per seed in the attack runner.
- The main synchronization bottleneck was scalar GPU access inside projection; it was removed by bulk transfer without changing candidate order or projection rules.

Source artifacts: `Reports/results/phase1_runtime_profile_old.json`, `Reports/results/phase1_runtime_profile_optimized_projection.json`, and `Reports/results/phase1_batch_equivalence.json`.
