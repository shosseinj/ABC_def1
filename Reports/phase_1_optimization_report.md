# Phase 1 engineering optimization report

## Status

**EQUIVALENCE PASS. Phase 1 remains suspended and was not resumed.**

The preserved benchmark worker, PID `35132`, remains stopped at seed 42, binary `B_inf=1`, sample 512/1000. It was not terminated, resumed, or modified during equivalence validation.

## Optimization

Profiling identified strict greedy projection as the dominant hot path. The old implementation performed thousands of individual GPU scalar reads and writes while building and applying the candidate ordering. The optimized projector performs one bulk transfer to CPU, applies the **same candidate tuples, lexicographic ordering, origin reservations, collision checks, budget accounting, and assignments**, and transfers the completed strict tensor back in bulk.

No AMP, reduced precision, TF32 change, `torch.compile`, approximate algorithm, reduced iteration count, attack batching, or scientific change was accepted. A proposed batched-attack path failed exact projected-output equivalence and was rejected. Benchmark attacks remain batch-size 1.

The staged runner also supports immutable manifest-ordered T=10 preprocessing caches, pinned host tensors, 25-sample atomic partial checkpoints, and progress logging every 25 samples. These engineering paths require their final regression/cache checks before Phase 1 resumes.

## Exact equivalence workload

- Seed: 42
- Representation: binary
- Samples: `6697`, `415`, `8930`, `1098`, `653`
- Coverage: attack success and failure, different classes, and packet counts from low to high
- Budget conditions: `B_inf=1`, `B1=500`, `B0=200`
- OLD evaluations: 15
- OPTIMIZED evaluations: 15
- Exact comparison/audit items: 15
- Total tracked work: **45/45**
- Persisted elapsed compute time: **1,286.03 seconds (00:21:26)**

All 15 comparisons had exact equality for:

- sample ID and requested beta;
- clean and attacked predictions;
- attack-success flag;
- realized `B_inf`, `B1`, and `B0`;
- final projected temporal representation;
- packet amplitudes;
- packet destinations/displacements;
- collision, mass, and active-budget audit outcome.

Result: **PASS** (`Reports/results/phase1_runtime_equivalence.json`).

## Runtime results

Times below are means over the five frozen samples for each family and include the complete attack evaluation for that sample.

| Budget family | OLD seconds/sample | OPTIMIZED seconds/sample | Speedup |
|---|---:|---:|---:|
| B∞ | 17.997 | 4.433 | 4.06× |
| B1 | 105.652 | 22.866 | 4.62× |
| B0 | 86.932 | 19.317 | 4.50× |
| Balanced overall | **70.193** | **15.538** | **4.52×** |

The balanced matrix totals were 1,052.90 seconds OLD and 233.08 seconds OPTIMIZED.

At the measured family-specific rates, all 66 N-MNIST seed/representation/budget cells would require approximately 303 attack-hours (12.6 days) before audit and fixed overhead. The remaining estimate will be refined after detailed GPU/auditor profiling and the preserved first condition completes.

## Resumability and evidence

- Live/final progress: `Reports/checkpoints/phase1_equivalence_progress.json`
- Atomic per-item evidence: `Reports/checkpoints/phase1_equivalence_items/`
- Human-readable log: `Reports/logs/phase1_equivalence_optimization.log`
- Machine-readable comparisons: `Reports/results/phase1_runtime_equivalence.json`
- Suspended benchmark boundary: `Reports/checkpoints/phase1_pause_state.json`

Phase 1 can proceed to the remaining engineering regression/profile checks. This report does **not** authorize killing the suspended worker and does not claim that benchmark execution has resumed.
