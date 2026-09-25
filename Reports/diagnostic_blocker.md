# Phase 1 diagnostic blocker — RESOLVED

## Resolution

**RESOLVED on 2026-09-22.** The authoritative `AGENTS.md` now defines one indivisible amplitude-bearing packet per nonzero temporal-grid cell. It explicitly prohibits integer-count expansion, collisions, merging, and accumulation, and defines B∞, B1, and B0 over unweighted cell-packet displacement.

The former blocker was solely the obsolete local unit-packet mandate. No inaccessible dataset, checkpoint, or unresolved scientific definition remains.

## Validation performed

- Updated `ResearchLoop/core/contract.py`, `ResearchLoop/core/audit.py`, and `attacks/pil_pgd.py` to state and enforce cell-packet semantics.
- Added regressions for indivisible high-amplitude packets, unweighted budgets, collision rejection, amplitude/mass preservation, and exact reconstruction.
- Complete contract/regression suite: **PASS, 17/17** (`Reports/logs/phase_1_cell_packet_regression_tests.log`).
- Revalidated all frozen N-MNIST manifests with the configured interpreter.

| Seed | Feasible | Source collision-free | Samples with amplitude > 1 | Maximum amplitude |
|---:|---:|---:|---:|---:|
| 42 | 1000/1000 | 1000/1000 | 1000/1000 | 11 |
| 123 | 1000/1000 | 1000/1000 | 1000/1000 | 12 |
| 777 | 1000/1000 | 1000/1000 | 1000/1000 | 13 |

Machine-readable evidence: `Reports/results/nmnist_representation_feasibility.json`.

## Scientific impact

The Phase 1 N-MNIST benchmark is now scientifically executable under the corrected contract. The seed-42 dry run passed 4/4 independent audits; artifacts under `Reports/results/nmnist_dry_run/` are marked `benchmark_result: false` and cannot be counted as benchmark results. Real cells are accepted only after independent reconstruction and model-prediction audit.
