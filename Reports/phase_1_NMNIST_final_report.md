# Phase 1 N-MNIST Final Report

**Current status: RUNNING; this report is not final until all 66 seed/representation/budget cells pass independent audit.**

The authoritative contract now defines one nonzero temporal-grid cell as one indivisible amplitude-bearing packet. Counts are not expanded; collisions, merging, accumulation, splitting, creation, and deletion are forbidden. Realized B∞, B1, and B0 are measured over unweighted cell-packet displacement.

Completed prerequisites:

- corrected contract/regression suite: 17/17 PASS;
- frozen-manifest feasibility: 1000/1000 for seeds 42, 123, and 777;
- deterministic seed-42 dry run: 4/4 independent audits PASS, excluded from benchmark results.

The real benchmark is running resumably in seed order 42, 123, 777. No partial cell is reported as a benchmark result; each completed cell requires attack validation, atomic serialization, independent dataset/checkpoint reconstruction, prediction verification, and audit PASS.
