# Phase 0.75 — Specification freeze

**Status: PASS (independent gate).** The Phase 0.75 gate returned `PASS` with no errors. The active benchmark uses exactly seeds 42, 123, and 777. Nine of nine required checkpoints passed strict loading, architecture, parameter-count, seed, hash, config/split provenance, and clean evaluation checks. Binary and integer clean accuracies, manifest denominators, checkpoint hashes, and manifest hashes are listed in `Reports/benchmark_specification.md` and machine-readable validation is in `Reports/results/benchmark_checkpoint_validation.json`.

N-MNIST official archives are verified in `Reports/results/nmnist_dataset_verification.json`. DVS-Gesture uses official train/test partitions. CIFAR10-DVS uses the frozen stratified 80/10/10 split because Tonic provides no official split. The exact timestamp, preprocessing, budgets, projection, ASR, manifest, auditor, three-seed statistics, and reference provenance policies are frozen in the specification.

The final contract suite executed 13 tests with zero failures. It includes all budget projectors, independent retiming audit, packet/line/value/timeline/capacity rejection, deterministic manifests, and proof that the final returned strict tensor is exactly model-evaluated. No locked-protocol attack result was generated during this freeze phase.

Gate evidence: `Reports/logs/phase_0_75_gate_attempt_3.log`. Phase 1 subsequently exposed an obsolete unit-packet wording conflict; the authoritative contract amendment resolved it in favor of indivisible amplitude-bearing nonzero-cell packets. This does not alter the recorded external Phase 0.75 gate outcome or require rerunning the phase.
