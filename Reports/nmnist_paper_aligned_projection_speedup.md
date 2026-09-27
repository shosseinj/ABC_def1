# N-MNIST paper-aligned projection speedup — 2026-09-27

## Scope and result

The official-source `B_inf` final strict projection was moved from a per-sample Python/CUDA synchronization loop to one CUDA thread per independent spatial/polarity event line. The upstream attack objective, iterations, relaxed projection, model, manifest, and active budget remain unchanged. The greedy choice within each event line uses the same highest score, earliest source-time tie order, and earliest displacement-index tie order. Distinct lines have no collision or budget interaction in `B_inf`.

All nine N-MNIST Binary seed-42 paper-aligned budget conditions now have completion markers and independent audits with `PASS`. Each uses a frozen 1,000-sample clean-correct manifest. `B0=200` achieved 994/1,000 (99.4% ASR); the other eight conditions achieved 1,000/1,000 (100% ASR).

For `B_inf=3`, the first 768 records were preserved from the earlier per-sample strict projector; the final 232 were computed with the new CUDA projector. This mixed-prefix fact is recorded in the run metadata. The independent auditor checked all 1,000 records. Exact attack equivalence between the two adapters was tested on four real samples, not on all 1,000.

## Tests and timing

- Projection output and per-packet displacement matched the upstream per-sample projector exactly in 36/36 `B_inf` cases: random sparse grids at radii 1/2/3, uniform ties, and real N-MNIST samples.
- Full 20-step attack on four frozen, clean-correct N-MNIST samples matched exactly for projected tensors and displacement: old 25.407 s; new 0.943 s, a measured 26.9x speedup for this four-sample check.
- The resumed 232-sample suffix took 6.3 s of attack-loop time. Its first full 64-sample batch took 2.0 s, versus 394–467 s for earlier 64-sample batches. Those batches contain different samples, so the comparison is indicative rather than a controlled speedup measurement.
- The independent `B_inf=3` artifact audit returned `PASS`, 1,000/1,000 records, zero errors.
- The global-budget projector's compact, per-sample candidate sorting now matches upstream tie ordering; the full projector suite passed 72/72 cases, including uniform ties. All three `B1` and three `B0` conditions were independently audited `PASS`.

## Provenance and limits

- Required Python interpreter: `C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe`.
- Test commands: `python.exe -u scripts/test_nmnist_paper_aligned_batched_projection.py --binf-only`, `--attack-equivalence`, `--benchmark-only`, and the default full suite, all with the interpreter above.
- Resume command: `python.exe -u scripts/run_nmnist_seed42_paper_aligned_attacks.py --budget-type B_inf --budget 3` with the same interpreter.
- Projection adapter SHA-256: `0daa6132bdd9e1606e31cdbe45e650fb740df8b5171aa90b48afd6e0fa0356ef`.
- `B_inf=3` result NPZ SHA-256: `6cc96dda909a9d928b5c4df2d6568cf234dabf2540616378ee1d4d0b7213af3a`.
- `B_inf=3` metadata SHA-256: `f28e9caaeb0fabd2fa4f0b0e753c286de4bae02bb8306a579d1663173f828c64`.
- `B_inf=3` independent audit SHA-256: `c3436903b315d53d5f203db27dba3b0efa348eec668e90ca1afa1c7133815ddb`.
- Completion marker SHA-256: `e320d574171a30d0aea1376e66a47e6d4336e997b6f6e6b964ac69f98b946dfe`.

These are local measurements against the repository's custom victim. Paper comparability still requires matching all protocol dimensions; the speedup and `PASS` audit do not establish paper replication.
