# Phase 1 batch-processing optimization report

## Scientific decision

**PASS with attack batching rejected.** The fastest scientifically equivalent attack batch size is **1**. Independent clean/audit inference uses exact-equivalent batch size **64**.

Ordinary multi-sample attack batching changed gradient trajectories and therefore packet destinations. Although it improved throughput and often retained the same attacked class, it changed the discrete benchmark output and is prohibited.

## Attack batch-size results

| Requested size | Effective size | Result | B∞ s/sample | B1 s/sample | B0 s/sample |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | **Selected; exact baseline** | 4.433 | 22.866 | 19.317 |
| 8 | 5 on frozen subset | **REJECTED: equivalence FAIL** | 2.186 | 10.554 | 10.345 |
| 16 | — | Not run after size-8 failure | — | — | — |
| 32 | — | Not run after size-8 failure | — | — | — |
| 64 | — | Not run after size-8 failure | — | — | — |

All 15 batch-size-8 comparisons failed exact packet-destination/projected-tensor equivalence. Packet amplitudes remained preserved, but realized B1/B0 values or destination identities changed. Consequently, the apparent ~2× raw attack speedup is scientifically invalid and is not used.

## Safe inference/auditor batching

| Batch size | Seconds/sample | Samples/second | Exact predictions | Peak VRAM |
|---:|---:|---:|---|---:|
| 1 | 0.004139 | 241.6 | PASS | 2,445 MiB |
| 8 | 0.000530 | 1,885.4 | PASS | 2,445 MiB |
| 16 | 0.000271 | 3,686.6 | PASS | 2,468 MiB |
| 32 | 0.000179 | 5,584.2 | PASS | 2,516 MiB |
| 64 | **0.000124** | **8,063.2** | **PASS** | 2,608 MiB |

Batch 64 gives a 33.4× inference throughput increase. Binary predictions matched exactly on 64 frozen samples and reproduced all 1,000 records, 834 successes, and PASS outcome of the completed independent B∞=1 audit. Integer predictions also matched exactly on 64 frozen samples.

## Hardware utilization

- GPU: NVIDIA RTX 4090, 24,564 MiB
- Legacy attack batch 1: mean GPU utilization 18.0%, peak 44%, peak VRAM 2,377 MiB
- Invalid attack batch 8: mean GPU utilization 14.1–25.2%, peak 51–88%, peak VRAM 2,490 MiB
- Selected audit batch 64: sampled GPU utilization 43%, peak VRAM 2,608 MiB

Attack batching did not efficiently saturate the GPU because strict per-sample projection remains CPU/Python work and exact independence prevents combining attack trajectories.

## Gates and runtime

- Contract/regression tests: **17/17 PASS**
- Prior exact projector equivalence: **15/15 PASS**
- Attack batch 8 equivalence: **FAIL; rejected**
- Selected attack batch 1: **PASS by exact baseline definition**
- Selected audit batch 64: **PASS**, binary and integer
- Auditor rules: unchanged

Previous exact-optimized balanced attack time is 15.538 seconds/sample. Since no multi-sample attack batch is equivalent, this remains the selected attack runtime; only audit inference gains the 33.4× speedup. Estimated remaining attack compute is approximately 302 hours before fixed overhead.

Machine-readable evidence: `Reports/results/phase1_batch_equivalence.json`, `Reports/results/phase1_integer_audit_batch_equivalence.json`, and `Reports/checkpoints/phase1_batch_progress.json`.
