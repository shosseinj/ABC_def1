# Phase 1 GPU vectorization validation

## Outcome

**PASS.** The fixed configuration `ATTACK_BATCH_SIZE=64`,
`AUDIT_BATCH_SIZE=64` passed exact-equivalence, regression, independent-audit,
profiler, and measured-speedup gates. No other attack batch size was benchmarked
in this validation.

## Preserved scientific state

- Completed seed-42 binary `B_inf=1` remains unchanged and independently audited.
- Seed-42 binary `B_inf=2` is frozen at the valid scalar prefix **175/1000**.
- Dataset, manifest, seed, beta, attack iterations/objective, `T=10`, packet
  identity/amplitude, capacity-1, budget definitions, ASR, and auditor rules were
  not changed.

## Hot-path findings and implementation

The former strict projector transferred tensors to CPU, built Python tuples,
sorted Python lists, and performed set-based reservation/collision checks for
every sample and iteration. These operations dominated runtime and repeatedly
synchronized CUDA with the host.

The replacement performs packet extraction, candidate generation, stable
lexicographic ordering, movement preparation, and result tensors with batched
PyTorch CUDA operations. An exact Numba CUDA kernel executes the greedy
reservation/collision/budget state machine independently for each sample.
Soft retiming uses CUDA `scatter_add_`; objectives and success/prediction tensors
remain on CUDA.

A true batch-64 victim convolution was diagnosed and rejected: although fast,
it changed near-zero attack-gradient signs and therefore packet destinations.
To satisfy exact equality, victim forwards remain independent batch-1 CUDA calls
inside the 64-sample attack container. Projection and candidate evaluation are
still vectorized across all 64 samples. This preserves the validated scalar
model arithmetic while removing the CPU/Python projection bottleneck.

## Exact-equivalence and audit evidence

The test used 64 records per budget family (five frozen scalar-reference samples
repeated deterministically) for binary N-MNIST seed 42:

- `B_inf=1`: 64/64 exact
- `B1=500`: 64/64 exact
- `B0=200`: 64/64 exact

Across **192/192** records, equality held for predictions, success flags,
realized `B_inf`, `B1`, and `B0`, projected tensors, packet amplitudes, packet
destinations, and audit outcomes. The independent reconstruction auditor passed
192/192 records. The contract regression suite passed **18/18** tests.

## Performance

| Budget | Seconds/sample | Samples/second | Scalar seconds/sample | Speedup | Mean GPU | Peak GPU | Peak VRAM | Mean CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_inf=1` | 0.2227 | 4.4898 | 4.4131 | 19.81x | 25.33% | 61% | 3263 MiB | 102.23% |
| `B1=500` | 0.4460 | 2.2422 | 22.8180 | 51.16x | 28.89% | 68% | 4580 MiB | 105.99% |
| `B0=200` | 0.4453 | 2.2456 | 19.2624 | 43.26x | 28.25% | 71% | 4588 MiB | 109.54% |

CPU utilization is process utilization and can exceed 100% on a multicore host.

## Profiler proof

The batch-64 CUDA profile recorded nonzero device time for the required
vectorized operators:

- `aten::sort`: 2164 microseconds
- `aten::gather`: 619 microseconds
- `aten::argsort`: 85 microseconds

Trace: `Reports/logs/phase1_gpu_vectorization_trace.json`

## Provenance

- Structured result: `Reports/results/phase1_gpu_vectorization_equivalence.json`
- Independent audit: `Reports/results/phase1_gpu_vectorization_audit.json`
- Serialized records: `Reports/results/phase1_gpu_vectorization_records.npz`
- Recovery checkpoint: `Reports/checkpoints/phase1_gpu_vectorization_staging.json`
- Validation command: `python -u scripts/validate_phase1_gpu_vectorization.py`

Phase 1 may therefore resume from sample 176 with attack batch 64 and audit batch
64 without rerunning completed `B_inf=1` or the valid 175-sample `B_inf=2` prefix.
