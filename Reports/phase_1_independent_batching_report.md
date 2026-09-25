# Phase 1 independent-sample GPU batching report

## Outcome

**Scientific equivalence PASS; throughput acceptance FAIL for batch size 8. Selected batch size: 1.**

The completed seed-42 binary B∞=1 result remains untouched. The incomplete B∞=2 condition has a validated atomic 75-sample prefix and can resume at sample 76.

## Root cause of the earlier batch failure

The first implementation placed multiple samples into one mathematical attack batch. That changed semantics in two ways:

1. cross-entropy and budget penalties used batch reductions, while the capacity term used a global source-packet denominator; this changed each sample's relative objective weighting;
2. batched convolution used different floating-point accumulation than batch-size-1 convolution, changing gradient-sign ties and therefore greedy packet destinations.

The following were **not** root causes:

- LIF state: membrane tensors include the batch dimension and were isolated;
- RNG: PIL-PGD contains no stochastic operation;
- early stopping/success masks: none are used;
- padding/variable lengths: inputs are fixed dense `[10,2,34,34]` grids;
- projection state: occupied/reserved sets and budgets are local per sample;
- shared hidden state: the model creates LIF membrane state locally on every forward.

## Correct independent implementation

`attacks/pil_pgd.py::run_independent_parallel` assigns every sample its own:

- `PILPGDAttack` object and logits;
- scalar objective and autograd graph;
- LIF forward with model batch size one;
- strict projector, reservations, occupancy, and budget counter;
- CUDA stream.

Thus GPU model work can overlap without mathematically batching samples. With deterministic algorithms and the frozen environment, worker size 8 exactly matched worker size 1 on 64 frozen B∞ samples.

## Batch-size benchmark

| Workers | Status | Seconds/sample | Samples/second | Mean GPU util. | Peak GPU util. | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PASS, selected | **2.930** | **0.3413** | 28.78% | 48% | 4,606 MiB |
| 8 | PASS equivalence, throughput rejected | 5.889 | 0.1698 | 22.15% | 89% | 6,814 MiB |
| 16 | Not run: stopped after throughput regression | — | — | — | — | — |
| 32 | Not run: stopped after throughput regression | — | — | — | — | — |
| 64 | Not run: stopped after throughput regression | — | — | — | — | — |

Worker size 8 was 0.498× as fast as size 1 (a 2.01× slowdown). The strict projector is CPU/Python dominated; its score ordering and reservation loop hold the GIL. Concurrent attacks therefore contend on CPU projection, launch many small kernels across streams, consume more VRAM, and reduce average GPU utilization. The model is too small for stream concurrency to offset this cost.

Per the predeclared stopping rule, larger sizes were not attempted after throughput stopped improving. The fastest exact-equivalent setting is batch size **1**.

## Equivalence checks

For worker size 8 versus size 1 on 64 frozen samples:

- projected temporal tensors: exact;
- packet displacements/destinations: exact;
- attacked predictions: exact;
- exact model-evaluated tensor: PASS.

The selected size 1 also passed representative B∞, B1, and B0 checks for attacked predictions, success, all realized budgets, projected representation, amplitudes, destinations, and audit outcome.

Evidence: `Reports/results/phase1_independent_batch_equivalence.json` and `Reports/checkpoints/phase1_independent_batch_progress.json`.
