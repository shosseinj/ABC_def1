# N-MNIST temporal representation evidence audit

## Conclusion

The reference paper is internally inconsistent in terminology, but its operational definition is unambiguous: a perturbable object is one **nonzero temporal-grid cell**, with an unchanged scalar value. On an integer grid that value is an integer count/amplitude. Capacity-1 means at most one such nonzero cell-packet may occupy an event-line/time-bin. It does **not** mean every raw event unit inside an integer count is independently scheduled by the released algorithm.

The updated authoritative repository contract now adopts this operational interpretation and explicitly forbids unit-packet expansion. The prior policy conflict is resolved.

## Evidence hierarchy and trace

Reference: *Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks*, arXiv:2602.03284v1 (3 February 2026). Pinned upstream commit: `19f42e63d31cbdc4ef2d5ef7b6e40716f92531c3`.

1. **Equation 5 / Definition 1:** the active set is `A(x)={(s,j):x[s,j]>0}`. There is one shift `delta[s,j]` for each nonzero cell, not one shift per unit of its value.
2. **Equation 9:** the entire value `x[s,j]` is multiplied by the source cell's shift probability and replayed at its target.
3. **Equations 11–12:** B1 and B0 surrogates sum over `A(x)`, so integer magnitude does not multiply budget cost.
4. **Algorithm 1:** strict projection is evaluated in-loop; the returned adversarial tensor is the final strict projection.
5. **Appendix D.1:** explicitly states that a nonzero entry is “one event packet (a single spike on binary grids or an integer-valued packet on integer grids).” Algorithm 2 sets `has_src[s,j]=1{x[s,j]>0}` and assigns `adv[t,j]=x[s,j]`.
6. **Appendix D.2, Eqs. 31–32:** projection preserves the per-line multiset of cell values and total sum; it describes a per-line permutation of nonzero packets.
7. **Appendix I, Eqs. 39–41:** B∞, B1, and B0 operate over moved nonzero `(s,j)` entries. B0 counts moved nonzero cells once regardless of integer amplitude.
8. **Pinned code:** `utils/attack.py` uses `has_src=(x_act>0)`, `src_mask=(x_enc_tbchw!=0)`, copies `x_act[s,j]` to the target, and decrements B1 by temporal distance or B0 by one. `utils/encoder.py::DVSEncoder` passes integer grids through without expansion.
9. **T=10:** the paper's main experiments and pinned `test.py` use discretized post-encoder temporal indices; the attack does not perturb native sensor timestamps.

The conflicting sentence in Section 4.1 says integer counts are “conceptually” decomposed into unit packets. That sentence is not implemented and is incompatible with the active-set equations, Appendix D/I, Algorithm 2, and released code whenever a count exceeds one.

## Answers to the protocol questions

| Question | Evidence-supported answer |
|---|---|
| Perturbable object | One active nonzero temporal cell `(s,j)`; binary value 1 or integer-valued packet. |
| Capacity-1 required | Yes, at the active cell-packet level. |
| Multiple raw events in a cell | Yes; represented by integer amplitude/count. |
| Individual raw-event identity preserved | No identity exists after binning; the nonzero cell packet and its value are preserved. |
| Collisions after retiming | Forbidden between active cell-packets on one event line; targets are reserved/occupied. Values are not merged or accumulated by strict projection. |
| Temporal order | No global raw-event ordering exists. Injective placement is enforced independently per fixed pixel/polarity event line. Symmetric swaps are blocked by reservation. |
| B∞ | Maximum absolute bin displacement among moved nonzero cell-packets. |
| B1 | Sum of absolute bin displacements over moved nonzero cell-packets; not count-amplitude weighted. |
| B0 | Number of moved nonzero cell-packets; not raw event units. |
| Timestamp unit | Discretized temporal-grid index, not native event timestamp. |

## Frozen-manifest feasibility after contract amendment

Machine-readable evidence is in `Reports/results/nmnist_representation_feasibility.json`.

| Seed | Samples | Cell-packet feasible | Source collision-free | Samples with amplitude > 1 | Max amplitude |
|---:|---:|---:|---:|---:|---:|
| 42 | 1000 | 1000 (100%) | 1000 | 1000 | 11 |
| 123 | 1000 | 1000 (100%) | 1000 | 1000 | 12 |
| 777 | 1000 | 1000 (100%) | 1000 | 1000 | 13 |

Integer-valued cell-packets support all frozen budget families and exact reconstruction for every sample because unchanged source-cell amplitude plus fixed event line and target time uniquely reconstruct the projected tensor. Budgets are upper bounds, so an unchanged packet placement is always feasible; the attack may use any valid displacement up to the requested beta.

## Pre-attack dry run

The deterministic seed-42 dry run passed independent reconstruction audits for integer B∞=1, B1=500, B0=200, and binary B1=500. The auditor independently reloaded the dataset/checkpoint, recovered packet identities and amplitudes, reconstructed strict adversarial tensors, checked collision-free occupancy and all three realized budgets, and reproduced clean/attacked predictions. These artifacts are explicitly excluded from benchmark results; see `Reports/results/nmnist_dry_run/dry_run_PASS.json`.

## Provenance classification

- `T=10`: reference paper/main implementation.
- Capacity-1 and no collision: reference paper and implementation, at nonzero-cell packet level.
- Integer count expansion into raw unit packets: prohibited by the authoritative local contract and not used by the executable benchmark.
- Current converter: `ResearchLoop/core/contract.py::cell_packets_from_integer_grid`; the historical unit-expansion API has been removed.
- Current grid attack's amplitude-cell behavior: `attacks/pil_pgd.py::strict_project_grid`, matching Appendix D and pinned code.
