# Frozen TEMP-DRIFT benchmark specification

**Frozen:** 2026-09-21T04:55:29.619705+00:00  
**Seeds:** exactly 42, 123, 777  
**Reference:** *Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks*, arXiv:2602.03284v1.

## Threat model and time

Untargeted white-box PIL-PGD retimes existing packets only. Time is the discrete post-preprocessing bin index with `T=10` and domain `[0,9]`. **One nonzero temporal-grid cell is one indivisible amplitude-bearing cell-packet.** Its complete integer count moves unchanged on its fixed x/y/polarity event line; counts are never expanded into unit packets. Strict projection forbids collisions, merging, and accumulation and enforces capacity-1 between cell-packets, origin reservation, valid time, and the active global budget. Forward evaluation uses the strict tensor; gradients use the soft-retiming straight-through surrogate. The final serialized displacement reconstructs exactly the returned model-evaluated tensor.

## Budgets

- `B_inf = max_i |t_adv[i]-t_clean[i]| <= beta`
- `B1 = sum_i |t_adv[i]-t_clean[i]| <= beta`, unweighted by packet amplitude
- `B0 = count_i[t_adv[i] != t_clean[i]] <= beta`, once per moved cell-packet

Every sample records all three realized values. N-MNIST grids are `B_inf={1,2,3}`, `B1={500,750,1000,1500}`, `B0={200,300,400,600}`. DVS-Gesture and CIFAR10-DVS grids are `B_inf={1,2,3}`, `B1={2000,4000,8000,16000}`, `B0={1000,2000,4000,8000}`, for both binary and integer representations.

## ASR and manifests

`ASR = 100 * successful attacks / clean-correct samples`. Eligibility is the clean-correct intersection of binary and integer inputs. Each dataset/model/seed manifest is frozen before attacks and reused unchanged across every representation, family, and beta. It stores sample IDs, labels, both clean predictions, denominator, and hash.

## Checkpoint matrix

| Dataset | Seed | Checkpoint SHA-256 | Integer clean acc. | Binary clean acc. | Frozen denominator | Manifest SHA-256 |
|---|---:|---|---:|---:|---:|---|
| N-MNIST | 42 | `c1f941a16e0004d11d9aa73e2689d92357a32be007986d336a91a7d93b7b6f59` | 98.49% | 85.48% | 1000 | `1839493222ca9f05637f83d29766b90112056f0b01988cc165a4898767700039` |
| N-MNIST | 123 | `7382ca960d3c21e57105d29576910ef9e9362aed847e0fb40d145fd07c82b384` | 98.14% | 95.65% | 1000 | `d169a7a2c37469baac8ed029cf9dceeae2c0444eef472df514bfb48eae80030f` |
| N-MNIST | 777 | `a0fae10b20db29b99da8675cf8be1a98f4c3a0bf47422d6c22b170e4113a2030` | 98.53% | 80.15% | 1000 | `6ace1cb472d34c879913deecbad0750560f59939c9954de9608769191290dea5` |
| DVS-Gesture | 42 | `b3faa98397d7abf138dfe255e8512ab87b241208b86fdaeba770f5546db889ab` | 84.09% | 12.50% | 30 | `6140be41cbc85f9244dedd5134371840b05672fe408b24ae7ed282119fd4df11` |
| DVS-Gesture | 123 | `f8c72f439f4db7d4edd1d3ed1ecd35001fe6a6da94c77c01f5fea7e363cd0323` | 78.03% | 19.32% | 38 | `fa4f30966209980518c7ac59fa22f6fdd09bea1fb653d7605a74fc668d4046b5` |
| DVS-Gesture | 777 | `255f1d62aec0e3605bdb1a853d24a34e19c044cae5490380a5908f38024dedcd` | 87.12% | 9.47% | 24 | `d9e6a4fd18bcc9132411c0ea1398ea225ad4cab40e7a76260eec6a2ffd60ab0a` |
| CIFAR10-DVS | 42 | `7f0c6a4af012082a79c15296191520146ad612f4d60a9478f593a500d9a5d7d7` | 39.10% | 12.20% | 70 | `5574cff5b2d9dac5ae33cd30a297fa4d415fdc828cd9773c1b7a36d80a6cfcc8` |
| CIFAR10-DVS | 123 | `1712510c621f021b312a70fdc8457f0237722fbb3bdf96e3a62f010fc4f4ce12` | 49.20% | 18.60% | 100 | `ff71e67a47b405289f02e57e647c59264956976fd3356c014ea42c84a1139143` |
| CIFAR10-DVS | 777 | `61702677146897c241bcc6504231b5f51a083aee82f9566ba6ecd770d88dbe06` | 50.30% | 17.30% | 100 | `c581f23ce795e87bed4316ebedf9143f936afa02093a152e8e9b8c2ed6df7714` |

## Dataset policies

### N-MNIST
Tonic structured events (x,y,t,p), sensor 34x34x2. Per-sample native timestamp range mapped by integer floor to 10 equal-duration bins; polarity-separated 34x34 additive uint8 count grid; binary grid is presence(count>0); integer grid retains count. Official train/test; frozen seed-42 stratified train/validation split; attacked set is 1000 deterministic samples clean-correct under both representations per model seed.

### DVS-Gesture
Tonic structured events (x,y,t,p), sensor 128x128x2. Official sample timestamps mapped to 10 equal-duration bins; coordinates downsampled 2x to 64x64; polarity-separated additive counts with per-sample max normalization for repository model; binary grid is presence(count>0). Official train/test; deterministic seed-42 stratified 80/20 train/validation split; attacked set is all official-test samples clean-correct under both representations per model seed.

### CIFAR10-DVS
Tonic structured events (x,y,t,p), sensor 128x128x2. Per-sample timestamps mapped to 10 equal-duration bins; polarity-separated 128x128 additive counts with per-sample max normalization for repository model; binary grid is presence(count>0). Tonic dataset has no official split; frozen seed-42 class-stratified 80/10/10 train/validation/test indices; attacked set is 100 deterministic test samples clean-correct under both representations per model seed.

## Audit, provenance, and statistics

`ResearchLoop/core/audit.py` independently verifies cell-packet identity/count, complete amplitudes, total mass, fixed line, integer timeline, collision-free capacity-1 occupancy, and all realized budgets. Across three seeds, reports use mean, sample SD (`ddof=1`), and two-sided 95% Student-t CI (`df=2`). Reference comparison values come only from `ResearchLoop/reference/reference_results.csv`. Rows are `PAPER_COMPARABLE` only on an exact protocol match; repository-specific models/preprocessing are otherwise `NON_COMPARABLE`.

## N-MNIST temporal representation resolution

**Evidence finding, 2026-09-22:** The earlier blocker was caused by interpreting each integer count as independently scheduled raw-event units. The paper's operational definition instead uses one active object per nonzero temporal cell. Definition 1/Eq. 5 defines `A(x)={(s,j):x[s,j]>0}`; Eq. 9 moves the full `x[s,j]`; Appendix D calls this “an integer-valued packet”; Algorithm 2 and pinned commit `19f42e63d31cbdc4ef2d5ef7b6e40716f92531c3` copy the full value and charge B1/B0 once per moved nonzero cell. Appendix I defines all budgets over these active cells. The attack unit is a discretized bin index, not a native sensor timestamp.

Under those reference semantics, capacity-1 remains required between nonzero cell-packets on each fixed pixel/polarity line. Collisions are forbidden rather than merged; integer amplitude/count is unchanged; the per-line multiset of cell values and total count mass are conserved; B∞ is maximum cell-packet displacement, B1 is its unweighted displacement sum, and B0 is the number of moved nonzero cell-packets. Serialization stores each source cell, unchanged value, fixed line, and displacement, permitting exact reconstruction and independent audit.

The authoritative `AGENTS.md` amendment now explicitly selects the operational cell-packet semantics and prohibits unit expansion. A fresh frozen-manifest validation found 1000/1000 feasible samples for each seed 42, 123, and 777; every sample contains amplitudes greater than one, directly exercising the corrected rule. The old Phase 1 blocker is therefore **RESOLVED**. Full evidence: `Reports/nmnist_temporal_semantics_audit.md` and `Reports/results/nmnist_representation_feasibility.json`.
