# Phase 15.3 Multi-Seed Replication

## Frozen Protocol

- Model seeds: `42, 123, 777, 2026, 6543`
- Fixed split seed: `42`
- Defense: `lambda_q=5`, `lambda_pred=0`, consistency disabled, epsilon `0.02T`
- Architecture, optimizer, epochs, preprocessing, selection rule, and Phase 14 attacks unchanged
- Baseline and defense share the same model seed in every pair

## Clean Performance

| Seed | Baseline accuracy | Defense accuracy | Delta | Baseline F1 | Defense F1 | N common |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8667 | 0.8333 | -0.0333 | 0.8611 | 0.8222 | 25 |
| 123 | 0.8667 | 0.8667 | 0.0000 | 0.8611 | 0.8611 | 26 |
| 777 | 0.9667 | 0.9667 | 0.0000 | 0.9666 | 0.9666 | 29 |
| 2026 | 0.7667 | 0.8000 | +0.0333 | 0.7613 | 0.7917 | 23 |
| 6543 | 0.7333 | 0.7333 | 0.0000 | 0.7070 | 0.7070 | 22 |

Baseline accuracy is `0.8400 ± 0.0925`; defense accuracy is `0.8400 ± 0.0863`.
Mean clean delta is `0.0000 ± 0.0236`, range `[-0.0333,+0.0333]`. One seed
improves, three are unchanged, and one degrades. The seed-42 clean loss is not a
reproducible systematic effect.

## Classical PGD 10%

| Seed | N | Baseline fail | Defense fail | Rescued | Broken | Net |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 25 | 8 | 7 | 1 | 0 | +1 |
| 123 | 26 | 7 | 7 | 0 | 0 | 0 |
| 777 | 29 | 11 | 10 | 1 | 0 | +1 |
| 2026 | 23 | 4 | 4 | 0 | 0 | 0 |
| 6543 | 22 | 3 | 3 | 0 | 0 | 0 |

Positive/zero/negative seeds: `2/3/0`. Total rescued/broken is `2/0` over 125
common observations. Mean net gain is 0.4 and median is 0. Baseline paired ASR is
`0.2558 ± 0.1007`; defense paired ASR is `0.2409 ± 0.0845`.

## Reference TEMP-DRIFT 10%

Only seed 42 improves. Positive/zero/negative seeds are `1/4/0`, total
rescued/broken is `1/0`, mean net gain is 0.2, and median net gain is 0.

## Gradient TEMP-DRIFT 10%

| Tau | Positive/zero/negative | Rescued/broken | Baseline ASR mean | Defense ASR mean |
|---:|---:|---:|---:|---:|
| 1% | 1/4/0 | 1/0 | 0.0927 | 0.0847 |
| 5% | 1/4/0 | 1/0 | 0.0611 | 0.0531 |
| 10% | 1/4/0 | 1/0 | 0.0699 | 0.0619 |

All apparent gradient improvements are confined to seed 42 and do not replicate in
the other four seeds.

## Sample Identity

PGD 10% rescues sample 57 in one seed and sample 147 in one seed. No sample is
rescued more than once for any attack configuration. There is no consistently helped
sample across seeds.

## Model-Dependent Drift

Across the two PGD 10% rescues:

- Feature drift: `0.28153 -> 0.28177`
- Logit drift: `0.63824 -> 0.63568`
- Prediction JS: `0.01111 -> 0.01118`

Only logit drift decreases slightly. The single seed-42 gradient rescue reduces all
three metrics, but this does not replicate in four other seeds.

## Statistical Interpretation

PGD 10% has two positive and three zero seed directions. A two-sided exact sign test
over the two nonzero seeds gives `p=0.5`. Reference and gradient priority settings
have only one positive and four zero seeds. The fixed dataset means sample observations
must not be treated as 125 independent experimental replications.

## Conclusion

The weak Phase 15.2 signal is seed-sensitive. Mean clean accuracy is unchanged, but
robustness gains are small, inconsistent, tied to different samples, and unsupported
by consistent model-dependent drift reduction. The evidence does not support a
QUANTUM-TEMP robustness claim.

## Artifacts

- `results/iris_phase153_clean_per_seed.{csv,json}`
- `results/iris_phase153_paired_per_seed.{csv,json}`
- `results/iris_phase153_multiseed_attack_summary.{csv,json}`
- `results/iris_phase153_sample_reproducibility.{csv,json}`
- `results/iris_phase153_drift_summary.csv`
- Per-seed training summaries and checkpoints
