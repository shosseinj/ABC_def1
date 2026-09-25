# Phase 14 Attack Results

## Experiment Setup

- Dataset: Iris test split
- Model: frozen 4-qubit, 4-layer QSNN
- Trainable model parameters: 47 during training, frozen during attacks
- Seed: 42
- Clean test accuracy: 0.8667
- Untargeted ASR denominator: 26 clean-correct test samples
- Timing budgets: 1%, 2%, 5%, and 10% of `T=100`

Attack definitions:

- Random Jitter: uniformly random timing perturbations without an optimization objective.
- Classical Timing: 20-step projected sign-gradient ascent maximizing true-label cross-entropy, with zero initialization and step size `epsilon / 5`.
- TEMP-DRIFT Reference: randomized product-state quantum-drift search. This is not the final TEMP-DRIFT method.
- TEMP-DRIFT Gradient: 40-step, 3-restart projected gradient optimization maximizing product-state `1 - Fidelity`, with step size `epsilon / 10`.

## Phase 14 Baseline Comparison

| Epsilon | Attack | Attacked accuracy | ASR | Successful attacks | Mean trace distance | Mean 1-Fidelity | Mean Delta_cls | Mean ISI TV |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | Random Jitter | 0.8667 | 0.0000 | 0/26 | 0.0085 | 0.000077 | 0.0000 | 0.0000 |
| 1% | Classical Timing | 0.8000 | 0.0769 | 2/26 | 0.0157 | 0.000247 | 0.0148 | 0.0444 |
| 1% | TEMP-DRIFT Reference | 0.8667 | 0.0000 | 0/26 | 0.0144 | 0.000208 | 0.0000 | 0.0000 |
| 2% | Random Jitter | 0.8667 | 0.0000 | 0/26 | 0.0170 | 0.000307 | 0.0074 | 0.0222 |
| 2% | Classical Timing | 0.8000 | 0.0769 | 2/26 | 0.0314 | 0.000987 | 0.0259 | 0.0778 |
| 2% | TEMP-DRIFT Reference | 0.8667 | 0.0000 | 0/26 | 0.0288 | 0.000829 | 0.0000 | 0.0000 |
| 5% | Random Jitter | 0.8667 | 0.0000 | 0/26 | 0.0422 | 0.001896 | 0.0481 | 0.1444 |
| 5% | Classical Timing | 0.7000 | 0.1923 | 5/26 | 0.0784 | 0.006151 | 0.0519 | 0.1556 |
| 5% | TEMP-DRIFT Reference | 0.8333 | 0.0385 | 1/26 | 0.0709 | 0.005030 | 0.0000 | 0.0000 |
| 10% | Random Jitter | 0.8333 | 0.0385 | 1/26 | 0.0822 | 0.007256 | 0.0852 | 0.2556 |
| 10% | Classical Timing | 0.5667 | 0.3462 | 9/26 | 0.1562 | 0.024397 | 0.0815 | 0.2444 |
| 10% | TEMP-DRIFT Reference | 0.7667 | 0.1154 | 3/26 | 0.1381 | 0.019125 | 0.0000 | 0.0000 |

## Gradient TEMP-DRIFT

| Epsilon | Tau | Attacked accuracy | ASR | Mean trace distance | Mean 1-Fidelity | Mean Delta_cls | Final feasible fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 1% | 0.8000 | 0.0769 | 0.0156 | 0.000243 | 0.0074 | 1.000 |
| 1% | 5% | 0.8000 | 0.0769 | 0.0157 | 0.000247 | 0.0074 | 1.000 |
| 1% | 10% | 0.8000 | 0.0769 | 0.0157 | 0.000247 | 0.0074 | 1.000 |
| 2% | 1% | 0.8333 | 0.0385 | 0.0295 | 0.000876 | 0.0074 | 1.000 |
| 2% | 5% | 0.8000 | 0.0769 | 0.0314 | 0.000987 | 0.0074 | 1.000 |
| 2% | 10% | 0.8000 | 0.0769 | 0.0314 | 0.000987 | 0.0074 | 1.000 |
| 5% | 1% | 0.8333 | 0.0385 | 0.0744 | 0.005677 | 0.0074 | 1.000 |
| 5% | 5% | 0.8000 | 0.0769 | 0.0661 | 0.005050 | 0.0074 | 1.000 |
| 5% | 10% | 0.8000 | 0.0769 | 0.0669 | 0.005168 | 0.0074 | 1.000 |
| 10% | 1% | 0.7667 | 0.1154 | 0.1387 | 0.020602 | 0.0074 | 1.000 |
| 10% | 5% | 0.7333 | 0.1538 | 0.1174 | 0.016910 | 0.0074 | 1.000 |
| 10% | 10% | 0.7333 | 0.1538 | 0.1229 | 0.017928 | 0.0074 | 1.000 |

## Runtime

- Phase 14 baseline comparison: 114.42 seconds
- Full gradient comparison: 197.64 seconds

## Interpretation

- Classical Timing is the strongest classification attack and reaches 34.62% ASR at the 10% timing budget.
- Gradient TEMP-DRIFT returns exact-feasible samples for every evaluated configuration and causes prediction flips, but does not consistently exceed the randomized reference in quantum drift at looser stealth thresholds.
- The current eight-bin ISI statistic is coarse for four Iris spike times, which limits sensitivity to `tau`.
- Historical randomized-reference results and gradient TEMP-DRIFT results remain separate and should not be interpreted as the same attack.

## Machine-Readable Results

- `results/iris_attack_comparison_phase14.csv`
- `results/iris_attack_comparison_phase14.json`
- `results/iris_attack_comparison_gradient.csv`
- `results/iris_attack_comparison_gradient.json`
