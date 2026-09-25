# Phase 15 QUANTUM-TEMP Results

## Interpreter

`C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`

No new environment was created.

## Defense Definition

QUANTUM-TEMP was implemented as staged training-time regularization without changing
the 4-qubit, 4-layer, 47-parameter QSNN architecture.

Stage A, QT-Q:

```text
L = CrossEntropy(clean)
  + lambda_q * (1 - measured-feature fidelity)
```

Stage B, QT-QP:

```text
L = CrossEntropy(clean)
  + lambda_q * (1 - measured-feature fidelity)
  + lambda_pred * symmetric_KL(clean_logits, perturbed_logits)
```

Training perturbations:

```text
delta ~ Uniform(-0.02T, +0.02T)
t_perturbed = clamp(t_clean + delta, 0, T)
```

The fidelity term measures product fidelity between per-qubit Z-measurement
distributions. Full input-state fidelity was not used as a model regularizer because a
shared unitary preserves it and therefore gives no model-parameter gradient.

## Training Configurations

| Variant | Training epsilon | lambda_q | lambda_pred | Consistency |
|---|---:|---:|---:|---|
| Baseline | none | 0 | 0 | Disabled |
| QT-Q | 2% of T | 0.1 | 0 | Disabled |
| QT-QP | 2% of T | 0.1 | 0.1 | Symmetric KL |

All variants used seed 42, the unchanged Iris split, 80 epochs, Adam with learning
rate 0.01, and the same QSNN architecture.

## Verification

- Phase 15 gate: 4 tests passed.
- Full regression: 29 tests passed in 5.10 seconds.
- Zero epsilon produced unchanged timings and zero drift.
- Nonzero perturbations respected the timing budget and `[0,T]` bounds.
- Quantum and prediction consistency losses produced finite, nonzero gradients.
- At least one model parameter changed after a defended training step.
- Phase 14 attacks remained unchanged.

Defended checkpoints differed from baseline:

- QT-Q maximum parameter difference: 0.0131
- QT-QP maximum parameter difference: 0.0169

## Clean Accuracy

| Model | Accuracy | Macro F1 | Best validation accuracy |
|---|---:|---:|---:|
| Baseline | 0.8667 | 0.8611 | 0.9667 |
| QT-Q | 0.8667 | 0.8611 | 0.9667 |
| QT-QP | 0.8667 | 0.8611 | 0.9667 |

Clean-accuracy cost: 0 percentage points.

## Random Timing Attack

Results were identical for Baseline, QT-Q, and QT-QP.

| Epsilon | ASR | Mean trace distance | Mean 1-Fidelity | Mean Delta_cls |
|---:|---:|---:|---:|---:|
| 1% | 0.0000 | 0.0085 | 0.000077 | 0.0000 |
| 2% | 0.0000 | 0.0170 | 0.000307 | 0.0074 |
| 5% | 0.0000 | 0.0422 | 0.001896 | 0.0481 |
| 10% | 0.0385 | 0.0822 | 0.007256 | 0.0852 |

## Classical Timing PGD

Results were identical for Baseline, QT-Q, and QT-QP.

| Epsilon | ASR | Mean trace distance | Mean 1-Fidelity | Mean Delta_cls |
|---:|---:|---:|---:|---:|
| 1% | 0.0769 | 0.0157 | 0.000247 | 0.0148 |
| 2% | 0.0769 | 0.0314 | 0.000987 | 0.0259 |
| 5% | 0.1923 | 0.0784 | 0.006151 | 0.0519 |
| 10% | 0.3462 | 0.1562 | 0.024397 | 0.0815 |

## TEMP-DRIFT Reference

Results were identical for Baseline, QT-Q, and QT-QP.

| Epsilon | ASR | Mean trace distance | Mean 1-Fidelity | Mean Delta_cls |
|---:|---:|---:|---:|---:|
| 1% | 0.0000 | 0.0144 | 0.000208 | 0.0000 |
| 2% | 0.0000 | 0.0288 | 0.000829 | 0.0000 |
| 5% | 0.0385 | 0.0709 | 0.005030 | 0.0000 |
| 10% | 0.1154 | 0.1381 | 0.019125 | 0.0000 |

The reference implementation remains a randomized search and is not the final
gradient TEMP-DRIFT attack.

## Gradient TEMP-DRIFT ASR

ASR was identical across Baseline, QT-Q, and QT-QP.

| Epsilon | Tau 1% | Tau 5% | Tau 10% |
|---:|---:|---:|---:|
| 1% | 0.0769 | 0.0769 | 0.0769 |
| 2% | 0.0385 | 0.0769 | 0.0769 |
| 5% | 0.0385 | 0.0769 | 0.0769 |
| 10% | 0.1154 | 0.1538 | 0.1538 |

All returned gradient TEMP-DRIFT samples passed the exact `Delta_cls <= tau`
constraint.

## Quantum Drift Assessment

QUANTUM-TEMP did not reduce reported input-state trace distance or `1-Fidelity` at
the requested regularization strengths.

Regularization magnitudes during training were small:

- Mean QT-Q quantum loss: approximately 0.0000640
- Mean QT-QP consistency loss: approximately 0.0000431

These losses changed checkpoint parameters but did not change test predictions or
attack robustness.

## Scientific Assessment

This is a negative result for the requested Phase 15 configuration.

- QUANTUM-TEMP did not reduce input-state quantum drift.
- QUANTUM-TEMP did not reduce attack success rate.
- No evaluated attack benefited from the defense.
- Clean accuracy was preserved with zero percentage-point cost.
- The result does not currently support a QUANTUM-TEMP robustness claim.

## Limitations

- Input-state drift is determined by timing perturbations and cannot be changed by
  downstream model parameters.
- Measured-feature fidelity is a model-dependent surrogate, not full-state fidelity.
- Iris has only 30 test samples and 26 clean-correct ASR samples.
- The exact eight-bin ISI metric is coarse for four Iris timings and is not proof of
  classical or perceptual stealth.
- Regularization strengths were not tuned using test data.

Total Phase 15 experiment runtime: 316.00 seconds.

## Recommended Next Step

Run a validation-only regularization-strength ablation for `lambda_q` and
`lambda_pred`. Select the configuration using clean and perturbed validation
performance, then evaluate the held-out test set once.

## Machine-Readable Artifacts

- `results/iris_quantum_temp_phase15.csv`
- `results/iris_quantum_temp_phase15.json`
- `results/iris_quantum_temp_baseline_training.json`
- `results/iris_quantum_temp_qt_q_training.json`
- `results/iris_quantum_temp_qt_qp_training.json`
