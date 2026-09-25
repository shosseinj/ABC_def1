# Phase 15.1 QUANTUM-TEMP Strength Ablation

## Protocol

Hyperparameters were selected only on the unchanged Iris validation split. The
selected configuration was written before the first final test evaluation. Model
architecture, initialization protocol, optimizer, learning rate, epochs, data split,
and Phase 14 attacks were unchanged.

## Lambda-Q Validation Ablation

| lambda_q | CE | Raw Lq | Weighted Lq | Rq | Clean val acc | Perturbed val acc | Random ASR | PGD ASR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.864920 | 0.00006402 | 0.00000640 | 0.000008 | 0.9667 | 0.9667 | 0.0000 | 0.1121 |
| 0.5 | 0.865008 | 0.00006398 | 0.00003199 | 0.000041 | 0.9667 | 0.9667 | 0.0000 | 0.1121 |
| 1 | 0.865117 | 0.00006394 | 0.00006394 | 0.000082 | 0.9667 | 0.9667 | 0.0000 | 0.1121 |
| 2 | 0.865330 | 0.00006386 | 0.00012772 | 0.000164 | 0.9667 | 0.9667 | 0.0000 | 0.1121 |
| 5 | 0.865931 | 0.00006364 | 0.00031818 | 0.000407 | 0.9667 | 0.9667 | 0.0086 | 0.0948 |
| 10 | 0.866807 | 0.00006332 | 0.00063316 | 0.000808 | 0.9667 | 0.9667 | 0.0086 | 0.1034 |
| 20 | 0.868109 | 0.00006289 | 0.00125789 | 0.001600 | 0.9333 | 0.9333 | 0.0357 | 0.1071 |

`lambda_q=5` was selected because it preserved clean validation accuracy and had the
lowest mean validation PGD ASR among eligible configurations.

## Lambda-Pred Validation Ablation

With `lambda_q=5`, every `lambda_pred` in `{0,0.1,0.5,1,2,5,10}` produced clean and
perturbed validation accuracy 0.9667, mean random ASR 0.0086, and mean PGD ASR
0.0948. Prediction consistency provided no validation benefit, so the tie-break rule
selected `lambda_pred=0`.

## Normalized Check

Even `lambda_q=20` contributed only 0.16% of CE. One separate validation-only variant
used `Lq / stopgrad(EMA_0.9(Lq) + floor)` with `lambda_q=0.1`. It raised `Rq` to
8.6%, but validation accuracy fell from 0.9667 to 0.9000 and mean PGD ASR worsened
from 0.1121 to 0.1944. It was rejected and never evaluated on test.

## Frozen Configuration

```text
lambda_q = 5
lambda_pred = 0
training epsilon = 0.02T
consistency disabled
selection data = validation only
```

## Final Clean Test

| Model | Accuracy | Macro F1 | Clean-correct count |
|---|---:|---:|---:|
| Baseline | 0.8667 | 0.8611 | 26 |
| Selected QUANTUM-TEMP | 0.8333 | 0.8222 | 25 |

Clean accuracy decreased by 3.33 percentage points, exceeding the 2-point criterion.

## Final Attack Success Counts

| Attack | Epsilon | Baseline | Selected |
|---|---:|---:|---:|
| Random | 1% | 0/26 | 0/25 |
| Random | 2% | 0/26 | 0/25 |
| Random | 5% | 0/26 | 0/25 |
| Random | 10% | 1/26 | 1/25 |
| Classical PGD | 1% | 2/26 | 2/25 |
| Classical PGD | 2% | 2/26 | 2/25 |
| Classical PGD | 5% | 5/26 | 4/25 |
| Classical PGD | 10% | 9/26 | 7/25 |
| Reference TEMP-DRIFT | 1% | 0/26 | 0/25 |
| Reference TEMP-DRIFT | 2% | 0/26 | 0/25 |
| Reference TEMP-DRIFT | 5% | 1/26 | 1/25 |
| Reference TEMP-DRIFT | 10% | 3/26 | 2/25 |

Gradient TEMP-DRIFT counts for tau `{1%,5%,10%}`:

| Epsilon | Baseline | Selected |
|---:|---|---|
| 1% | 2/26, 2/26, 2/26 | 1/25, 1/25, 1/25 |
| 2% | 1/26, 2/26, 2/26 | 1/25, 2/25, 2/25 |
| 5% | 1/26, 2/26, 2/26 | 1/25, 2/25, 2/25 |
| 10% | 3/26, 4/26, 4/26 | 1/25, 2/25, 3/25 |

## Gradient Norms

At the selected checkpoint, raw validation gradient norms were:

| Loss | Circuit | Classifier |
|---|---:|---:|
| CE | 0.07113 | 0.10905 |
| Quantum | 0.000105 | 0 |
| Consistency | 0.000181 | 0.000065 |

Auxiliary gradients remain orders of magnitude below CE. The quantum term does not
directly affect classifier-head parameters.

## Assessment

The original Phase 15 regularizers were too weak in effective loss scale. Stronger
quantum regularization modestly improved validation PGD ASR at `lambda_q=5`, but
prediction consistency added no benefit. Final attack counts improved in some cases,
including Classical PGD at 5% and 10%, but clean accuracy fell by 3.33 points and the
ASR denominators differ. Model-dependent feature, logit, and probability drift changes
were small and not consistently improved.

This is a mixed but ultimately negative result under the predefined criteria. It is
not sufficient for a QUANTUM-TEMP robustness claim.

## Artifacts

- `results/iris_quantum_temp_lambda_q_ablation.csv`
- `results/iris_quantum_temp_lambda_q_ablation.json`
- `results/iris_quantum_temp_lambda_pred_ablation.csv`
- `results/iris_quantum_temp_lambda_pred_ablation.json`
- `results/iris_quantum_temp_normalized_check.json`
- `results/iris_quantum_temp_phase151_final.csv`
- `results/iris_quantum_temp_phase151_final.json`
- `configs/iris_quantum_temp_selected.json`
