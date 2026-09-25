# MNIST 8x8 PCA-Dimension Ablation (Seed 42)

## Frozen comparison protocol

PCA dimension was the configured factor. All arms use the same frozen development IDs (10,000 train, 2,000 validation), split seed 42, model seed 42, 28x28 to 8x8 adaptive average pooling, training-only PCA and training-range scaling, TTFS latency and angle encoding, 8 qubits, 4 cyclic re-upload blocks, circuit depth 44, 234 parameters, Adam at 0.003, batch size 64, gradient clipping at 1.0, 400-epoch maximum, and early stopping after 20 epochs without lower validation CE. Selection is lowest validation CE, then earliest epoch. Held-out data and attacks/defenses were not accessed.

Important interpretation limit: four 8-qubit blocks provide 32 encoding slots. Under the fixed cyclic schedule, PCA-16 uploads every component twice; PCA-24 uploads components 1-8 twice and 9-24 once; PCA-32 uploads every component once. The architecture, capacity, depth, and cyclic rule remain fixed, but component exposure is necessarily dimension-dependent. This experiment therefore tests whether larger PCA inputs help the current four-block cyclic architecture, not the isolated causal value of retained PCA information under equal per-component exposure.

The predeclared clear-improvement threshold is +0.010 absolute validation accuracy relative to PCA-16.

## Comparison

| PCA Dim | Explained Variance | Val Acc | Macro-F1 | Val CE | Best Epoch | Runtime |
| ------: | -----------------: | ------: | -------: | -----: | ---------: | ------: |
| 16 | 0.876441 | 0.8680 | 0.8675 | 0.4432 | 400 | 2763.0 s* |
| 24 | 0.954152 | 0.8530 | 0.8524 | 0.4912 | 336 | 7200.2 s** |
| 32 | 0.984184 | 0.8355 | 0.8346 | 0.5218 | 351 | 7923.8 s |

\* PCA-16 runtime records only its epoch-251-to-400 continuation and is not directly comparable to full-run runtime.

\** PCA-24 runtime is segmented because the process was interrupted after checkpointing epoch 356. The selected epoch-336 state was unchanged; the protocol stopping boundary reconstructed from complete history is epoch 356.

All runtimes are descriptive and non-comparable: PCA-16 covers only its continuation, PCA-24 is segmented, and identical hardware load and software timing conditions were not recorded across arms.

Absolute validation-accuracy changes from PCA-16:

- PCA-24: `-0.0150`
- PCA-32: `-0.0325`

## Configuration details

| PCA Dim | Raw Features | Quantum Input | Qubits | Parameters | Best Epoch | Stop Epoch | Convergence |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 64 | 16 | 8 | 234 | 400 | 400 | Not established |
| 24 | 64 | 24 | 8 | 234 | 336 | 356 | Early-stop criterion reached |
| 32 | 64 | 32 | 8 | 234 | 351 | 371 | Early-stop criterion reached |

Validation per-class accuracy:

| PCA Dim | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.950 | 0.980 | 0.885 | 0.815 | 0.835 | 0.840 | 0.905 | 0.850 | 0.825 | 0.795 |
| 24 | 0.930 | 0.975 | 0.850 | 0.840 | 0.815 | 0.790 | 0.900 | 0.860 | 0.815 | 0.755 |
| 32 | 0.920 | 0.955 | 0.855 | 0.765 | 0.830 | 0.760 | 0.920 | 0.755 | 0.805 | 0.790 |

PCA-16 validation confusion matrix:

```text
[[190, 0, 2, 0, 0, 3, 2, 0, 3, 0],
 [0, 196, 1, 0, 0, 2, 0, 0, 1, 0],
 [1, 1, 177, 3, 4, 0, 6, 1, 7, 0],
 [0, 4, 11, 163, 0, 3, 0, 11, 5, 3],
 [0, 3, 3, 2, 167, 1, 4, 1, 3, 16],
 [7, 1, 3, 9, 0, 168, 6, 1, 4, 1],
 [0, 1, 5, 2, 3, 7, 181, 0, 1, 0],
 [2, 4, 6, 2, 2, 2, 0, 170, 1, 11],
 [1, 7, 6, 9, 2, 8, 0, 0, 165, 2],
 [0, 4, 3, 3, 13, 3, 0, 13, 2, 159]]
```

PCA-24 validation confusion matrix:

```text
[[186, 0, 2, 1, 0, 6, 3, 0, 2, 0],
 [0, 195, 1, 2, 0, 1, 0, 0, 1, 0],
 [2, 1, 170, 3, 3, 1, 7, 1, 11, 1],
 [2, 2, 9, 168, 0, 2, 0, 7, 6, 4],
 [1, 4, 3, 1, 163, 1, 5, 2, 1, 19],
 [8, 1, 2, 13, 1, 158, 6, 1, 6, 4],
 [2, 1, 1, 1, 5, 9, 180, 1, 0, 0],
 [0, 4, 5, 2, 2, 2, 0, 172, 2, 11],
 [1, 4, 4, 10, 4, 11, 0, 1, 163, 2],
 [0, 2, 3, 4, 13, 2, 0, 23, 2, 151]]
```

PCA-32 validation confusion matrix:

```text
[[184, 0, 4, 1, 0, 4, 2, 2, 3, 0],
 [0, 191, 3, 1, 0, 2, 1, 1, 1, 0],
 [1, 2, 171, 4, 4, 0, 9, 4, 5, 0],
 [5, 5, 12, 153, 1, 6, 0, 8, 6, 4],
 [0, 2, 0, 2, 166, 2, 3, 1, 1, 23],
 [9, 3, 5, 10, 2, 152, 5, 2, 10, 2],
 [6, 0, 2, 0, 3, 1, 184, 0, 3, 1],
 [0, 7, 6, 3, 3, 7, 0, 151, 1, 22],
 [2, 7, 8, 9, 0, 8, 1, 4, 161, 0],
 [0, 4, 1, 4, 12, 0, 0, 15, 6, 158]]
```

## Decision

Neither larger PCA dimension met the +0.010 rule; both reduced validation accuracy and worsened validation CE. On this single fixed seed and split, PCA-16 compression is not confirmed as the primary remaining bottleneck for the fixed four-block cyclic architecture. The result does not isolate PCA information from dimension-dependent feature exposure, does not establish a seed-general effect, and does not justify five-seed training automatically.

`MNIST PCA BOTTLENECK: NOT CONFIRMED`
