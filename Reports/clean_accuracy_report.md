# Full-test clean accuracy report

Clean accuracy was evaluated without attacks on every sample in each frozen benchmark test split. No attack clean-correct manifest, prediction filtering, class balancing, or subsampling was used. All Binary rows now use independently trained, representation-matched Binary checkpoints; earlier count-checkpoint-on-Binary values are preserved under `Reports/results/legacy/`.

## Results

| Dataset | Representation | Seed 42 | Seed 123 | Seed 777 | Mean ± sample SD | 95% t CI |
|---|---|---:|---:|---:|---:|---:|
| N-MNIST | Binary-grid | 98.7300% | 98.2900% | 98.5300% | 98.5167% ± 0.2203% | [97.9694%, 99.0639%] |
| N-MNIST | Integer-grid | 98.4900% | 98.1400% | 98.5300% | 98.3867% ± 0.2146% | [97.8537%, 98.9196%] |
| DVS-Gesture | Binary-grid | 78.7879% | 80.6818% | 84.4697% | 81.3131% ± 2.8930% | [74.1264%, 88.4998%] |
| DVS-Gesture | Integer-grid | 84.0909% | 78.0303% | 87.1212% | 83.0808% ± 4.6289% | [71.5821%, 94.5795%] |
| CIFAR10-DVS | Binary-grid | 48.1000% | 51.8000% | 51.3000% | 50.4000% ± 2.0075% | [45.4131%, 55.3869%] |
| CIFAR10-DVS | Integer-grid | 39.1000% | 49.2000% | 50.3000% | 46.2000% ± 6.1733% | [30.8646%, 61.5354%] |

## Run details

- **N-MNIST / binary / seed 42**: 9873/10000 correct = 98.7300%; checkpoint `checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **N-MNIST / binary / seed 123**: 9829/10000 correct = 98.2900%; checkpoint `checkpoints/nmnist_binary_true/nmnist_binary_seed123_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **N-MNIST / binary / seed 777**: 9853/10000 correct = 98.5300%; checkpoint `checkpoints/nmnist_binary_true/nmnist_binary_seed777_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **N-MNIST / integer / seed 42**: 9849/10000 correct = 98.4900%; checkpoint `checkpoints/nmnist_snn_clean_seed42_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **N-MNIST / integer / seed 123**: 9814/10000 correct = 98.1400%; checkpoint `checkpoints/nmnist_snn_clean_seed123_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **N-MNIST / integer / seed 777**: 9853/10000 correct = 98.5300%; checkpoint `checkpoints/nmnist_snn_clean_seed777_best.pt`; split: Tonic N-MNIST official test partition (train=False).
- **DVS-Gesture / binary / seed 42**: 208/264 correct = 78.7879%; checkpoint `checkpoints/dvs_gesture_binary_true/dvs_gesture_binary_seed42_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **DVS-Gesture / binary / seed 123**: 213/264 correct = 80.6818%; checkpoint `checkpoints/dvs_gesture_binary_true/dvs_gesture_binary_seed123_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **DVS-Gesture / binary / seed 777**: 223/264 correct = 84.4697%; checkpoint `checkpoints/dvs_gesture_binary_true/dvs_gesture_binary_seed777_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **DVS-Gesture / integer / seed 42**: 222/264 correct = 84.0909%; checkpoint `checkpoints/dvs_gesture_snn_seed42_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **DVS-Gesture / integer / seed 123**: 206/264 correct = 78.0303%; checkpoint `checkpoints/dvs_gesture_snn_seed123_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **DVS-Gesture / integer / seed 777**: 230/264 correct = 87.1212%; checkpoint `checkpoints/dvs_gesture_snn_seed777_best.pt`; split: Tonic DVS-Gesture official test partition (train=False).
- **CIFAR10-DVS / binary / seed 42**: 481/1000 correct = 48.1000%; checkpoint `checkpoints/cifar10_dvs_binary_true/cifar10_dvs_binary_seed42_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).
- **CIFAR10-DVS / binary / seed 123**: 518/1000 correct = 51.8000%; checkpoint `checkpoints/cifar10_dvs_binary_true/cifar10_dvs_binary_seed123_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).
- **CIFAR10-DVS / binary / seed 777**: 513/1000 correct = 51.3000%; checkpoint `checkpoints/cifar10_dvs_binary_true/cifar10_dvs_binary_seed777_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).
- **CIFAR10-DVS / integer / seed 42**: 391/1000 correct = 39.1000%; checkpoint `checkpoints/cifar10_dvs_snn_seed42_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).
- **CIFAR10-DVS / integer / seed 123**: 492/1000 correct = 49.2000%; checkpoint `checkpoints/cifar10_dvs_snn_seed123_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).
- **CIFAR10-DVS / integer / seed 777**: 503/1000 correct = 50.3000%; checkpoint `checkpoints/cifar10_dvs_snn_seed777_best.pt`; split: Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition).

## Sanity checks

- **N-MNIST: PASS.** Evaluated 10000/10000 official-test samples per seed; Binary checkpoints were independently trained with exact occupancy preprocessing in training and evaluation; no count checkpoint was reused; all seeds used all 10 prediction classes and passed the class-collapse check.
- **DVS-Gesture Binary: PASS.** Evaluated all 264 official-test samples per seed using independently initialized Binary-trained checkpoints; exact occupancy was used in training and evaluation; all seeds used all 11 classes and passed the collapse check.
- **CIFAR10-DVS Binary: PASS.** Evaluated all 1,000 frozen test-split samples per seed using independently initialized Binary-trained checkpoints. The shared hash-verified cache stores uint8 raw-event occupancy with no normalization and is cast to float32 only at model input; all seeds used all 10 classes and passed the collapse check.
- **Integer caveat:** the currently listed DVS-Gesture and CIFAR10-DVS Integer rows remain legacy normalized-count results, not strict raw-integer-grid results. They remain `NON_COMPARABLE` (and CIFAR10-DVS seed 42 additionally has the earlier precision mismatch) until strict Integer models are trained.

## Statistical note

The summary uses the arithmetic mean and sample standard deviation (`ddof=1`) across the three requested seeds. The 95% interval is a two-sided Student-t interval with df=2. With only three seeds it is mathematically defined but very imprecise and should be interpreted cautiously.

CIFAR10-DVS has no official train/test partition. Accordingly, its complete test set here is the frozen class-stratified 10% benchmark split, not an attack subset.

## Statistical workflow reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
