# Clean-accuracy representation audit

## Verdict

**FAIL — a concrete representation mismatch was found.** All nine checkpoints were trained only on additive count or per-sample-normalized count inputs. No checkpoint is explicitly Binary-grid trained. The clean evaluator and Phase 1 N-MNIST attack runner threshold the same integer-trained model input to occupancy and reuse the integer-trained checkpoint. Therefore all current Binary-grid clean accuracies and Binary-grid attack rows are **REPRESENTATION_MISMATCH** and must not be presented as Binary-grid SNN benchmark results.

DVS-Gesture and CIFAR10-DVS Binary accuracy is near random because occupancy thresholding changes the amplitude distribution seen by BatchNorm and fixed-threshold LIF neurons; it is out-of-distribution inference, not evidence of weak properly trained Binary-grid models. N-MNIST seed variability is likewise OOD checkpoint sensitivity, not a label, channel, time-order, or split bug.

DVS-Gesture and CIFAR10-DVS count checkpoints were trained on per-sample-max-normalized fractional values. Those rows may be described as normalized-count results, but they are **NON_COMPARABLE** to a strict Integer-grid protocol that preserves integer cell amplitudes.

A second, narrower mismatch exists for CIFAR10-DVS seed 42: it was trained/evaluated originally with on-the-fly float32 frames, while the benchmark clean evaluator uses a float16 cache cast back to float32. Seeds 123 and 777 were trained from that cache.

## Checkpoints

| Dataset | Seed | Checkpoint | Architecture | Trained representation | Binary checkpoint? | Same checkpoint reused? |
|---|---:|---|---|---|---|---|
| N-MNIST | 42 | `checkpoints/nmnist_snn_clean_seed42_best.pt` | NMNISTConvSNN | integer_count | No | Yes |
| N-MNIST | 123 | `checkpoints/nmnist_snn_clean_seed123_best.pt` | NMNISTConvSNN | integer_count | No | Yes |
| N-MNIST | 777 | `checkpoints/nmnist_snn_clean_seed777_best.pt` | NMNISTConvSNN | integer_count | No | Yes |
| DVS-Gesture | 42 | `checkpoints/dvs_gesture_snn_seed42_best.pt` | DVSGestureConvSNN | per_sample_max_normalized_count | No | Yes |
| DVS-Gesture | 123 | `checkpoints/dvs_gesture_snn_seed123_best.pt` | DVSGestureConvSNN | per_sample_max_normalized_count | No | Yes |
| DVS-Gesture | 777 | `checkpoints/dvs_gesture_snn_seed777_best.pt` | DVSGestureConvSNN | per_sample_max_normalized_count | No | Yes |
| CIFAR10-DVS | 42 | `checkpoints/cifar10_dvs_snn_seed42_best.pt` | CIFAR10DVSConvSNN | per_sample_max_normalized_count | No | Yes |
| CIFAR10-DVS | 123 | `checkpoints/cifar10_dvs_snn_seed123_best.pt` | CIFAR10DVSConvSNN | per_sample_max_normalized_count | No | Yes |
| CIFAR10-DVS | 777 | `checkpoints/cifar10_dvs_snn_seed777_best.pt` | CIFAR10DVSConvSNN | per_sample_max_normalized_count | No | Yes |

All checkpoint hashes matched the frozen specification, embedded seeds matched the requested seeds, and strict architecture loading passed. Full payload metadata and hashes are in the audit JSON.

### Recorded training-time metrics

These metrics are for each checkpoint's trained count/count-normalized representation, not Binary occupancy.

| Dataset | Seed | Best validation accuracy | Recorded test accuracy |
|---|---:|---:|---:|
| N-MNIST | 42 | 98.24% | 98.49% |
| N-MNIST | 123 | 97.96% | 98.14% |
| N-MNIST | 777 | 98.36% | 98.53% |
| DVS-Gesture | 42 | 85.19% | 84.09% |
| DVS-Gesture | 123 | 85.65% | 78.03% |
| DVS-Gesture | 777 | 88.89% | 87.12% |
| CIFAR10-DVS | 42 | 42.50% | 39.10% |
| CIFAR10-DVS | 123 | 53.10% | 49.20% |
| CIFAR10-DVS | 777 | 49.70% | 50.30% |

## Training vs evaluation vs attack preprocessing

| Field | N-MNIST | DVS-Gesture | CIFAR10-DVS |
|---|---|---|---|
| Temporal bins | 10 / 10 / 10 | 10 / 10; attack not implemented | 10 / 10; attack not implemented |
| Bin assignment | `floor((t-t0)*10/duration)`, clipped; identical | Same formula; cached tensor identical | Same formula; float16 cache quantizes float32 result |
| Layout | `[T,polarity,y,x]`; identical | `[T,polarity,y,x]`; identical | `[T,polarity,y,x]`; identical |
| Polarity order | channel 0 then 1; identical | channel 0 then 1; identical | channel 0 then 1; identical |
| Resolution | 34×34; no crop/resize | 64×64 after `x//2,y//2` | 128×128; no crop/resize |
| Count handling | additive integer counts, no normalization | additive counts divided by per-sample maximum | additive counts divided by per-sample maximum |
| Training dtype | uint8 counts → float32 | float16 cache → float32 | seed 42 float32; seeds 123/777 float16 cache → float32 |
| Evaluation dtype | same for Integer; Binary `(x>0).float32` | float16 cache → float32; Binary threshold | float16 cache → float32; Binary threshold |
| Binary training path | **Absent** | **Absent** | **Absent** |
| Phase 1 attack path | same uint8 count frames; Binary threshold or Integer unchanged | not implemented/run | not implemented/run |

## Validity assignments

| Dataset | Representation | Seed | Status | Reason |
|---|---|---:|---|---|
| N-MNIST | binary | 42 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| N-MNIST | binary | 123 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| N-MNIST | binary | 777 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| N-MNIST | integer | 42 | **VALID** | training representation, checkpoint, labels, and evaluation preprocessing are compatible |
| N-MNIST | integer | 123 | **VALID** | training representation, checkpoint, labels, and evaluation preprocessing are compatible |
| N-MNIST | integer | 777 | **VALID** | training representation, checkpoint, labels, and evaluation preprocessing are compatible |
| DVS-Gesture | binary | 42 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| DVS-Gesture | binary | 123 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| DVS-Gesture | binary | 777 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| DVS-Gesture | integer | 42 | **NON_COMPARABLE** | fractional per-sample-normalized counts, not strict Integer-grid amplitudes |
| DVS-Gesture | integer | 123 | **NON_COMPARABLE** | fractional per-sample-normalized counts, not strict Integer-grid amplitudes |
| DVS-Gesture | integer | 777 | **NON_COMPARABLE** | fractional per-sample-normalized counts, not strict Integer-grid amplitudes |
| CIFAR10-DVS | binary | 42 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| CIFAR10-DVS | binary | 123 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| CIFAR10-DVS | binary | 777 | **REPRESENTATION_MISMATCH** | checkpoint was trained only on count/count-normalized input, not binary occupancy |
| CIFAR10-DVS | integer | 42 | **INVALID_PREPROCESSING** | seed-42 checkpoint trained with float32 on-the-fly frames but benchmark evaluation uses float16 cache |
| CIFAR10-DVS | integer | 123 | **NON_COMPARABLE** | fractional per-sample-normalized counts, not strict Integer-grid amplitudes |
| CIFAR10-DVS | integer | 777 | **NON_COMPARABLE** | fractional per-sample-normalized counts, not strict Integer-grid amplitudes |

## Preprocessing and alignment checks

- **N-MNIST:** 20-sample training/evaluation exact tensors 20/20; labels 20/20; binary occupancy semantics PASS=True; max absolute tensor difference=0.
- **DVS-Gesture:** 20-sample training/evaluation exact tensors 20/20; labels 20/20; binary occupancy semantics PASS=True; max absolute tensor difference=0.
- **CIFAR10-DVS:** 20-sample training/evaluation exact tensors 3/20; labels 20/20; binary occupancy semantics PASS=True; max absolute tensor difference=0.000233531.
- **N-MNIST labels:** 100/100 deterministic sample IDs aligned; 10 classes; mismatches=0.
- **DVS-Gesture labels:** 100/100 deterministic sample IDs aligned; 11 classes; mismatches=0.
- **CIFAR10-DVS labels:** 100/100 deterministic sample IDs aligned; 10 classes; mismatches=0.

No polarity swap, channel swap, time reversal, label permutation, tensor-layout error, wrong checkpoint hash, architecture-load mismatch, or split-index mismatch was detected. Binary tensors contain only 0 and 1 and preserve occupancy exactly; the bug is using them with checkpoints that were never trained on them.

## Model-input compatibility

All model shapes matched: N-MNIST `[B,10,2,34,34]`, DVS-Gesture `[B,10,2,64,64]`, and CIFAR10-DVS `[B,10,2,128,128]`. Every model contains inference-mode BatchNorm and fixed-threshold (`1.0`) LIF dynamics. Binary conversion therefore changes the amplitude distribution relative to checkpoint training even though shape, channels, polarity, and temporal order remain correct. Exact observed ranges and means are recorded per dataset and representation in `input_compatibility` within the audit JSON.

## Confusion and activation evidence

Full confusion matrices for every dataset, representation, and seed are in `Reports/results/clean_accuracy_confusion/`. Per-class accuracy and prediction/true histograms are in `Reports/results/clean_accuracy_audit.json`. Same-sample logits are in `Reports/results/accuracy_audit/paired_binary_integer_logits.csv`; layer spike-rate summaries are in `Reports/results/accuracy_audit/activation_summary.csv`.

DVS-Gesture Binary predictions collapse to only 4–5 of 11 classes (dominant-class counts 224/264, 140/264, and 254/264 for seeds 42/123/777). CIFAR10-DVS seed 42 uses only 3 classes and assigns 643/1000 samples to class 6; seeds 123 and 777 use all classes but remain strongly concentrated (largest bins 366 and 599). N-MNIST uses all 10 classes, but seeds 42 and 777 overpredict class 8 (2,103 and 2,631 samples), while seed 123 is much less collapsed. This explains the N-MNIST Binary seed variability.

## Consequences for attack results

- **All current Binary-grid attack results:** preserve as diagnostic artifacts, but mark **NON_COMPARABLE / REPRESENTATION_MISMATCH** and remove from Binary-grid paper benchmark rows.
- **N-MNIST Integer-grid attack results:** internally usable as attacks on count-trained checkpoints and independently valid under the frozen packet contract. They remain `NON_COMPARABLE` to the reference paper because the attacked manifest is selected from the Binary∩Integer clean-correct intersection rather than an independently defined Integer clean-correct population.
- **DVS-Gesture/CIFAR10-DVS attacks:** none were rerun by this audit. Any future Binary attack must wait for separately trained Binary checkpoints and new manifests. Their current count models are normalized-count, not strict Integer-grid models.
- **CIFAR10-DVS seed42 Integer row:** mark **INVALID_PREPROCESSING** pending resolution of float32-training versus float16-cache evaluation; do not aggregate it with seeds 123/777 as if preprocessing were identical.

## Minimal corrective action

1. Stop and do not interpret current Binary rows as Binary-grid model results.
2. Mark or remove those table rows; do not overwrite their artifacts.
3. If Binary-grid benchmarking remains required, train separate Binary-preprocessed checkpoints, freeze them, compute full-test clean accuracy, create representation-appropriate clean-correct manifests, and rerun only Binary attacks.
4. Relabel DVS-Gesture/CIFAR10-DVS as normalized-count or define and train strict Integer-grid checkpoints.
5. Resolve CIFAR10-DVS seed42 float32/float16 preprocessing provenance before aggregating its Integer result.

No models were trained and no attacks were run or modified during this audit.
