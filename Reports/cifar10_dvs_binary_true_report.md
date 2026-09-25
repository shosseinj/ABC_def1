# True CIFAR10-DVS Binary-grid clean accuracy

Status: **PASS**

All checkpoints were independently initialized and trained using exact Binary occupancy. The shared cache is raw-event-derived uint8 occupancy, has no normalization, and is cast to float32 only at model input. This eliminates the former normalized-float16 representation mismatch.

| Seed | Correct / 1,000 | Accuracy | Class-collapse check |
|---:|---:|---:|---|
| 42 | 481 | 48.10% | PASS |
| 123 | 518 | 51.80% | PASS |
| 777 | 513 | 51.30% | PASS |

Three-seed mean: **50.40 ± 2.01%** (sample SD). Two-sided 95% Student-t CI: **45.41–55.39%**.

These results supersede the invalid 12.20%, 18.60%, and 17.30% evaluations that applied Binary occupancy to normalized-count-trained checkpoints. Binary attacks remain pending and must use these checkpoints and Binary-only clean-correct manifests.
