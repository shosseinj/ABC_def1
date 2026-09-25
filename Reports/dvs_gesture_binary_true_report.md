# True DVS-Gesture Binary-grid clean accuracy

Status: **PASS**

All checkpoints were independently initialized and trained with exact Binary occupancy during training and evaluation. No normalized-count checkpoint was reused.

| Seed | Correct / 264 | Accuracy | Class-collapse check |
|---:|---:|---:|---|
| 42 | 208 | 78.79% | PASS |
| 123 | 213 | 80.68% | PASS |
| 777 | 223 | 84.47% | PASS |

Three-seed mean: **81.31 ± 2.89%** (sample SD). Two-sided 95% Student-t CI: **74.13–88.50%**.

These results supersede the invalid 12.50%, 19.32%, and 9.47% evaluations that applied Binary occupancy to normalized-count-trained checkpoints. Binary attacks remain pending and must use these representation-matched checkpoints and Binary-only clean-correct manifests.
