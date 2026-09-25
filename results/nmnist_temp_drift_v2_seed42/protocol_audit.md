# TEMP-DRIFT-v2 Seed-42 Audit

The historical flawed TEMP-DRIFT implementation and outputs were preserved in their original directories. This directory is a new versioned protocol.

## Objective

For logits `z` and true label `y`, the attack objective is

`m(z,y) = z_y - max(z_j, j != y)`.

The search minimizes `m`. Therefore a lower candidate value means lower true-class separation. If `m < 0`, the strongest competitor exceeds the true-class logit and the candidate is misclassified. Unit tests explicitly verify these inequalities and the candidate-selection direction.

## Checks

- Objective-direction unit tests: passed, 4 tests.
- Feasibility: 100% of 1,200 primary records satisfied timestamp bounds and nondecreasing ordering.
- Candidate budget: 1,600 classifier queries per sample and epsilon.
- Repeated ASR, 5%: SNN 0.92, 0.91, 0.92; QSNN 0.99, 0.99, 0.99.
- Repeated ASR, 10%: SNN 0.91, 0.91, 0.92; QSNN 0.98, 0.98, 0.98.
- All per-sample objective values, predictions, timestamp vectors, query counts, runtimes, and feasibility flags are saved.

The objective implementation is directionally correct and reproducible at aggregate ASR level. `objective_improved_rate` is below 1 for some cells because the fixed derivative-free budget does not find a lower-margin candidate for every sample; this is an optimization-efficiency result, not a sign inversion. The five-seed campaign remains deferred pending scientific review of whether this budget and candidate initialization are sufficient for the intended claim.
