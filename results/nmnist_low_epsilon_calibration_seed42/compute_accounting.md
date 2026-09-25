# Attack Compute Accounting

The attack implementations were not modified.

## PGD

For 20 update steps, the current implementation evaluates:

- 21 hard-forward candidate evaluations, including the initial point and the final iterate;
- 20 differentiable model forwards used to obtain gradients;
- 20 backward/gradient evaluations.

Thus the defensible raw accounting is 41 model forward evaluations, 20 backward evaluations, and 21 hard candidate evaluations per sample and epsilon. A single scalar "query" count is not scientifically equivalent across attacks; if a unified budget is required, report both forward and backward counts rather than equating one PGD step with one TEMP candidate.

## TEMP-DRIFT-v2

The fixed search performs 600 initial candidates plus four generations of 250 candidates, for 1,600 candidate evaluations. Every candidate is a frozen-model forward evaluation. It performs no backward pass. Its accounting is therefore 1,600 candidate evaluations, 1,600 model forwards, and zero backward evaluations per sample and epsilon.

The final five-seed campaign should preserve these three fields separately. Compute-matched attacks were explicitly not run in this calibration.
