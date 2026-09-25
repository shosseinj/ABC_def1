# Seed-42 N-MNIST Attack Calibration Audit

## Scope

The frozen Seed-42 SNN and QSNN-v3 checkpoints were not changed or retrained. The same 100 common clean-correct sample IDs and labels were used. No five-seed attack campaign was started.

## TEMP-DRIFT finding

The observed QSNN TEMP-DRIFT ASR decrease at larger epsilon is reproducible, not primarily random-search noise: three trials produced ASR 0.89, 0.89, 0.89 at 5% and 0.77, 0.77, 0.78 at 10%.

The current TEMP-DRIFT implementation contains an objective-direction defect. `temp_drift()` assigns each candidate `-out[:, label]`, then retains candidates with the smallest values and finally returns `argmin(values)`. Minimizing the negative true-class logit maximizes the true-class logit. This is the opposite of an untargeted attack objective. The implementation therefore cannot be interpreted as a valid optimized TEMP-DRIFT attack.

The fixed candidate budget is 1,600 candidates per sample and epsilon. Repeated trials show that stochasticity at this budget changes aggregate ASR by at most one percentage point in the investigated cells, so candidate randomness is not the main explanation.

All saved candidates satisfy the timestamp bound and nondecreasing-order feasibility checks. At 10%, QSNN TEMP-DRIFT moves a mean 89.38% of events to a different temporal bin, versus 44.31% at 5%. Thus the decrease is not caused by failure to perturb the representation. The projection and ordering constraints strongly couple timestamp changes and cause broad bin migration at high epsilon, but no feasibility failure was observed.

Independent searches are performed at each epsilon and candidates from smaller epsilon are not retained in larger-epsilon searches. Consequently, monotonic ASR is not guaranteed even after correcting the objective unless the protocol explicitly includes nested candidate retention. Such a change must be declared and frozen before the five-seed campaign.

## Decision

The protocol is not ready for a five-seed campaign. The TEMP-DRIFT objective must first be corrected in a new versioned protocol, then rerun on this same Seed-42 calibration set. Existing negative and anomalous results must remain preserved as evidence from the flawed implementation and must not be silently overwritten.
