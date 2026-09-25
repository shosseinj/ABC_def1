# Phase 18 — Representation-Bottleneck Diagnosis

## 1. Objective

Locate the earliest measurable class-1/class-2 separability bottleneck without retraining or test evaluation.

## 2. Diagnostic scope

Exploratory diagnosis on frozen training and validation observations only; no robustness claim.

## 3. Frozen inputs

Split seed 42; model seeds 42/777/2026; Phase 17.2 baselines and prespecified Phase 17.1 PGD-1 defenses.

## 4. Canonical dataset sanity

`iris_phase18_dataset_sanity.csv/json` verifies sklearn load_iris, 150x4, three classes of 50, and retained 90/30/30 split counts.

## 5. Held-out boundary

The restricted loader returned no test arrays and no test loader was invoked. sklearn internally materializes its canonical table; Phase 18 selected no held-out feature row from it.

## 6. Stable sample identity

Original Iris row indices are retained; train/validation IDs are unique and disjoint and all representation alignments were asserted.

## 7. Example CSV role

Any example CSV is documentation only and was excluded as a canonical data input.

## 8. Raw representation

Retained raw vectors are exact canonical load_iris().data rows indexed only by train_ids/val_ids. Validation raw class-1/2 ratio was 1.1684.

## 9. Normalization

Scaler fitting used training only. Validation normalized class-1/2 ratio was 0.8578.

## 10. TTFS representation

Continuous TTFS is T(1-x), T=100. Validation TTFS ratio was 0.8578, equal to normalized within numerical precision.

## 11. TTFS information-loss test

Raw/normalized/TTFS training-standardized linear diagnostics were each 0.90 on the same 20 class-1/2 validation rows; no extra continuous-TTFS loss was detected.

## 12. TTFS collisions

Cross-class exact=0 and near=0 at frozen L-infinity <= 0.5.

## 13. TTFS saturation and ordering

{"ordering_violations_train": 0, "ordering_violations_validation": 0, "train_T": 7, "train_zero": 3, "validation_T": 1, "validation_zero": 5}

## 14. Quantum measured features

Four pre-head Pauli-Z expectations were extracted for all six frozen models. Validation separation remained nonzero in every model.

## 15. Logits

Three frozen-head logits were extracted without optimization or checkpoint selection.

## 16. Probabilities

Three softmax probabilities were derived from each logit vector; probability JS uses these exact rows.

## 17. Representation separability

`iris_phase18_representation_separability.csv/json` reports class means, sample SDs, centroids, mean radial spreads, centroid distance, ratio, nearest-centroid accuracy and k=3 validation purity.

## 18. Linear separability

Training-fitted standardization and logistic regression used 60 class-1/2 training rows; evaluation used 20 validation rows. Quantum accuracy ranged 0.85--1.00.

## 19. Nearest-centroid and purity controls

Validation assignment references training centroids/neighbors. Stable ties use distance, original ID and class order; self-neighbors are excluded.

## 20. Sample 119 pipeline

Sample 119 is validation class 2 with raw [6.0, 2.2, 5.0, 1.5]; every stage, distance, neighbor label, logit, probability, margin and prediction is reported.

## 21. Prespecified samples 122 and 142

Both controls were traced with the same rules and were not selected after viewing outcomes.

## 22. Margin analysis

`iris_phase18_margin_analysis.csv` reports class n, mean, median, sample SD, minimum, p10, negative/<0.05/<0.10/<0.20 counts and class accuracy per seed/model. Explicit <0.20 findings (count/n): seed 42 baseline class 0: 0/10; seed 42 baseline class 1: 10/10; seed 42 baseline class 2: 6/10; seed 42 defense class 0: 0/10; seed 42 defense class 1: 10/10; seed 42 defense class 2: 6/10; seed 777 baseline class 0: 0/10; seed 777 baseline class 1: 4/10; seed 777 baseline class 2: 2/10; seed 777 defense class 0: 0/10; seed 777 defense class 1: 3/10; seed 777 defense class 2: 2/10; seed 2026 baseline class 0: 0/10; seed 2026 baseline class 1: 5/10; seed 2026 baseline class 2: 4/10; seed 2026 defense class 0: 0/10; seed 2026 defense class 1: 10/10; seed 2026 defense class 2: 9/10.

## 23. Seed comparison

Quantum linear separability was 0.85/0.90 for seed 42 baseline/defense and 1.00 for both models at seeds 777 and 2026; frozen-head clean behavior remained seed dependent.

## 24. Frozen Phase 14 attack

Classical PGD source hash is recorded; epsilon 2%/10%, 20 iterations, alpha=epsilon/5 and no random start were unchanged.

## 25. Attack cross-stage sensitivity

Per class/model/seed, tables report timing, quantum and logit L2 movement, probability JS and safe-denominator ratios. These are descriptive cross-unit sensitivities: timing, expectation, logit and JS scales have different units. Nonzero drift or flips do not establish adversarial amplification, and no dimensionless amplification criterion was prespecified.

## 26. Paired attack outcomes

Common-clean-correct comparisons report rescued, broken, both-fail and both-robust. At 2%, seed 2026 had 0 rescued and 2 broken; effects are not uniformly favorable.

## 27. Answers to 14 scientific questions

Q1 source: canonical sklearn. Q2 split: 90/30 validation with 30 held out. Q3 IDs: stable. Q4 raw ambiguity: descriptive overlap exists. Q5 normalization loss: scale-dependent geometry changes but linear information is retained. Q6 TTFS loss: not detected. Q7 collisions: none. Q8 quantum compression: not reproducible across seeds. Q9 head instability: a plausible descriptive contributor. Q10 sample 119: atypical mixed profile, not class-wide. Q11 attack response: measurable drift and flips, not proof of amplification. Q12 defense: not uniformly beneficial. Q13 robustness: unsupported. Q14 bottleneck: no unique causal bottleneck was established.

## 28. Root-cause categories

If forced into the requested taxonomy, assign F MIXED_CAUSE with qualified, non-causal evidence. A RAW_DATA_AMBIGUITY and D CLASSIFIER_BOUNDARY_INSTABILITY may be listed only as descriptive contributors. E ADVERSARIAL_AMPLIFICATION is not selected because no defensible dimensionless criterion was prespecified. Evidence is limited to n=20 class-1/class-2 validation observations, three checkpoint seeds, and one fixed split; it does not establish a unique causal bottleneck.

## 29. Phase 19 recommendation

Preregister one classifier/head diagnostic ablation—not a remedy—fit on training representations and compared with paired IDs across additional split seeds before any held-out test access.

Phase 14 SHA-256: `08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65`

All detailed values are retained regardless of favorability.
