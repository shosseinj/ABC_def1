# Phase 21 — Representation Shift and Conditional Consistency

## 1. Agents and Skills Used

Used scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane guidance. Implementation correctness remains separate from scientific success.

## 2. Interpreter

Python interpreter: C:\Users\jafari.h\AppData\Local\Programs\Python\Python310\python.exe. The protected beginner interpretation is `iris_phase21_beginner_summary.md`.

## 3. Files Added

Added the Phase 21 implementation, runner, tests, exact Stage-A outputs, three plots, provenance records, report, and exhaustive manifest.

## 4. Files Modified

Modified `phase_runner.py`, `experiments/iris/data.py`, `experiments/iris/training.py`, `experiments/iris/phase21.py`, `scripts/run_phase21_representation_consistency.py`, and `tests/test_phase21_representation_consistency.py`. TTFS, circuit, and attacks were unchanged.

## 5. Regression Status

Focused tests pass. This establishes implementation and artifact-contract correctness, not defense efficacy.

## 6. Phase 21 Protocol

A repository-local initially recorded protocol is identified by hash `afc047aad2294bc01ea13b748324c6b5419538670ed408bf60a4d6b4db29902b`. Prospective timing and immutability are not independently established; the deviation log records audit corrections.

## 7. Development Splits

Preferred split seeds 271, 811, 1618, 2718, and 4242 were absent from recorded seed context, subject to the stated limits. Each split has disjoint 90/30/30 canonical IDs and labels.

## 8. Model Seeds

Model seeds 42, 777, and 2026 are nested repetitions within five splits. Split is the descriptive replication unit, n=5.

## 9. Stage A Representation Metrics

Q1: Exactly 3,600 rows trace canonical IDs through raw and normalized features, TTFS, measured quantum features, logits, margins, and predictions for 15 cells, two attacks, four epsilons, and 30 validation samples.

## 10. Robust vs Failed Samples

Q2: Successful and robust groups both require clean correctness. The complete Cartesian grid includes 44 undefined-empty cells with n=0 and null means; no denominator was silently changed.

## 11. Class-Wise Representation Shift

Q3: Class-wise shifts are descriptive only. Sample 119 remained ordinary canonical data and never entered a gate, objective, filter, or selection rule.

## 12. Centroid Crossing

Q4: All 3,600 centroid records use training-only centroids and retain clean/attacked assignments, distances, and crossing indicators.

## 13. Local Purity Stability

Q5: All local-purity records use deterministic k=3 training-only neighbors and retain clean/attacked purity and label-sequence changes.

## 14. Multi-Split Reproducibility

Q6: Quantum successful-minus-robust contrasts were -0.001428, 0.019474, -0.004941, 0.014382, and -0.002979 for splits 271, 811, 1618, 2718, and 4242; aggregate +0.004901 with descriptive t4 95% CI [-0.008996, 0.018799]. Only 2/5 split directions were positive. Margin contrasts were {271: -0.04621895890260089, 811: -0.010884983136980499, 1618: -0.058810103356758334, 2718: -0.0231554666665636, 4242: -0.03562607165837973}; pooled Spearman is descriptive only. The manifest records all generator-owned results plus exactly 15 checkpoint hashes.

## 15. Stage A Gate

Q7: Stage A failed with criteria {'criterion1_aggregate_quantum_positive': True, 'criterion2_quantum_positive_splits': False, 'criterion3_margin_and_spearman': False, 'criterion4_top_contributor_exclusion': True}: pass/fail/fail/pass. Confirmatory criterion 4 used top contributor ID 73; its exclusion retained 4/5 positive directions. Exhaustive all-ID sensitivity is explicitly post-output/post-hoc and not a gate input.

## 16. Stage B Defense Definition

Q8: The prespecified Stage-B defense was CE plus one cosine representation-consistency loss under random 0.02T jitter. It was not run because Stage A failed.

## 17. Lambda-Repr Ablation

Q9: Lambda-repr values 0.1, 0.5, 1, and 2 were not evaluated; no conditional Stage-B artifacts exist.

## 18. Loss-Scale Analysis

CE, Lrepr, and weighted-loss ratios are not estimable because Stage B was not run.

## 19. Gradient Analysis

Separate qlayer/head gradient norms from CE and Lrepr are not estimable; the expected structural zero/none head contribution was therefore not observed as an experiment result.

## 20. Clean Accuracy

Q10: Candidate clean accuracy and its validation-only clean gate are not estimable. Non-execution is not evidence of clean preservation.

## 21. Clean Representation Geometry

Candidate train-only centroid distance and within-class spread are not estimable.

## 22. Representation Collapse Check

Q11: Anti-collapse geometry and the train-fitted validation linear diagnostic are not estimable because no candidate was trained.

## 23. Random Timing Results

Baseline random-jitter Stage-A observations remain in the diagnosis artifact; candidate random-timing comparisons were not run.

## 24. Classical PGD Results

Baseline Phase-14 PGD metadata record objective, 20 iterations, epsilon*T/5 effective step, no random start, losses, maximum gradient norm, and seed. Candidate PGD was not run.

## 25. Paired Robustness

Q12: Common-clean-correct candidate pairing and rescued, broken, both-fail, and both-robust outcomes are not estimable; no paired candidate artifact exists.

## 26. Small-Epsilon Safety

Small-epsilon candidate safety is not estimable. Failure to run Stage B cannot support a safety or robustness claim.

## 27. TEMP-DRIFT Transfer

TEMP-DRIFT transfer was not run because preceding Stage-B gates were unavailable.

## 28. Stage B Gate

Q13: The Stage-B gate is not estimable and is not treated as passed.

## 29. Selected Configuration

No configuration was selected; selection cardinality is zero and no selected-configuration placeholder exists.

## 30. Independent Scientific Audit

Independent scientific audit status: PASS. See `iris_phase21_scientific_audit.md`; as an agent-owned report it remains excluded from generator-complete hashes.

## 31. Test-Access Status

Q14: No hidden feature rows were requested or returned. The ID-restricted provider rejects hidden IDs and logs allowed requests/subset proof. sklearn internally materializes its canonical bundle before permitted indexing.

## 32. Root-Cause Update

Root classification: C REPRESENTATION_INSTABILITY_NOT_SUPPORTED. This is a scientific mechanism-gate failure, not an implementation failure and not evidence that every representation defense fails.

## 33. Scientific Conclusion

The evidence supports a negative Stage-A gate. It does not support representation instability as the prespecified robust-versus-failed mechanism, hidden-test generalization, or any defense benefit.

## 34. Beginner-Friendly Explanation

See `iris_phase21_beginner_summary.md`. In plain terms, attacked failures did not show the expected reproducible extra quantum movement across splits, so the proposed defense was not tried.

## 35. Recommended Phase 22

Phase 22: no defense; do not continue representation-defense tuning. Independently replicate Stage A only if scientifically justified; otherwise revisit architecture/encoding assumptions only with new evidence.
