# Phase 20 — Preregistered Cosine-Margin Calibration

## 1. Agents and Skills Used

Used scientific-critical-thinking, experimental-design, statistical-analysis, and PennyLane guidance. Scientific success is assessed separately from implementation correctness.

## 2. Interpreter

Python interpreter: C:\Users\jafari.h\AppData\Local\Programs\Python\Python310\python.exe. The protected beginner interpretation is `iris_phase20_beginner_summary.md`.

## 3. Files Added

Added the Phase 20 module, runner, tests, exact result artifacts, event/seed ledgers, final artifact manifest, report, and three required plots.

## 4. Files Modified

Modified `experiments/iris/data.py` and `phase_runner.py`; TTFS, circuit, qlayer architecture, and frozen attack definitions were not changed.

## 5. Regression Status

Focused Phase 20 gate passed. This establishes implementation/schema integrity, not defense efficacy.

## 6. Phase 20 Preregistered Protocol

Local filesystem evidence places initial hash `ff847a202a35a0a0d127cf0c77ff75da7e8b61eb7540fd915e92f8d0c9f6b93f` before the first observed checkpoint only. Current hash `11e4d74fcb979de61d75ef94d711d558a81c0735c253ede8be188be3d46ca950` is a post-output audit correction. Neither is independently registered/timestamped, and timing before every output is not established.

## 7. New Development Splits

Seeds [314, 1001, 4096, 9001, 12345] were absent from the recorded seed-context scan. The ledger cannot detect deleted, unrecorded, or external runs.

## 8. Model Seeds

Model seeds [42, 777, 2026] are nested repetitions within each of five splits; split seed is the inferential unit (n=5), not 15 independent units.

## 9. Frozen HEAD_A Reference

HEAD_A is the warm/co-adapted deployment reference. Its aggregate accuracy for the best-observed comparison was 0.9267, class1 0.8333, class2 0.9467, and class1/2 margin 0.7464.

## 10. HEAD_E Calibration Grid

Exactly nine HEAD_E configurations used s={5,10,15}, m={.05,.10,.15}, with training z_j=s(cos_j-m I[j=y]) and inference z_j=s cos_j. All nine shared initialization within split/model; no extension occurred.

## 11. Clean Calibration Results

Q1: Best observed by mean accuracy delta was s15_m0.15: accuracy 0.9333 versus A 0.9267, delta +0.0067, 95% split-level CI [-0.0143, +0.0276]. It was not eligible.

## 12. Split-Level Stability

Q2: s15_m0.15 accuracy directions were 2 positive, 2 tied, and 1 negative splits; deltas were [0.0, 0.0333333412806193, 0.011111100514729818, 0.0, -0.011111120382944742]. The CI crosses zero.

## 13. Class 1/2 Results

Q3: s15_m0.15 class1 accuracy was 0.8400 (A 0.8333); class2 was 0.9600 (A 0.9467); class1/2 binary accuracy was 0.9000 (A 0.8900).

## 14. Margin Stability

Q4: s15_m0.15 class1/2 mean margin was 2.9869 versus A 0.7464, split-mean delta +2.2405; scale-dependent raw margins are not formulation-comparable without qualification.

## 15. Sample 119 Diagnostic

Q5: Sample 119 was not present in the Phase 20 validation diagnostic records. Ordinary canonical membership was retained wherever assigned; its identity was never a criterion, objective, filter, tie-break, or tuning target.

## 16. Repeated Fragile Samples

Q6: Fragility was aggregated post hoc by original ID across 102 observed validation IDs. These diagnostics were absent from selection inputs.

## 17. Clean Calibration Gate

Q7: Clean gate failed. Although s15_m0.15 had the largest observed mean accuracy delta, minimum-class deltas were nonnegative in only 1/5 splits (required at least 3); its minimum-class split deltas were [-0.0333333412806193, 0.16666664679845175, -0.06666666269302368, -0.0333333412806193, -0.03333336114883423].

## 18. Frozen Selected Configuration

Q8: `iris_phase20_selected_config.json` records selection_count=0 and selected=null before the attack branch. No configuration was frozen for attack evaluation.

## 19. Random Timing Results

Q9: Random timing was explicitly not run because the clean prerequisite failed; the attack artifacts contain `status=not_run`.

## 20. Classical PGD Results

Q10: Unchanged Phase 14 PGD at 1/2/5/10% was not invoked. Frozen source hashes remain recorded, but there are no Phase 20 PGD observations.

## 21. Paired Robustness

Q11: Common-clean-correct paired rescued/broken/both-fail/both-robust outcomes are not estimable because no candidate entered attacks; no changed denominator was used to claim robustness.

## 22. Small-Epsilon Safety

Q12: Small-epsilon safety is not estimable. Absence of attack execution is not evidence of safety or robustness.

## 23. Final Phase 20 Gate

Final gate: clean=false, attacks_run=false, attack=false, end_to_end=false, Phase21 candidate=null. The artifact manifest and focused tests validate the generated contract.

## 24. Statistical Interpretation

Paired split-level summaries use n=5 splits and descriptive t4 95% CIs. The best observed improvement is uncertain, and minimum-class direction failed the preregistered threshold. Model seeds are not treated as independent inferential replicates.

## 25. Independent Scientific Audit

Independent scientific audit: PASS after material corrections; see `iris_phase20_scientific_audit.md`. That agent-owned artifact is outside generator-complete hashes and may be populated externally.

## 26. Test-Access Status

No hidden test features were received or evaluated. The IDs/labels-only manifest helper may internally cause sklearn to materialize its canonical bundle, but Phase20 receives no hidden feature rows; full feature-returning held-out APIs are fail-closed.

## 27. Scientific Conclusion

Evidence supports a negative clean calibration gate: this fixed narrow grid did not produce an eligible deployment candidate. It does not support robustness, hidden-test generalization, or the claim that every possible classifier calibration is ineffective.

## 28. Beginner-Friendly Explanation

See `iris_phase20_beginner_summary.md`, the protected agent-owned interpreter artifact.

## 29. Recommended Phase 21

clean gate failed, narrow calibration insufficient, do not broaden classifier search, no Phase21 candidate, return to representation robustness only as next research direction.
