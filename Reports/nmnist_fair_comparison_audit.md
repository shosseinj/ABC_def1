# N-MNIST four-model comparison audit — 2026-09-27

## Verdict

**A published-reference comparison is now available for N-MNIST Binary Table 1, with explicit limits.** The three reference rows in `readme_jafar.md` are values from Yu et al., arXiv:2602.03284v1, while the custom-model row is measured locally. The published values remain reference results, never local measurements. This comparison does not isolate architecture from training and checkpoint differences.

| Local result | Scientific status for comparison with paper Tables 1–2 | Decisive reason |
|---|---|---|
| Original N-MNIST Binary seed-42 row in README | `NON_COMPARABLE` | Equal-duration timestamp bins and SHA-ordered attacked samples differ from the official equal-event-count frames and first 1,000 clean-correct test samples. |
| Original and corrected N-MNIST Integer rows | `NON_COMPARABLE` | They use the repository's equal-duration temporal grid. The corrected runs fix representation and manifest errors but do not change this paper mismatch. |
| New N-MNIST Binary seed-42 paper-aligned artifacts | `PARTIAL_PROTOCOL_MATCH`; suitable for a descriptive published-reference table | They use equal-event-count `T=10`, the first 1,000 clean-correct test samples, paper budget cells, and a verified official-source attack adapter. The victim is a separately trained 25,482-parameter custom SNN with local training and checkpoint selection. |

The paper's N-MNIST experiment evaluates 1,000 correctly classified test samples, `T=10`, untargeted white-box attacks, and the three model classes above. Table 1 has nine Binary budget cells; Table 2 has eleven Integer cells. The paper reports ASR only on clean-correct examples. Source: https://arxiv.org/pdf/2602.03284v1, Section 5 and Tables 1–2. The official implementation is https://github.com/yuyi-sd/Spike-Retiming-Attacks.

The paper-aligned local Binary run has 98.56% clean test accuracy (9,856/10,000), versus published 99.06% ConvNet, 99.62% ResNet18, and 99.64% VGGSNN clean accuracies. This difference is descriptive; it does not by itself prove an attack or implementation defect. The run's nine attack conditions independently passed packet, budget, prediction, and ASR audits. `B0=200` had 994/1,000 successes and the remaining eight had 1,000/1,000. These values belong to the local custom model and are not evidence that it is less robust than the three published models, because the baseline measurements were not produced under one common local evaluation.

The currently cached official repository has no N-MNIST ConvNet, ResNet18, or VGGSNN checkpoint. Its `checkpoints/` directory contains only CIFAR10-DVS ResNet18 and DVS-Gesture VGGSNN files. The local project's N-MNIST checkpoints are custom SNN/QSNN models. The official repository provides model definitions and attack evaluation but no N-MNIST training script in this checkout. Therefore the three comparison-model predictions and attacks cannot yet be independently reproduced from available checkpoint artifacts.

## Protocol for the requested published-reference comparison

1. Freeze the N-MNIST raw-data version, official train/test partition, `T=10` equal-event-count frame construction, Binary definition, labels, and hashes.
2. Train and select the custom checkpoint by a documented policy, and evaluate it on all 10,000 official test examples. Report its clean accuracy and the difference from the published models' clean accuracies.
3. Freeze the custom model's first 1,000 clean-correct examples in official test order before any attack. Reuse that manifest across all nine Binary budget cells. Published reference rows retain the paper's own clean-correct test subsets; per-sample pairing cannot be inferred from the paper's summary table.
4. Apply the untargeted white-box PIL-PGD implementation, step count, penalties, budgets, and per-example independent budget projection. Preserve amplitude-bearing cell packets, collision freedom, and all realized budgets in every record.
5. Independently reconstruct and re-predict each custom-model adversarial example. Publish denominator, numerator, ASR, run IDs, commands, and hashes. Label the three published model rows as reference values and the custom row as measured local output. Avoid claims that ASR differences are caused only by architecture.

A controlled architecture-only comparison would require evaluating or retraining all four models in one local pipeline. The user has chosen the published-reference comparison instead; no local evaluation of the three reference models is planned.

## Execution decision

`scripts/run_remaining_benchmark.ps1` defaults to `FairnessAudit` and does not launch local Integer experiments implicitly. `-Stage NmnistIntegerLocal` resumes the equal-duration local study, which is valid as a separate internal experiment but does not resolve the four-model paper comparison. The previous PowerShell failure was caused by treating a Numba stderr warning as a terminating error; the runner now records stderr and uses the Python exit code to decide failure. No Python or opencode process was active when this audit was written.

The interrupted local Integer study has eight independently audited completed conditions out of 33. Its next partial artifact, seed 42 `B0=300`, remains in `Reports/results/nmnist_integer_corrected/checkpoints/`. No completed audit was overwritten by the PowerShell warning.
