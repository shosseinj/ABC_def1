# Verified experimental results and reporting boundary

## Controlled four-model local comparison, seed 42
Values are from `Reports/nmnist_controlled_four_models_lr1e4_comparison.md`, the eight clean-result JSON files and sixteen attack JSON/audit files under `Reports/results/nmnist_controlled_four_models_lr1e4/`. Each attack cell has a 1,000-sample joint clean-correct denominator; only `B_inf=1` and `B0=200` were completed for this campaign. Results are **local, common-recipe comparisons**, not numbers transcribed from the reference paper. The saved comparison states independent audit PASS for each cell; this read-only audit inspected representative audit files, not a fresh execution of all 16 auditors.

| Representation | Model | Official-test clean accuracy | ASR `B_inf=1` | ASR `B0=200` |
|---|---|---:|---:|---:|
| Binary | Custom SNN | 97.99% | 100.0% | 99.5% |
| Binary | ConvNet | 96.90% | 1.9% | 1.2% |
| Binary | Spiking ResNet18 | 98.61% | 0.1% | 0.0% |
| Binary | VGGSNN | 99.21% | 0.0% | 0.0% |
| Integer | Custom SNN | 98.14% | 99.9% | 98.0% |
| Integer | ConvNet | 96.58% | 0.6% | 1.1% |
| Integer | Spiking ResNet18 | 98.12% | 0.2% | 0.1% |
| Integer | VGGSNN | 99.30% | 0.0% | 0.0% |

The striking attack gap is **observed**, but its cause is NOT VERIFIED. Same attack source, preprocessing, budgets, sample IDs and training hyperparameters improve fairness; they do not make architectures identical or prove that optimization is equally strong against every model. The source/paper B0 penalty coefficient discrepancy (5 in upstream implementation, 10 in paper general description) remains unresolved; see `Reports/nmnist_controlled_four_models_lr1e4_comparison.md:27`.

## Earlier local custom-SNN paper-aligned tables
`readme_jafar.md:134-159` reports Binary clean 98.56%, `B_inf=1` 100.0%, `B0=200` 99.4%; Integer clean 98.60%, `B_inf=1` 100.0%, `B0=200` 94.5%. Those are separate checkpoints/training and sample manifests. Do not replace the controlled custom rows with these values or mix them into a common-recipe ranking. Other budget cells in Table 1/2 belong only to these earlier local experiments and published reference rows; controlled four-model evidence does not cover them.

## QSNN-v3 and timestamp attacks
Five-seed QSNN-v3 validation accuracy is 97.272% ± 0.212% and macro-F1 97.274% ± 0.212%; the five-seed SNN validation reference accuracy is 98.132% ± 0.241%. These are **validation** numbers, not official-test results; evidence: `readme_jafar.md:175-186` and `results/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/summary.json`.

The seed-42, 100-sample access-matched comparison reports QSNN PGD/TEMP ASR 3%/4% at epsilon 5% and 3%/3% at epsilon 10%; the SNN has 0%/0% at both epsilons. The runtime-matched QSNN results are 3%/4% at 5% and 3%/5% at 10%; SNN 0%/0% and 0%/1%. Source: `readme_jafar.md:251-269`, `results/nmnist_budget_matched_comparison_v2_seed42/condition_summaries.csv`; `results/nmnist_budget_matched_comparison_v2_seed42/audit.json` reports passed with 3,200 records checked, including 2,400 source records. Differences were reported as nonsignificant; no superiority claim is supported (`readme_jafar.md:282-286`).

The canonical 2,400-record audit is complete (`results/nmnist_attack_protocol_v3_auditable_seed42/STATUS.json`). The budget-matched extension states 1,600 new records, zero failures (`results/nmnist_budget_matched_comparison_v2_seed42/audit.json`). These counts concern the **timestamp** study; they do not validate the separate grid-cell campaign.
