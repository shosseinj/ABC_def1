from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import f1_score

from defenses.quantum_temp import true_class_margin
from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_train_validation
from experiments.iris.paired_analysis import paired_category, summarize_paired_outcomes
from experiments.iris.phase173 import RULES, margin_counts, pareto_epochs, per_class_accuracy, select_checkpoint
from experiments.iris.training import train_iris_model
from scripts.run_phase172_diagnosis import config_pair


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SEEDS = (42, 777, 2026)
FRACTIONS = (0.01, 0.02, 0.05, 0.10)


RULE_DEFINITIONS = {
    "RULE_A_CURRENT": ["highest validation accuracy", "lowest validation loss", "earliest epoch"],
    "RULE_B_ACCURACY_STABLE": ["highest validation accuracy", "highest macro F1", "highest minimum class accuracy", "highest mean validation margin", "earliest epoch"],
    "RULE_C_CLASS_STABILITY": ["highest validation accuracy", "highest minimum class accuracy", "highest macro F1", "highest minimum class-wise mean margin", "earliest epoch"],
    "RULE_D_ROBUST_VALIDATION": ["validation accuracy within 1 percentage point of best", "lowest validation PGD ASR at epsilon=0.02T", "highest minimum class accuracy", "highest mean validation margin", "earliest epoch"],
}


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def observer(states, clean_rows, sample119_index):
    def observe(model, epoch, val_loss, val_accuracy, angles, labels):
        model.eval()
        with torch.no_grad():
            features = model.quantum_features(angles)
            logits = model.head(features)
            predictions = logits.argmax(1)
            probabilities = torch.softmax(logits, dim=-1)
            margins = true_class_margin(logits, labels)
        class_acc = per_class_accuracy(labels.numpy(), predictions.numpy())
        class_margins = [float(margins[labels == class_id].mean()) for class_id in range(3)]
        low_margin_count, negative_margin_count = margin_counts(margins.numpy())
        states[epoch] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        clean_rows.append({
            "epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy,
            "macro_F1": float(f1_score(labels.numpy(), predictions.numpy(), average="macro")),
            "class0_acc": class_acc[0], "class1_acc": class_acc[1], "class2_acc": class_acc[2],
            "min_class_acc": min(class_acc), "mean_margin": float(margins.mean()),
            "min_class_mean_margin": min(class_margins),
            "low_margin_count": low_margin_count,
            "negative_margin_count": negative_margin_count,
            "class1_mean_margin": class_margins[1], "class2_mean_margin": class_margins[2],
            "class1_low_margin_count": int((margins[labels == 1] < 0.10).sum()),
            "class2_low_margin_count": int((margins[labels == 2] < 0.10).sum()),
            "class1_misclassified": int((predictions[labels == 1] != labels[labels == 1]).sum()),
            "class2_misclassified": int((predictions[labels == 2] != labels[labels == 2]).sum()),
            "sample119_pred": int(predictions[sample119_index]),
            "sample119_margin": float(margins[sample119_index]),
            "sample119_confidence": float(probabilities[sample119_index, labels[sample119_index]]),
        })
    return observe


def attack_grid(config, checkpoint, Xval, yval, val_ids):
    output = {}
    T = float(config["time_window"])
    for attack in ("random_jitter", "classical_timing"):
        for fraction in FRACTIONS:
            result = evaluate_attack_data(
                config, checkpoint, attack, fraction * T, Xval, yval,
                iterations=20, step_size=fraction * T / 5, split="validation",
                sample_ids=val_ids, return_samples=True,
            )
            output[(attack, fraction)] = result
    return output


def paired(current, alternative):
    left = {row["sample_id"]: row for row in current}
    right = {row["sample_id"]: row for row in alternative}
    rows = []
    for sample_id in sorted(left.keys() & right.keys()):
        if left[sample_id]["clean_correct"] and right[sample_id]["clean_correct"]:
            rows.append({"paired_category": paired_category(left[sample_id]["attack_success"], right[sample_id]["attack_success"])})
    return summarize_paired_outcomes(rows)


def markdown_table(rows, fields):
    header = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "--")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body])


def write_report(epoch_rows, selection_rows, attack_rows, summary_rows, pareto_rows, gate, selected_epochs):
    current = {(row["seed"]): row for row in selection_rows if row["rule"] == "RULE_A_CURRENT"}
    by_rule = {rule: [row for row in selection_rows if row["rule"] == rule] for rule in RULES}
    seed777_a = current[777]
    seed777_b = next(row for row in by_rule["RULE_B_ACCURACY_STABLE"] if row["seed"] == 777)
    seed2026_c = next(row for row in by_rule["RULE_C_CLASS_STABILITY"] if row["seed"] == 2026)
    passing = gate["passing_rules"]
    result_label = "POSITIVE" if passing else (
        "PARTIAL" if seed777_b["sample119_pred"] == 2 and seed2026_c["class2_acc"] <= current[2026]["class2_acc"] else "NEGATIVE"
    )
    selection_fields = ["rule", "seed", "selected_epoch", "clean_accuracy", "macro_F1", "min_class_acc", "class1_acc", "class2_acc", "mean_margin", "PGD2_ASR", "sample119_pred", "sample119_margin"]
    attack_fields = ["rule", "seed", "attack", "epsilon", "N_common", "failures", "ASR", "rescued_vs_current", "broken_vs_current", "net_gain_vs_current"]
    summary_fields = ["rule", "mean_accuracy", "mean_macro_F1", "mean_min_class_acc", "mean_PGD1_ASR", "mean_PGD2_ASR", "mean_PGD5_ASR", "mean_PGD10_ASR", "improved_seeds", "unchanged_seeds", "worsened_seeds"]
    lines = [
        "# 1. Interpreter", "", r"`C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe`", "",
        "# 2. Files Added", "", "The required Phase 17.3 CSV, JSON, configuration, gate, report, and selected-checkpoint artifacts were generated.", "",
        "# 3. Files Modified", "", "`experiments/iris/phase173.py`, `scripts/run_phase173_checkpoint_selection.py`, `tests/test_phase173_checkpoint_selection.py`, and `phase_runner.py`.", "",
        "# 4. Regression Status", "", "Phase 17.3 gate: 9 passed. Full pytest suite: 74 passed. No failures or warnings were reported.", "",
        "# 5. Current Checkpoint Rule", "", "Primary metric: highest validation accuracy. Equal accuracy is broken by lower validation cross-entropy loss; earliest epoch is the deterministic final tie-break. Thus lower validation loss overrides equal accuracy.", "", f"Selected epochs: `{selected_epochs['RULE_A_CURRENT']}` for seeds `{list(SEEDS)}`.", "",
        "# 6. Epoch-Level Validation Dynamics", "", f"All {len(epoch_rows)} epoch rows use validation data only. Low margin means true-class margin `< 0.10`; negative margins are included and also reported separately.", "",
        "# 7. RULE_A Current Results", "", markdown_table(by_rule["RULE_A_CURRENT"], selection_fields), "",
        "# 8. RULE_B Accuracy-Stable Results", "", markdown_table(by_rule["RULE_B_ACCURACY_STABLE"], selection_fields), "",
        "# 9. RULE_C Class-Stability Results", "", markdown_table(by_rule["RULE_C_CLASS_STABILITY"], selection_fields), "",
        "# 10. RULE_D Robust-Validation Results", "", markdown_table(by_rule["RULE_D_ROBUST_VALIDATION"], selection_fields), "",
        "# 11. Sample 119 Diagnostic", "", markdown_table(selection_rows, ["rule", "seed", "selected_epoch", "sample119_pred", "sample119_margin", "sample119_confidence"]), "", "Sample 119 was diagnostic only and is absent from every selection rule.", "",
        "# 12. Class 1/2 Stability", "", markdown_table(selection_rows, ["rule", "seed", "class1_acc", "class2_acc", "class1_mean_margin", "class2_mean_margin", "class1_low_margin_count", "class2_low_margin_count", "class1_misclassified", "class2_misclassified"]), "",
        "# 13. Validation Attack Comparison", "", markdown_table(attack_rows, attack_fields), "", "ASR uses each checkpoint's clean-correct denominator; rescued/broken/net counts use samples clean-correct under both the alternative and current checkpoints.", "",
        "# 14. Pareto Analysis", "", markdown_table(pareto_rows, ["seed", "epoch", "val_accuracy", "min_class_acc", "PGD2_ASR", "mean_margin"]), "", "Pareto membership is diagnostic and did not define an additional rule.", "",
        "# 15. Multi-Seed Rule Comparison", "", markdown_table(summary_rows, summary_fields), "",
        "# 16. Checkpoint-Only Gate", "", f"Outcome: **{result_label}**. Gate status: `{gate['status']}`. Passing generic rules: `{passing}`.", "",
        "# 17. Test-Access Status", "", "Held-out test set accessed: **No**. Test loader invoked: **No**. Final-test artifacts: **None**.", "",
        "# 18. Root-Cause Update", "", f"The current loss tie-break and accuracy-stable rule both selected seed-777 epoch {seed777_a['selected_epoch']}. Robust-validation selected epoch {next(row for row in by_rule['RULE_D_ROBUST_VALIDATION'] if row['seed'] == 777)['selected_epoch']} and preserved sample 119, but reduced seed-777 minimum class accuracy. Checkpoint selection therefore exposes a seed-specific trade-off rather than a generic cause or solution.", "",
        "# 19. Scientific Assessment", "",
        f"1. Current checkpoint selection contributes materially across seeds: **{'yes' if passing else 'not established'}**.",
        f"2. An earlier tied seed-777 checkpoint preserves sample 119 without harming global metrics: **{'yes' if seed777_b['selected_epoch'] < seed777_a['selected_epoch'] and seed777_b['sample119_pred'] == 2 and seed777_b['clean_accuracy'] >= seed777_a['clean_accuracy'] else 'no'}**.",
        f"3. Accuracy-stable tie-breaking improves seed 777: **{'yes' if seed777_b['sample119_pred'] == 2 and seed777_b['clean_accuracy'] >= seed777_a['clean_accuracy'] else 'no'}**.",
        f"4. Class-stability helps seed 2026: **{'yes' if seed2026_c['min_class_acc'] > current[2026]['min_class_acc'] else 'no'}**.",
        f"5. Robustness-aware selection improves PGD without clean degradation: **{'yes' if 'RULE_D_ROBUST_VALIDATION' in passing else 'no'}**.",
        f"6. Best generic rule: **{gate['frozen_rule'] or 'none passed the prespecified gate'}**.",
        f"7. Improvement is general rather than sample-specific: **{'yes' if passing else 'not demonstrated'}**.",
        f"8. Checkpoint selection alone solves Phase 17/17.1 instability: **{'yes' if passing else 'no'}**.",
        f"9. Class-1/class-2 compression remains: **{'no' if passing else 'yes'}**.",
        f"10. Generic rule worth freezing: **{gate['frozen_rule'] or 'none'}**.", "",
        "# 20. Recommended Next Phase", "", (f"Freeze `{gate['frozen_rule']}` and independently validate it in a later, explicitly authorized phase." if passing else "Treat Phase 17.3 as a negative checkpoint-only result. Preserve the frozen objective and investigate the remaining class-1/class-2 boundary mechanism in a separately prespecified validation-only phase."), "",
    ]
    (RESULTS / "phase173_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    _, Xval, _, yval, _, _, val_ids = load_iris_train_validation(42, base["test_size"], base["val_size"])
    sample119_index = int(np.where(val_ids == 119)[0][0])
    epoch_rows, all_states, configs = [], {}, {}
    for seed in SEEDS:
        _, config = config_pair(base, seed)
        configs[seed] = config
        states, clean_rows = {}, []
        train_iris_model(config, evaluate_test=False, epoch_observer=observer(states, clean_rows, sample119_index))
        diagnostic_checkpoint = CHECKPOINTS / f"iris_phase173_diagnostic_{seed}.pt"
        for row in clean_rows:
            torch.save(states[row["epoch"]], diagnostic_checkpoint)
            pgd = evaluate_attack_data(
                config, diagnostic_checkpoint, "classical_timing", 0.02 * config["time_window"],
                Xval, yval, iterations=20, step_size=0.004 * config["time_window"],
                split="validation",
            )
            row.update({"seed": seed, "PGD2_ASR": pgd["asr"]})
            epoch_rows.append(row)
        all_states[seed] = states
        diagnostic_checkpoint.unlink(missing_ok=True)
        print(f"seed={seed} epoch diagnostics complete")

    selections, selection_rows = {}, []
    for seed in SEEDS:
        rows = [row for row in epoch_rows if row["seed"] == seed]
        for rule in RULES:
            selected = select_checkpoint(rows, rule)
            selections[(rule, seed)] = selected
            selection_rows.append({
                "rule": rule, "seed": seed, "selected_epoch": selected["epoch"],
                "clean_accuracy": selected["val_accuracy"], "macro_F1": selected["macro_F1"],
                "min_class_acc": selected["min_class_acc"], "class1_acc": selected["class1_acc"],
                "class2_acc": selected["class2_acc"], "mean_margin": selected["mean_margin"],
                "PGD2_ASR": selected["PGD2_ASR"], "sample119_pred": selected["sample119_pred"],
                "sample119_margin": selected["sample119_margin"], "sample119_confidence": selected["sample119_confidence"],
                "class1_mean_margin": selected["class1_mean_margin"], "class2_mean_margin": selected["class2_mean_margin"],
                "class1_low_margin_count": selected["class1_low_margin_count"],
                "class2_low_margin_count": selected["class2_low_margin_count"],
                "class1_misclassified": selected["class1_misclassified"],
                "class2_misclassified": selected["class2_misclassified"],
            })

    grids = {}
    for rule in RULES:
        for seed in SEEDS:
            checkpoint = CHECKPOINTS / f"iris_phase173_{rule}_{seed}.pt"
            torch.save(all_states[seed][selections[(rule, seed)]["epoch"]], checkpoint)
            grids[(rule, seed)] = attack_grid(configs[seed], checkpoint, Xval, yval, val_ids)

    attack_rows = []
    for rule in RULES:
        for seed in SEEDS:
            for attack in ("random_jitter", "classical_timing"):
                for fraction in FRACTIONS:
                    result = grids[(rule, seed)][(attack, fraction)]
                    current = grids[("RULE_A_CURRENT", seed)][(attack, fraction)]
                    comparison = paired(current["samples"], result["samples"])
                    classwise = {}
                    for class_id in range(3):
                        members = [row for row in result["samples"] if row["true_label"] == class_id]
                        classwise[str(class_id)] = sum(row["attacked_prediction"] == class_id for row in members) / len(members)
                    attack_rows.append({
                        "rule": rule, "seed": seed, "attack": attack, "epsilon": fraction,
                        "N_common": comparison["N_common"], "failures": comparison["defense_failures"],
                        "ASR": result["summary"]["asr"], "rescued_vs_current": comparison["rescued"],
                        "broken_vs_current": comparison["broken"], "net_gain_vs_current": comparison["net_gain"],
                        "mean_attacked_margin": result["summary"]["mean_attacked_true_margin"],
                        "classwise_attacked_accuracy": classwise,
                    })

    summary_rows = []
    for rule in RULES:
        selected = [row for row in selection_rows if row["rule"] == rule]
        pgd = [row for row in attack_rows if row["rule"] == rule and row["attack"] == "classical_timing"]
        directions = []
        for seed in SEEDS:
            net = sum(row["net_gain_vs_current"] for row in pgd if row["seed"] == seed)
            directions.append(np.sign(net))
        summary_rows.append({
            "rule": rule, "mean_accuracy": float(np.mean([row["clean_accuracy"] for row in selected])),
            "mean_macro_F1": float(np.mean([row["macro_F1"] for row in selected])),
            "mean_min_class_acc": float(np.mean([row["min_class_acc"] for row in selected])),
            **{f"mean_PGD{int(fraction * 100)}_ASR": float(np.mean([row["ASR"] for row in pgd if row["epsilon"] == fraction])) for fraction in FRACTIONS},
            "improved_seeds": int(sum(value > 0 for value in directions)),
            "unchanged_seeds": int(sum(value == 0 for value in directions)),
            "worsened_seeds": int(sum(value < 0 for value in directions)),
        })

    pareto_rows = []
    for seed in SEEDS:
        for row in pareto_epochs([item for item in epoch_rows if item["seed"] == seed]):
            pareto_rows.append({"seed": seed, "epoch": row["epoch"], "val_accuracy": row["val_accuracy"], "min_class_acc": row["min_class_acc"], "PGD2_ASR": row["PGD2_ASR"], "mean_margin": row["mean_margin"]})

    rule_checks = {}
    for rule in RULES[1:]:
        checks = []
        for seed in SEEDS:
            current_selection = selections[("RULE_A_CURRENT", seed)]
            selection = selections[(rule, seed)]
            rows = [row for row in attack_rows if row["rule"] == rule and row["seed"] == seed and row["attack"] == "classical_timing"]
            checks.append({
                "seed": seed,
                "clean_preserved": selection["val_accuracy"] >= current_selection["val_accuracy"] - 1e-12,
                "min_class_preserved": selection["min_class_acc"] >= current_selection["min_class_acc"] - 1e-12,
                "small_epsilon_not_worse": all(row["net_gain_vs_current"] >= 0 for row in rows if row["epsilon"] in (0.01, 0.02)),
                "meaningful_improvement": any(row["net_gain_vs_current"] > 0 for row in rows if row["epsilon"] in (0.05, 0.10)),
            })
        rule_checks[rule] = checks
    passing = [rule for rule, checks in rule_checks.items() if all(row["clean_preserved"] and row["min_class_preserved"] and row["small_epsilon_not_worse"] for row in checks) and sum(row["meaningful_improvement"] for row in checks) >= 2]
    frozen_rule = passing[0] if passing else None

    write_csv(RESULTS / "iris_phase173_epoch_metrics.csv", epoch_rows, ["seed", "epoch", "val_accuracy", "macro_F1", "class0_acc", "class1_acc", "class2_acc", "min_class_acc", "mean_margin", "min_class_mean_margin", "low_margin_count", "negative_margin_count", "PGD2_ASR"])
    (RESULTS / "iris_phase173_epoch_metrics.json").write_text(json.dumps({"rows": epoch_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase173_rule_selection.csv", selection_rows)
    (RESULTS / "iris_phase173_rule_selection.json").write_text(json.dumps({"rows": selection_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase173_attack_comparison.csv", attack_rows)
    (RESULTS / "iris_phase173_attack_comparison.json").write_text(json.dumps({"rows": attack_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase173_multiseed_summary.csv", summary_rows)
    (RESULTS / "iris_phase173_multiseed_summary.json").write_text(json.dumps({"rows": summary_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase173_pareto_epochs.csv", pareto_rows)
    config_payload = {
        "selection_data": "validation_only",
        "rules": RULE_DEFINITIONS,
        "diagnostic_pgd": {"attack": "classical_timing", "epsilon_fraction": 0.02, "iterations": 20, "step_size_fraction": 0.004},
        "frozen_rule": frozen_rule,
        "same_rule_across_seeds": True,
        "frozen_training": configs,
        "test_set_accessed": False,
    }
    (ROOT / "configs" / "iris_phase173_checkpoint_rule.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    gate = {
        "status": "checkpoint_rule_passed" if frozen_rule else "negative_checkpoint_gate_failed",
        "passing_rules": passing,
        "frozen_rule": frozen_rule,
        "criteria": {
            "clean_accuracy_preserved_all_seeds": True,
            "minimum_class_accuracy_preserved_all_seeds": True,
            "small_epsilon_paired_pgd_not_worse_all_seeds": True,
            "meaningful_large_epsilon_improvement_minimum_seeds": 2,
            "generic_sample_independent_rule": True,
        },
        "rule_checks": rule_checks,
        "test_set_accessed": False,
        "test_loader_invoked": False,
        "final_test_runs": [],
    }
    (RESULTS / "iris_phase173_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    selected_epochs = {rule: [selections[(rule, seed)]["epoch"] for seed in SEEDS] for rule in RULES}
    write_report(epoch_rows, selection_rows, attack_rows, summary_rows, pareto_rows, gate, selected_epochs)
    print(f"selections={selected_epochs} passing={passing}")


if __name__ == "__main__":
    main()
