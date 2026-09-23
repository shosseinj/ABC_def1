"""Validation-only test of changing clean QSNN max epochs from 160 to 240."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_train_validation
from experiments.iris.training import train_iris_model
from scripts.run_clean_qsnn_160_epoch_test import validation_metrics

SEEDS = (42, 123, 777, 2026, 6543)
SPLIT_SEED = 42


def aggregate(rows, prefix):
    accuracy = np.asarray([row[f"{prefix}_validation_accuracy"] for row in rows])
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_sample_sd": float(accuracy.std(ddof=1)),
        "minimum_seed_accuracy": float(accuracy.min()),
        "class1_accuracy_mean": float(np.mean([row[f"{prefix}_class_accuracy"]["1"] for row in rows])),
    }


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results" / "clean_qsnn_160_epoch_test.json").read_text(encoding="utf-8"))
    before = {row["seed"]: row for row in prior["rows"]}
    if tuple(before) != SEEDS or prior["only_change"]["max_epochs_after"] != 160:
        raise RuntimeError("Frozen 160-epoch comparator does not match this experiment.")
    if any((base.get("quantum_temp_enabled", False), base.get("adversarial_training_enabled", False),
            base.get("consistency_enabled", False))):
        raise RuntimeError("Attack/defense training must remain disabled.")
    _, validation, _, labels, _, _, _ = load_iris_train_validation(
        SPLIT_SEED, base["test_size"], base["val_size"]
    )
    rows, histories = [], []
    for seed in SEEDS:
        config = {**base, "seed": seed, "split_seed": SPLIT_SEED, "epochs": 240}
        checkpoint = ROOT / "checkpoints" / f"iris_clean_240epoch_seed_{seed}.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
        if run["test_evaluated"] or run["predictions"] is not None or run["targets"] is not None:
            raise RuntimeError("Held-out data was exposed.")
        measured = validation_metrics(run["model"], validation, labels, base["time_window"])
        old = before[seed]
        delta = measured["accuracy"] - old["after_validation_accuracy"]
        row = {
            "seed": seed,
            "before_validation_accuracy": old["after_validation_accuracy"],
            "after_validation_accuracy": measured["accuracy"],
            "before_validation_macro_f1": old["after_validation_macro_f1"],
            "after_validation_macro_f1": measured["macro_f1"],
            "before_class_accuracy": old["after_class_accuracy"],
            "after_class_accuracy": measured["class_accuracy"],
            "before_validation_ce": old["after_validation_ce"],
            "after_validation_ce": measured["cross_entropy"],
            "best_epoch": int(run["metrics"]["best_epoch"]),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "impact": "IMPROVED" if delta > 1e-12 else "WORSE" if delta < -1e-12 else "UNCHANGED",
        }
        rows.append(row)
        histories.extend({"seed": seed, "epoch": epoch, "training_loss": train_loss,
                          "validation_loss": val_loss, "validation_accuracy": val_accuracy}
                         for epoch, train_loss, val_loss, val_accuracy in run["history"])
        print(f"seed={seed} best_epoch={row['best_epoch']} val_acc={measured['accuracy']:.4f}")

    before_summary, after_summary = aggregate(rows, "before"), aggregate(rows, "after")
    beneficial = (after_summary["accuracy_mean"] > before_summary["accuracy_mean"]
                  and after_summary["minimum_seed_accuracy"] >= before_summary["minimum_seed_accuracy"]
                  and after_summary["class1_accuracy_mean"] > before_summary["class1_accuracy_mean"])
    slow_seeds_at_limit = [seed for seed in (123, 6543)
                           if next(row for row in rows if row["seed"] == seed)["best_epoch"] == 240]
    output = {
        "validation_only": True, "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "only_change": {"max_epochs_before": 160, "max_epochs_after": 240},
        "split_seed": SPLIT_SEED, "model_seeds": SEEDS,
        "checkpoint_rule": "highest validation accuracy, then lowest validation CE, then earliest epoch",
        "rows": rows, "before_summary": before_summary, "after_summary": after_summary,
        "slow_seeds_selecting_epoch_240": slow_seeds_at_limit,
        "epoch_count_alone_not_sufficient": bool(slow_seeds_at_limit),
        "conclusion": "BENEFICIAL" if beneficial else "NOT BENEFICIAL",
    }
    results = ROOT / "results"
    (results / "clean_qsnn_240_epoch_test.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (results / "clean_qsnn_240_epoch_test.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "before_class_accuracy": json.dumps(row["before_class_accuracy"]),
                 "after_class_accuracy": json.dumps(row["after_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (results / "clean_qsnn_240_epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)

    print("\n| Seed | 160-Epoch Val Acc | 240-Epoch Val Acc | Class-1 Before | Class-1 After | Best Epoch | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row['seed']} | {row['before_validation_accuracy']:.4f} | {row['after_validation_accuracy']:.4f} | {row['before_class_accuracy']['1']:.4f} | {row['after_class_accuracy']['1']:.4f} | {row['best_epoch']} | {row['impact']} |")
    print(f"\n240-EPOCH TRAINING: {output['conclusion']}")
    if slow_seeds_at_limit:
        print("EPOCH COUNT ALONE IS NOT SUFFICIENT")


if __name__ == "__main__":
    main()
