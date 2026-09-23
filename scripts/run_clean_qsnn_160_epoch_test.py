"""Validation-only test of changing clean QSNN max epochs from 80 to 160."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_train_validation
from experiments.iris.training import to_theta, train_iris_model

SEEDS = (42, 123, 777, 2026, 6543)
SPLIT_SEED = 42


def validation_metrics(model, features, labels, time_window):
    labels = np.asarray(labels, dtype=int)
    with torch.no_grad():
        logits = model(to_theta(features, time_window))
        predictions = logits.argmax(1).numpy()
        ce = float(F.cross_entropy(logits, torch.tensor(labels, dtype=torch.long)))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "class_accuracy": {
            str(label): float(np.mean(predictions[labels == label] == label))
            for label in range(3)
        },
        "cross_entropy": ce,
    }


def summary(rows, prefix):
    accuracy = np.asarray([row[f"{prefix}_validation_accuracy"] for row in rows])
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_sample_sd": float(accuracy.std(ddof=1)),
        "minimum_seed_accuracy": float(accuracy.min()),
        "class1_accuracy_mean": float(np.mean([
            row[f"{prefix}_class_accuracy"]["1"] for row in rows
        ])),
    }


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "results" / "clean_qsnn_multiseed_baseline.json").read_text(encoding="utf-8"))
    before = {row["seed"]: row for row in previous["rows"]}
    if tuple(before) != SEEDS or int(base["epochs"]) != 80:
        raise RuntimeError("Frozen 80-epoch baseline does not match this experiment.")
    if any((base.get("quantum_temp_enabled", False),
            base.get("adversarial_training_enabled", False),
            base.get("consistency_enabled", False))):
        raise RuntimeError("Attack/defense training must remain disabled.")

    _, validation, _, labels, _, _, _ = load_iris_train_validation(
        SPLIT_SEED, base["test_size"], base["val_size"]
    )
    rows, histories = [], []
    for seed in SEEDS:
        config = {**base, "seed": seed, "split_seed": SPLIT_SEED, "epochs": 160}
        checkpoint = ROOT / "checkpoints" / f"iris_clean_160epoch_seed_{seed}.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
        if run["test_evaluated"] or run["predictions"] is not None or run["targets"] is not None:
            raise RuntimeError("Held-out data was exposed.")
        measured = validation_metrics(run["model"], validation, labels, base["time_window"])
        old = before[seed]
        delta = measured["accuracy"] - old["validation_accuracy"]
        row = {
            "seed": seed,
            "before_validation_accuracy": old["validation_accuracy"],
            "after_validation_accuracy": measured["accuracy"],
            "before_validation_macro_f1": old["validation_macro_f1"],
            "after_validation_macro_f1": measured["macro_f1"],
            "before_class_accuracy": old["validation_class_accuracy"],
            "after_class_accuracy": measured["class_accuracy"],
            "before_validation_ce": old["validation_cross_entropy"],
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

    before_summary, after_summary = summary(rows, "before"), summary(rows, "after")
    beneficial = (
        after_summary["accuracy_mean"] > before_summary["accuracy_mean"]
        and after_summary["minimum_seed_accuracy"] >= before_summary["minimum_seed_accuracy"]
        and after_summary["class1_accuracy_mean"] > before_summary["class1_accuracy_mean"]
    )
    output = {
        "validation_only": True, "held_out_accessed": False,
        "attacks_run": False, "defenses_run": False,
        "only_change": {"max_epochs_before": 80, "max_epochs_after": 160},
        "split_seed": SPLIT_SEED, "model_seeds": SEEDS,
        "checkpoint_rule": "highest validation accuracy, then lowest validation CE, then earliest epoch",
        "rows": rows, "before_summary": before_summary, "after_summary": after_summary,
        "conclusion": "BENEFICIAL" if beneficial else "NOT BENEFICIAL",
    }
    results = ROOT / "results"
    (results / "clean_qsnn_160_epoch_test.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (results / "clean_qsnn_160_epoch_test.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "before_class_accuracy": json.dumps(row["before_class_accuracy"]),
                 "after_class_accuracy": json.dumps(row["after_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (results / "clean_qsnn_160_epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)

    print("\n| Seed | 80-Epoch Val Acc | 160-Epoch Val Acc | Class-1 Acc Before | Class-1 Acc After | Best Epoch | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row['seed']} | {row['before_validation_accuracy']:.4f} | {row['after_validation_accuracy']:.4f} | {row['before_class_accuracy']['1']:.4f} | {row['after_class_accuracy']['1']:.4f} | {row['best_epoch']} | {row['impact']} |")
    print(f"\n160-EPOCH TRAINING: {output['conclusion']}")


if __name__ == "__main__":
    main()
