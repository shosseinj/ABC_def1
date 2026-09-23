"""Final validation-only comparison and freeze of 380 versus 600 clean epochs."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_train_validation
from experiments.iris.training import train_iris_model
from scripts.run_clean_qsnn_160_epoch_test import validation_metrics
from scripts.run_clean_qsnn_240_epoch_test import aggregate

SEEDS = (42, 123, 777, 2026, 6543)
STRONG_SEEDS = (42, 777, 2026)
SPLIT_SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results" / "clean_qsnn_380_epoch_test.json").read_text(encoding="utf-8"))
    before = {row["seed"]: row for row in prior["rows"]}
    if tuple(before) != SEEDS or prior["only_change"]["max_epochs_after"] != 380:
        raise RuntimeError("Frozen 380-epoch comparator does not match.")
    if float(base["learning_rate"]) != 0.01 or any((base.get("quantum_temp_enabled", False),
            base.get("adversarial_training_enabled", False), base.get("consistency_enabled", False))):
        raise RuntimeError("Frozen fixed-LR clean-training protocol changed.")
    _, validation, _, labels, _, _, _ = load_iris_train_validation(
        SPLIT_SEED, base["test_size"], base["val_size"]
    )
    rows, histories = [], []
    for seed in SEEDS:
        # train_iris_model constructs a new IrisQSNN and Adam optimizer after reseeding.
        config = {**base, "seed": seed, "split_seed": SPLIT_SEED, "epochs": 600}
        checkpoint = ROOT / "checkpoints" / f"iris_clean_600epoch_seed_{seed}.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
        if run["test_evaluated"] or run["predictions"] is not None or run["targets"] is not None:
            raise RuntimeError("Held-out data was exposed.")
        measured = validation_metrics(run["model"], validation, labels, base["time_window"])
        old = before[seed]; delta = measured["accuracy"] - old["after_validation_accuracy"]
        row = {
            "seed": seed, "before_validation_accuracy": old["after_validation_accuracy"],
            "after_validation_accuracy": measured["accuracy"],
            "before_validation_macro_f1": old["after_validation_macro_f1"],
            "after_validation_macro_f1": measured["macro_f1"],
            "before_class_accuracy": old["after_class_accuracy"],
            "after_class_accuracy": measured["class_accuracy"],
            "before_validation_ce": old["after_validation_ce"],
            "after_validation_ce": measured["cross_entropy"],
            "best_epoch": int(run["metrics"]["best_epoch"]),
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
            "impact": "IMPROVED" if delta > 1e-12 else "WORSE" if delta < -1e-12 else "UNCHANGED",
        }
        rows.append(row)
        histories.extend({"seed": seed, "epoch": epoch, "training_loss": train_loss,
                          "validation_loss": val_loss, "validation_accuracy": val_accuracy}
                         for epoch, train_loss, val_loss, val_accuracy in run["history"])
        print(f"seed={seed} best_epoch={row['best_epoch']} val_acc={measured['accuracy']:.4f}")

    before_summary, after_summary = aggregate(rows, "before"), aggregate(rows, "after")
    strong_worse = [row["seed"] for row in rows
                    if row["seed"] in STRONG_SEEDS and row["after_validation_accuracy"] < row["before_validation_accuracy"]]
    beneficial = (after_summary["accuracy_mean"] > before_summary["accuracy_mean"]
                  and after_summary["minimum_seed_accuracy"] >= before_summary["minimum_seed_accuracy"]
                  and after_summary["class1_accuracy_mean"] >= before_summary["class1_accuracy_mean"]
                  and not strong_worse)
    chosen_epochs = 600 if beneficial else 380
    selected = []
    for row in rows:
        if beneficial:
            path = ROOT / row["checkpoint"]
            selected.append({"seed": row["seed"], "best_epoch": row["best_epoch"],
                             "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"]})
        else:
            old = before[row["seed"]]; path = ROOT / old["checkpoint"]
            selected.append({"seed": row["seed"], "best_epoch": old["best_epoch"],
                             "checkpoint": old["checkpoint"], "checkpoint_sha256": sha256(path)})
    slow_at_final = [seed for seed in (123, 6543)
                     if next(row for row in rows if row["seed"] == seed)["best_epoch"] == 600]
    output = {
        "validation_only": True, "held_out_accessed": False, "attacks_run": False,
        "defenses_run": False, "scheduler_used": False, "resumed_from_checkpoint": False,
        "previous_weights_loaded": False, "fresh_model_and_optimizer_per_seed": True,
        "only_change": {"max_epochs_before": 380, "max_epochs_after": 600},
        "learning_rate": 0.01, "split_seed": SPLIT_SEED, "model_seeds": SEEDS,
        "checkpoint_rule": "highest validation accuracy, then lowest validation CE, then earliest epoch",
        "rows": rows, "before_summary": before_summary, "after_summary": after_summary,
        "slow_seeds_selecting_epoch_600": slow_at_final, "strong_seeds_worse": strong_worse,
        "conclusion": "BENEFICIAL" if beneficial else "NOT BENEFICIAL",
        "frozen_protocol": {"max_epochs": chosen_epochs, "status": "FROZEN", "selected_checkpoints": selected},
    }
    results = ROOT / "results"
    (results / "clean_qsnn_600_epoch_final.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (results / "final_clean_protocol.json").write_text(json.dumps(output["frozen_protocol"], indent=2), encoding="utf-8")
    with (results / "clean_qsnn_600_epoch_final.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "before_class_accuracy": json.dumps(row["before_class_accuracy"]),
                 "after_class_accuracy": json.dumps(row["after_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (results / "clean_qsnn_600_epoch_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)
    print("\n| Seed | 380-Epoch Acc | 600-Epoch Acc | Class-1 Before | Class-1 After | Best Epoch | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row['seed']} | {row['before_validation_accuracy']:.4f} | {row['after_validation_accuracy']:.4f} | {row['before_class_accuracy']['1']:.4f} | {row['after_class_accuracy']['1']:.4f} | {row['best_epoch']} | {row['impact']} |")
    print(f"\n600-EPOCH TRAINING: {output['conclusion']}")
    print("FINAL CLEAN PROTOCOL: FROZEN")


if __name__ == "__main__":
    main()
