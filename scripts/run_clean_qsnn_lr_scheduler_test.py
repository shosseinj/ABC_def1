"""Validation-only test of ReduceLROnPlateau in 240-epoch clean QSNN training."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_train_validation
from experiments.iris.training import set_seed, to_theta
from models.qsnn import IrisQSNN
from scripts.run_clean_qsnn_160_epoch_test import validation_metrics

SEEDS = (42, 123, 777, 2026, 6543)
SPLIT_SEED = 42


def train_with_scheduler(config, xtrain, ytrain, xval, yval, checkpoint):
    set_seed(int(config["seed"]))
    model = IrisQSNN(4, 4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-4
    )
    tx, vx = to_theta(xtrain, config["time_window"]), to_theta(xval, config["time_window"])
    ty, vy = torch.tensor(ytrain, dtype=torch.long), torch.tensor(yval, dtype=torch.long)
    best_key = None; best_state = None; best_lr = None; best_reductions = None
    reductions = 0; history = []
    for epoch in range(1, 241):
        model.train(); optimizer.zero_grad()
        loss = F.cross_entropy(model(tx), ty); loss.backward(); optimizer.step()
        model.eval()
        lr_at_evaluation = float(optimizer.param_groups[0]["lr"])
        with torch.no_grad():
            logits = model(vx)
            val_loss = float(F.cross_entropy(logits, vy))
            val_accuracy = float((logits.argmax(1) == vy).float().mean())
        key = (-val_accuracy, val_loss, epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_lr = lr_at_evaluation; best_reductions = reductions
        scheduler.step(val_loss)
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr < lr_at_evaluation:
            reductions += 1
        history.append({
            "seed": config["seed"], "epoch": epoch, "training_loss": float(loss.detach()),
            "validation_loss": val_loss, "validation_accuracy": val_accuracy,
            "learning_rate": lr_at_evaluation, "lr_reductions": reductions,
        })
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint)
    return model, {
        "best_epoch": best_key[2], "best_validation_accuracy": -best_key[0],
        "best_validation_loss": best_key[1], "selected_lr": best_lr,
        "reductions_at_selected_epoch": best_reductions,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "total_lr_reductions": reductions, "history": history,
    }


def aggregate(rows, prefix):
    values = np.asarray([row[f"{prefix}_validation_accuracy"] for row in rows])
    return {
        "accuracy_mean": float(values.mean()), "accuracy_sample_sd": float(values.std(ddof=1)),
        "minimum_seed_accuracy": float(values.min()),
        "class1_accuracy_mean": float(np.mean([row[f"{prefix}_class_accuracy"]["1"] for row in rows])),
    }


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results" / "clean_qsnn_240_epoch_test.json").read_text(encoding="utf-8"))
    before = {row["seed"]: row for row in prior["rows"]}
    if tuple(before) != SEEDS or prior["only_change"]["max_epochs_after"] != 240:
        raise RuntimeError("Frozen fixed-LR 240-epoch comparator does not match.")
    if float(base["learning_rate"]) != 0.01 or any((base.get("quantum_temp_enabled", False),
            base.get("adversarial_training_enabled", False), base.get("consistency_enabled", False))):
        raise RuntimeError("Frozen optimizer or clean-training protocol changed.")
    xtrain, xval, ytrain, yval, _, _, _ = load_iris_train_validation(
        SPLIT_SEED, base["test_size"], base["val_size"]
    )
    rows, histories = [], []
    for seed in SEEDS:
        config = {**base, "seed": seed, "split_seed": SPLIT_SEED, "epochs": 240}
        checkpoint = ROOT / "checkpoints" / f"iris_clean_scheduler_seed_{seed}.pt"
        model, training = train_with_scheduler(config, xtrain, ytrain, xval, yval, checkpoint)
        measured = validation_metrics(model, xval, yval, base["time_window"])
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
            "best_epoch": training["best_epoch"], "selected_lr": training["selected_lr"],
            "final_lr": training["final_lr"], "lr_reductions": training["total_lr_reductions"],
            "reductions_at_selected_epoch": training["reductions_at_selected_epoch"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "impact": "IMPROVED" if delta > 1e-12 else "WORSE" if delta < -1e-12 else "UNCHANGED",
        }
        rows.append(row); histories.extend(training["history"])
        print(f"seed={seed} best_epoch={row['best_epoch']} acc={measured['accuracy']:.4f} reductions={row['lr_reductions']}")

    before_summary, after_summary = aggregate(rows, "before"), aggregate(rows, "after")
    strong_seeds = (42, 777, 2026)
    no_strong_regression = all(next(row for row in rows if row["seed"] == seed)["after_validation_accuracy"]
                               >= before[seed]["after_validation_accuracy"] for seed in strong_seeds)
    beneficial = (after_summary["accuracy_mean"] > before_summary["accuracy_mean"]
                  and after_summary["minimum_seed_accuracy"] >= before_summary["minimum_seed_accuracy"]
                  and no_strong_regression)
    changed = [row["seed"] for row in rows if row["impact"] != "UNCHANGED"]
    output = {
        "validation_only": True, "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "only_change": "ReduceLROnPlateau on validation CE", "scheduler": {
            "factor": 0.5, "patience": 20, "minimum_lr": 1e-4, "initial_lr": 0.01,
        }, "max_epochs": 240, "split_seed": SPLIT_SEED, "model_seeds": SEEDS,
        "checkpoint_rule": "highest validation accuracy, then lowest validation CE, then earliest epoch",
        "rows": rows, "before_summary": before_summary, "after_summary": after_summary,
        "changed_seeds": changed, "improvement_concentrated_in_slow_seeds": changed == [123, 6543],
        "conclusion": "BENEFICIAL" if beneficial else "NOT BENEFICIAL",
    }
    results = ROOT / "results"
    (results / "clean_qsnn_lr_scheduler_test.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (results / "clean_qsnn_lr_scheduler_test.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "before_class_accuracy": json.dumps(row["before_class_accuracy"]),
                 "after_class_accuracy": json.dumps(row["after_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (results / "clean_qsnn_lr_scheduler_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)
    print("\n| Seed | Fixed-LR Acc | Scheduler Acc | Class-1 Before | Class-1 After | Best Epoch | LR Reductions | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(f"| {row['seed']} | {row['before_validation_accuracy']:.4f} | {row['after_validation_accuracy']:.4f} | {row['before_class_accuracy']['1']:.4f} | {row['after_class_accuracy']['1']:.4f} | {row['best_epoch']} | {row['lr_reductions']} | {row['impact']} |")
    print(f"\nLR SCHEDULER: {output['conclusion']}")


if __name__ == "__main__":
    main()
