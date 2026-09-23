"""Train, freeze, then clean-evaluate the five-seed QSNN baseline."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_splits, load_iris_train_validation
from experiments.iris.training import to_theta, train_iris_model
from models.qsnn import IrisQSNN


SEEDS = (42, 123, 777, 2026, 6543)
SPLIT_SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_metrics(model, features, labels, T):
    targets = np.asarray(labels, dtype=int)
    with torch.no_grad():
        logits = model(to_theta(features, T))
        predictions = logits.argmax(1).numpy()
        loss = float(F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long)))
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "class_accuracy": {
            str(label): float(np.mean(predictions[targets == label] == label))
            for label in range(3)
        },
        "cross_entropy": loss,
    }


def mean_sample_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    if any((base.get("quantum_temp_enabled", False),
            base.get("adversarial_training_enabled", False),
            base.get("consistency_enabled", False))):
        raise RuntimeError("Clean baseline requires every defense/attack training flag disabled.")
    expected = {"n_qubits": 4, "n_layers": 4, "epochs": 80, "learning_rate": 0.01}
    if any(base[key] != value for key, value in expected.items()):
        raise RuntimeError("Frozen clean-training configuration does not match the existing protocol.")

    _, validation, _, validation_labels, _, train_ids, validation_ids = load_iris_train_validation(
        seed=SPLIT_SEED, test_size=float(base["test_size"]), val_size=float(base["val_size"])
    )
    T = float(base["time_window"])
    rows = []
    histories = []
    checkpoints = []
    for seed in SEEDS:
        config = {**base, "seed": seed, "split_seed": SPLIT_SEED,
                  "quantum_temp_enabled": False,
                  "adversarial_training_enabled": False,
                  "consistency_enabled": False}
        checkpoint = ROOT / "checkpoints" / f"iris_clean_frozen_seed_{seed}.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
        if run["test_evaluated"] or run["predictions"] is not None or run["targets"] is not None:
            raise RuntimeError("Held-out data was exposed during training.")
        if run["metrics"]["trainable_parameters"] != 47:
            raise RuntimeError("QSNN trainable-parameter count changed.")
        validation_result = clean_metrics(run["model"], validation, validation_labels, T)
        row = {
            "seed": seed,
            "best_epoch": int(run["metrics"]["best_epoch"]),
            "validation_accuracy": validation_result["accuracy"],
            "validation_macro_f1": validation_result["macro_f1"],
            "validation_class_accuracy": validation_result["class_accuracy"],
            "validation_cross_entropy": validation_result["cross_entropy"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "held_out_accuracy": None,
            "held_out_macro_f1": None,
            "held_out_class_accuracy": None,
            "held_out_cross_entropy": None,
        }
        rows.append(row)
        checkpoints.append({key: row[key] for key in (
            "seed", "best_epoch", "checkpoint", "checkpoint_sha256"
        )})
        histories.extend({
            "seed": seed, "epoch": epoch, "training_loss": training_loss,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
        } for epoch, training_loss, validation_loss, validation_accuracy in run["history"])
        print(f"trained seed={seed} best_epoch={row['best_epoch']} val_acc={row['validation_accuracy']:.4f}")

    validation_summary = {
        "accuracy": mean_sample_sd([row["validation_accuracy"] for row in rows]),
        "macro_f1": mean_sample_sd([row["validation_macro_f1"] for row in rows]),
        "class_accuracy_mean": {
            str(label): float(np.mean([row["validation_class_accuracy"][str(label)] for row in rows]))
            for label in range(3)
        },
        "minimum_seed_accuracy": float(min(row["validation_accuracy"] for row in rows)),
        "maximum_seed_accuracy": float(max(row["validation_accuracy"] for row in rows)),
    }
    results = ROOT / "results"
    frozen_protocol = {
        "status": "frozen_before_held_out_evaluation",
        "dataset": "iris", "split_seed": SPLIT_SEED, "model_seeds": SEEDS,
        "split_sizes": {"train": int(len(train_ids)), "validation": int(len(validation_ids)), "held_out": 30},
        "architecture": {"n_qubits": 4, "n_layers": 4, "n_classes": 3, "trainable_parameters": 47},
        "preprocessing": "MinMaxScaler fit on training only, clipped to [0,1]",
        "encoding": "existing TTFS then angle encoding", "optimizer": "Adam",
        "learning_rate": float(base["learning_rate"]), "epochs": int(base["epochs"]),
        "training_objective": "clean full-batch cross entropy on training data only",
        "checkpoint_rule": "highest validation accuracy, then lowest validation CE, then earliest epoch",
        "attacks_run": False, "defenses_run": False, "held_out_access_before_freeze": False,
        "checkpoints": checkpoints, "validation_summary": validation_summary,
        "source_sha256": {
            "training": sha256(ROOT / "experiments" / "iris" / "training.py"),
            "data": sha256(ROOT / "experiments" / "iris" / "data.py"),
            "model": sha256(ROOT / "models" / "qsnn.py"),
            "ttfs": sha256(ROOT / "encoding" / "ttfs.py"),
            "config": sha256(ROOT / "configs" / "iris.json"),
        },
    }
    protocol_path = results / "clean_qsnn_frozen_protocol.json"
    protocol_path.write_text(json.dumps(frozen_protocol, indent=2), encoding="utf-8")

    # This is the sole held-out feature access, after protocol/checkpoint freezing.
    _, _, held_out, _, _, held_out_labels, _ = load_iris_splits(
        seed=SPLIT_SEED, test_size=float(base["test_size"]), val_size=float(base["val_size"])
    )
    for row in rows:
        config = {**base, "seed": row["seed"], "split_seed": SPLIT_SEED}
        model = IrisQSNN(
            int(config["n_qubits"]), int(config["n_layers"]), int(config["n_classes"])
        )
        model.load_state_dict(torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=True))
        model.eval()
        measured = clean_metrics(model, held_out, held_out_labels, T)
        row["held_out_accuracy"] = measured["accuracy"]
        row["held_out_macro_f1"] = measured["macro_f1"]
        row["held_out_class_accuracy"] = measured["class_accuracy"]
        row["held_out_cross_entropy"] = measured["cross_entropy"]

    held_out_summary = {
        "accuracy": mean_sample_sd([row["held_out_accuracy"] for row in rows]),
        "macro_f1": mean_sample_sd([row["held_out_macro_f1"] for row in rows]),
        "class_accuracy_mean": {
            str(label): float(np.mean([row["held_out_class_accuracy"][str(label)] for row in rows]))
            for label in range(3)
        },
    }
    mean_accuracy = held_out_summary["accuracy"]["mean"]
    classification = "STRONG" if mean_accuracy >= 0.95 else "ACCEPTABLE" if mean_accuracy >= 0.90 else "WEAK"
    final = {
        **frozen_protocol, "status": "frozen_and_evaluated_once_on_held_out",
        "held_out_loader_calls_after_freeze": 1, "rows": rows,
        "held_out_summary": held_out_summary, "classification": classification,
    }
    (results / "clean_qsnn_multiseed_baseline.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    with (results / "clean_qsnn_multiseed_baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row,
                 "validation_class_accuracy": json.dumps(row["validation_class_accuracy"]),
                 "held_out_class_accuracy": json.dumps(row["held_out_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (results / "clean_qsnn_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)

    print("\n| Seed | Best Epoch | Val Accuracy | Val Macro-F1 | Held-out Accuracy | Held-out Macro-F1 |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['seed']} | {row['best_epoch']} | {row['validation_accuracy']:.4f} | {row['validation_macro_f1']:.4f} | {row['held_out_accuracy']:.4f} | {row['held_out_macro_f1']:.4f} |")
    print(f"\nMEAN CLEAN ACCURACY = {mean_accuracy:.4f} +/- {held_out_summary['accuracy']['sample_sd']:.4f}")
    print(f"CLEAN BASELINE: {classification}")


if __name__ == "__main__":
    main()
