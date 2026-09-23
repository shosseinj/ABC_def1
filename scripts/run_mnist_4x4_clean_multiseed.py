"""Train and freeze the validation-selected 4x4 MNIST clean QSNN protocol."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import load_mnist_development
from experiments.mnist.training import train_mnist_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def main():
    config_path = ROOT / "configs" / "mnist_4x4_clean.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("MNIST clean baseline cannot run attacks or defenses.")
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_development(config)
    results_dir = ROOT / "results"
    split_manifest = {
        "dataset": "MNIST", "official_train_partition": True,
        "split_seed": config["split_seed"],
        "train_indices": train_ids.tolist(), "validation_indices": val_ids.tolist(),
        "held_out_partition": "official MNIST test indices 0..9999; features not accessed",
        "overlap_count": int(np.intersect1d(train_ids, val_ids).size),
    }
    (results_dir / "mnist_4x4_split_indices.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    rows, histories, checkpoints = [], [], []
    for seed in config["model_seeds"]:
        checkpoint = ROOT / "checkpoints" / f"mnist_4x4_clean_seed_{seed}.pt"
        run = train_mnist_model(
            {**config, "seed": seed}, train_x, train_y, val_x, val_y, checkpoint
        )
        row = {
            "seed": seed, "best_epoch": run["best_epoch"],
            "validation_accuracy": run["validation"]["accuracy"],
            "validation_macro_f1": run["validation"]["macro_f1"],
            "validation_cross_entropy": run["validation"]["cross_entropy"],
            "validation_class_accuracy": run["validation"]["class_accuracy"],
            "runtime_seconds": run["runtime_seconds"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
        }
        rows.append(row)
        histories.extend(run["history"])
        checkpoints.append({key: row[key] for key in (
            "seed", "best_epoch", "checkpoint", "checkpoint_sha256"
        )})
        print(f"[SEED COMPLETE] seed={seed} best_epoch={row['best_epoch']} val_acc={row['validation_accuracy']:.4f}", flush=True)

    summary = {
        "validation_accuracy": mean_sd([row["validation_accuracy"] for row in rows]),
        "validation_macro_f1": mean_sd([row["validation_macro_f1"] for row in rows]),
        "validation_cross_entropy": mean_sd([row["validation_cross_entropy"] for row in rows]),
        "minimum_seed_accuracy": float(min(row["validation_accuracy"] for row in rows)),
        "class_accuracy_mean": {
            str(label): float(np.mean([row["validation_class_accuracy"][str(label)] for row in rows]))
            for label in range(10)
        },
        "average_training_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows])),
    }
    acceptable = (
        summary["validation_accuracy"]["mean"] >= config["acceptable_mean_validation_accuracy"]
        and summary["validation_macro_f1"]["mean"] >= config["acceptable_mean_validation_macro_f1"]
        and summary["minimum_seed_accuracy"] >= config["acceptable_minimum_seed_accuracy"]
        and summary["validation_accuracy"]["sample_sd"] <= config["acceptable_maximum_accuracy_sample_sd"]
        and min(summary["class_accuracy_mean"].values()) >= config["acceptable_minimum_mean_class_accuracy"]
    )
    protocol = {
        "status": "FROZEN_BEFORE_HELD_OUT" if acceptable else "NEEDS_IMPROVEMENT",
        "dataset": "MNIST", "representation": "4x4 average-pooled pixels",
        "input_features": 16, "split_seed": config["split_seed"],
        "model_seeds": config["model_seeds"],
        "split_sizes": {"train": len(train_ids), "validation": len(val_ids), "held_out": 10000},
        "architecture": {"n_qubits": 4, "reupload_blocks": 4, "circuit_depth": 28,
                         "quantum_features": 8, "trainable_parameters": 122},
        "encoding": config["encoding"], "checkpoint_rule": config["checkpoint_rule"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "config": config, "config_sha256": sha256(config_path),
        "checkpoints": checkpoints, "rows": rows, "validation_summary": summary,
    }
    (results_dir / "mnist_4x4_clean_frozen_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    with (results_dir / "mnist_4x4_clean_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys())
        writer.writeheader(); writer.writerows(histories)
    with (results_dir / "mnist_4x4_clean_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "validation_class_accuracy": json.dumps(row["validation_class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
        writer.writeheader(); writer.writerows(flat)
    (results_dir / "mnist_4x4_clean_validation.json").write_text(
        json.dumps({"rows": rows, "summary": summary, "classification": "ACCEPTABLE" if acceptable else "NEEDS_IMPROVEMENT"}, indent=2),
        encoding="utf-8",
    )
    print(f"MNIST DEVELOPMENT GATE: {'ACCEPTABLE' if acceptable else 'NEEDS IMPROVEMENT'}", flush=True)


if __name__ == "__main__":
    main()
