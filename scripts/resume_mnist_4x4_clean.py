"""Resume only exact epoch-40 MNIST clean checkpoints using validation CE."""
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
from experiments.mnist.training import resume_mnist_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads((ROOT / "configs" / "mnist_4x4_clean.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results" / "mnist_4x4_clean_validation.json").read_text(encoding="utf-8"))
    with (ROOT / "results" / "mnist_4x4_clean_training_history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    train_x, val_x, train_y, val_y, _, _ = load_mnist_development(config)
    prior_by_seed = {int(row["seed"]): row for row in prior["rows"]}
    rows, combined_history = [], list(history)
    missing = []
    for seed in config["model_seeds"]:
        source = ROOT / "checkpoints" / f"mnist_4x4_clean_seed_{seed}.pt"
        prior_row = prior_by_seed[seed]
        if not source.exists() or int(prior_row["best_epoch"]) != int(config["resume_epoch"]):
            reason = "missing" if not source.exists() else f"contains_best_epoch_{prior_row['best_epoch']}_not_epoch_40"
            print(f"[RESUME MISSING] seed={seed} checkpoint={source} reason={reason}", flush=True)
            missing.append({"seed": seed, "checkpoint": str(source), "reason": reason})
            continue
        output = ROOT / "checkpoints" / f"mnist_4x4_clean_resumed_seed_{seed}.pt"
        run = resume_mnist_model(
            {**config, "seed": seed}, train_x, train_y, val_x, val_y,
            source, history, output,
        )
        combined_history = [row for row in combined_history if int(row["seed"]) != seed] + run["history"]
        rows.append({
            "seed": seed, "checkpoint_loaded": str(source.relative_to(ROOT)),
            "resumed_epoch": run["resumed_epoch"], "stopping_epoch": run["stopping_epoch"],
            "best_epoch": run["best_epoch"], "stopping_reason": run["stopping_reason"],
            "optimizer_restored": run["optimizer_restored"],
            "best_validation_cross_entropy": run["best_val_loss"],
            "validation_accuracy": run["validation"]["accuracy"],
            "validation_macro_f1": run["validation"]["macro_f1"],
            "validation_class_accuracy": run["validation"]["class_accuracy"],
            "runtime_seconds": run["runtime_seconds"],
            "checkpoint": str(output.relative_to(ROOT)), "checkpoint_sha256": sha256(output),
        })
    combined_history.sort(key=lambda row: (int(row["seed"]), int(row["epoch"])))
    output_history = ROOT / "results" / "mnist_4x4_clean_resumed_history.csv"
    with output_history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "epoch", "training_loss", "validation_cross_entropy", "validation_accuracy", "validation_macro_f1"])
        writer.writeheader(); writer.writerows(combined_history)
    summary = None
    if len(rows) == len(config["model_seeds"]):
        accuracies = np.asarray([row["validation_accuracy"] for row in rows])
        summary = {
            "validation_accuracy_mean": float(accuracies.mean()),
            "validation_accuracy_sample_sd": float(accuracies.std(ddof=1)),
            "minimum_validation_accuracy": float(accuracies.min()),
            "mean_best_epoch": float(np.mean([row["best_epoch"] for row in rows])),
            "seeds_reaching_epoch_250": int(sum(row["stopping_reason"] == "max_epoch" for row in rows)),
        }
    artifact = {"status": "INCOMPLETE_MISSING_EPOCH_40_CHECKPOINTS" if missing else "COMPLETE",
                "held_out_accessed": False, "attacks_run": False, "rows": rows,
                "missing": missing, "summary": summary}
    (ROOT / "results" / "mnist_4x4_clean_resumed.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print("MNIST RESUMED TRAINING: MAY NEED LONGER TRAINING", flush=True)


if __name__ == "__main__":
    main()
