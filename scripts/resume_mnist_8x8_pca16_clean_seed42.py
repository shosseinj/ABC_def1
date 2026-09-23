"""Continue the clean seed-42 PCA-16 QSNN without refitting preprocessing."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development
from experiments.mnist.training_resume_8x8 import resume_clean_training


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_clean_resume.json").read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("This entry point is clean-only.")
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen seed-42 split.")

    reducer_path = ROOT / "results" / "mnist_8x8_pca16_train_only_reducer.npz"
    reducer = TrainingOnlyPCAReducer.load(reducer_path)
    if reducer.pca.components_.shape != (config["input_features"], config["downsampled_features"]):
        raise RuntimeError("Saved reducer dimensions do not match the frozen architecture.")
    train_reduced = reducer.transform(train_x)
    val_reduced = reducer.transform(val_x)

    history_path = ROOT / "results" / "mnist_8x8_pca16_clean_seed42_history.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42.pt"
    output = ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42_resumed.pt"
    run = resume_clean_training(
        config, train_reduced, train_y, val_reduced, val_y,
        checkpoint, history, output,
    )
    terminal = run["history"][-10:]
    terminal_losses = [float(row["validation_loss"]) for row in terminal]
    loss_change_last_10 = terminal_losses[0] - terminal_losses[-1] if len(terminal_losses) > 1 else 0.0
    result = {
        "loaded_checkpoint": str(checkpoint.relative_to(ROOT)),
        "loaded_reducer": str(reducer_path.relative_to(ROOT)),
        "resumed_epoch": config["resume_epoch"], "best_epoch": run["best_epoch"],
        "stopping_epoch": run["stopping_epoch"], "stopping_reason": run["stopping_reason"],
        "optimizer_restored": run["optimizer_restored"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "runtime_seconds": run["runtime_seconds"],
        "loss_change_over_last_10_epochs": float(loss_change_last_10),
        "loss_still_improving_near_stop": bool(loss_change_last_10 > 0.001),
        **run["best_metrics"],
    }
    result_path = ROOT / "results" / "mnist_8x8_pca16_clean_seed42_resumed.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    resumed_history = ROOT / "results" / "mnist_8x8_pca16_clean_seed42_resumed_history.csv"
    with resumed_history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run["history"][0].keys())
        writer.writeheader()
        writer.writerows(run["history"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
