"""Continue the clean 4-block seed-42 capacity model from epoch 250."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.capacity_resume import resume_capacity_variant
from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development


def loss_change(history, window):
    rows = history[-window:]
    return float(rows[0]["validation_loss"]) - float(rows[-1]["validation_loss"])


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_blocks4_resume_400.json").read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("This entry point is clean-only.")
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")
    reducer_path = ROOT / "results" / "mnist_8x8_pca16_train_only_reducer.npz"
    reducer = TrainingOnlyPCAReducer.load(reducer_path)
    model_checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_capacity_blocks4_seed42.pt"
    recovery_checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_capacity_blocks4_seed42_latest.pt"
    output_checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_capacity_blocks4_seed42_epoch400.pt"
    run = resume_capacity_variant(
        config, reducer.transform(train_x), train_y, reducer.transform(val_x), val_y,
        model_checkpoint, recovery_checkpoint, output_checkpoint,
    )
    change_10 = loss_change(run["history"], 10)
    change_20 = loss_change(run["history"], 20)
    result = {
        "checkpoint_loaded": str(model_checkpoint.relative_to(ROOT)),
        "recovery_checkpoint": str(recovery_checkpoint.relative_to(ROOT)),
        "optimizer_restored": run["optimizer_restored"], "resumed_epoch": 250,
        "best_epoch": run["best_epoch"], "stopping_epoch": run["stopping_epoch"],
        "stopping_reason": run["stopping_reason"], "runtime_seconds": run["runtime_seconds"],
        "validation_loss_change_final_10_epochs": change_10,
        "validation_loss_change_final_20_epochs": change_20,
        "convergence_established": run["stopping_reason"] == "early_stop",
        "train": run["train"], "validation": run["validation"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
    }
    (ROOT / "results" / "mnist_8x8_pca16_blocks4_seed42_epoch400.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "results" / "mnist_8x8_pca16_blocks4_seed42_epoch400_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run["history"][0].keys())
        writer.writeheader()
        writer.writerows(run["history"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
