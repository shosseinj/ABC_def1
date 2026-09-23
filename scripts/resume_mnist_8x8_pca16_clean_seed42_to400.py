"""Continue the rich epoch-250 clean checkpoint through at most epoch 400."""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development
from experiments.mnist.training_resume_8x8 import resume_clean_training


def _loss_change(history, window):
    rows = history[-window:]
    return float(rows[0]["validation_loss"]) - float(rows[-1]["validation_loss"])


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_clean_resume_400.json").read_text(encoding="utf-8"))
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
    checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42_resumed.pt"
    output = ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42_epoch400.pt"
    run = resume_clean_training(
        config, reducer.transform(train_x), train_y, reducer.transform(val_x), val_y,
        checkpoint, [], output,
    )
    loss_change_10 = _loss_change(run["history"], 10)
    loss_change_20 = _loss_change(run["history"], 20)
    converged = run["stopping_reason"] == "early_stop"
    result = {
        "loaded_checkpoint": str(checkpoint.relative_to(ROOT)),
        "loaded_reducer": str(reducer_path.relative_to(ROOT)),
        "optimizer_restored": run["optimizer_restored"],
        "resumed_epoch": config["resume_epoch"], "best_epoch": run["best_epoch"],
        "stopping_epoch": run["stopping_epoch"], "stopping_reason": run["stopping_reason"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "runtime_seconds": run["runtime_seconds"],
        "validation_loss_change_final_10_epochs": loss_change_10,
        "validation_loss_change_final_20_epochs": loss_change_20,
        "convergence_established": converged,
        **run["best_metrics"],
    }
    (ROOT / "results" / "mnist_8x8_pca16_clean_seed42_epoch400.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (ROOT / "results" / "mnist_8x8_pca16_clean_seed42_epoch400_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run["history"][0].keys())
        writer.writeheader()
        writer.writerows(run["history"])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
