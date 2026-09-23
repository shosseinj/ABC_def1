"""Recover the interrupted diagnostic without rerunning the completed 3-block arm."""
from pathlib import Path
import csv
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.capacity_training import _theta, evaluate, train_capacity_variant
from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development
from models.mnist_qsnn import MNISTReducedQSNN


def evaluate_checkpoint(path, blocks, train_theta, train_y, val_theta, val_y, batch_size):
    model = MNISTReducedQSNN(8, blocks, 10)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("model_state", payload))
    return {
        "blocks": blocks, "circuit_depth": model.circuit_depth(),
        "trainable_parameters": model.trainable_parameter_count(),
        "train": evaluate(model, train_theta, train_y, batch_size, True),
        "validation": evaluate(model, val_theta, val_y, batch_size, True),
    }


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_capacity_seed42.json").read_text(encoding="utf-8"))
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")
    reducer_path = ROOT / "results" / "mnist_8x8_pca16_train_only_reducer.npz"
    reducer = TrainingOnlyPCAReducer.load(reducer_path)
    train_reduced, val_reduced = reducer.transform(train_x), reducer.transform(val_x)
    train_theta = _theta(train_reduced, config["time_window"])
    val_theta = _theta(val_reduced, config["time_window"])

    reference = evaluate_checkpoint(
        ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42_epoch400.pt",
        2, train_theta, train_y, val_theta, val_y, config["batch_size"],
    )
    reference.update({"best_epoch": 400, "stopping_epoch": 400, "stopping_reason": "reference", "runtime_seconds": None})
    blocks3 = evaluate_checkpoint(
        ROOT / "checkpoints" / "mnist_8x8_pca16_capacity_blocks3_seed42.pt",
        3, train_theta, train_y, val_theta, val_y, config["batch_size"],
    )
    blocks3.update({
        "best_epoch": 250, "stopping_epoch": 250, "stopping_reason": "max_epoch",
        "runtime_seconds": None,
        "runtime_note": "Unavailable because the parent diagnostic timed out after this arm completed.",
    })
    blocks4 = train_capacity_variant(
        config, 4, train_reduced, train_y, val_reduced, val_y,
        ROOT / "checkpoints" / "mnist_8x8_pca16_capacity_blocks4_seed42.pt",
    )
    history = blocks4.pop("history")
    result = {
        "seed": 42, "loaded_reducer": str(reducer_path.relative_to(ROOT)),
        "feature_schedule": config["feature_schedule"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "runs": [reference, blocks3, blocks4],
    }
    (ROOT / "results" / "mnist_8x8_pca16_capacity_seed42.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "results" / "mnist_8x8_pca16_capacity_blocks4_seed42_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
