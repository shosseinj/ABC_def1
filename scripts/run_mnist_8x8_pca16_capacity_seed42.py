"""Run the frozen-data seed-42 3/4-block clean capacity diagnostic."""
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


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_capacity_seed42.json").read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("Capacity diagnostics must remain clean-only.")
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")
    reducer_path = ROOT / "results" / "mnist_8x8_pca16_train_only_reducer.npz"
    reducer = TrainingOnlyPCAReducer.load(reducer_path)
    train_reduced = reducer.transform(train_x)
    val_reduced = reducer.transform(val_x)

    reference_model = MNISTReducedQSNN(8, 2, 10)
    reference_payload = torch.load(
        ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42_epoch400.pt",
        map_location="cpu", weights_only=True,
    )
    reference_model.load_state_dict(reference_payload["model_state"])
    reference_train = evaluate(reference_model, _theta(train_reduced, config["time_window"]), train_y, config["batch_size"], True)
    reference_val = evaluate(reference_model, _theta(val_reduced, config["time_window"]), val_y, config["batch_size"], True)
    runs = [{
        "blocks": 2, "circuit_depth": reference_model.circuit_depth(),
        "trainable_parameters": reference_model.trainable_parameter_count(),
        "best_epoch": 400, "stopping_epoch": 400, "stopping_reason": "reference",
        "train": reference_train, "validation": reference_val,
        "runtime_seconds": None, "history": [],
    }]
    for blocks in (3, 4):
        runs.append(train_capacity_variant(
            config, blocks, train_reduced, train_y, val_reduced, val_y,
            ROOT / "checkpoints" / f"mnist_8x8_pca16_capacity_blocks{blocks}_seed42.pt",
        ))

    result = {
        "seed": 42, "loaded_reducer": str(reducer_path.relative_to(ROOT)),
        "feature_schedule": config["feature_schedule"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "runs": [{key: value for key, value in run.items() if key != "history"} for run in runs],
    }
    (ROOT / "results" / "mnist_8x8_pca16_capacity_seed42.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    histories = [row for run in runs for row in run["history"]]
    with (ROOT / "results" / "mnist_8x8_pca16_capacity_seed42_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys())
        writer.writeheader()
        writer.writerows(histories)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
