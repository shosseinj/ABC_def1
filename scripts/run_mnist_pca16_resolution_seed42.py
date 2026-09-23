"""Run one gated clean MNIST resolution arm with a training-only PCA reducer."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_resolution_development
from experiments.mnist.resolution_training import train_resolution_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, choices=(12, 14), required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / "mnist_pca16_resolution_seed42.json").read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("Resolution diagnostics must remain clean-only.")
    if args.resolution == 14:
        result_12 = json.loads((ROOT / "results" / "mnist_12x12_pca16_seed42.json").read_text(encoding="utf-8"))
        required = config["reference_validation_accuracy"] + config["meaningful_accuracy_improvement"]
        if result_12["validation"]["accuracy"] < required:
            raise RuntimeError("12x12 did not pass the predeclared gate; 14x14 is prohibited.")

    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_resolution_development(config, args.resolution)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")
    reducer = TrainingOnlyPCAReducer(config["pca_features"]).fit(train_x)
    reducer_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca16_train_only_reducer.npz"
    reducer.save(reducer_path)
    reduced_train, reduced_val = reducer.transform(train_x), reducer.transform(val_x)
    run = train_resolution_variant(
        config, args.resolution, reduced_train, train_y, reduced_val, val_y,
        ROOT / "checkpoints" / f"mnist_{args.resolution}x{args.resolution}_pca16_blocks4_seed42.pt",
    )
    history = run.pop("history")
    run.update({
        "explained_variance": float(np.sum(reducer.pca.explained_variance_ratio_)),
        "reducer": str(reducer_path.relative_to(ROOT)), "pca_fit_partition": "training only",
        "validation_clipping_fraction": float(np.mean((reducer.pca.transform(val_x) < reducer.minimum) | (reducer.pca.transform(val_x) > reducer.maximum))),
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
    })
    result_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca16_seed42.json"
    result_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    with (ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca16_seed42_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
