"""Run one gated clean MNIST PCA-24 resolution arm."""
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
    config = json.loads(
        (ROOT / "configs" / "mnist_pca24_resolution_seed42.json").read_text(encoding="utf-8")
    )
    if config["attacks_enabled"] or config["defenses_enabled"] or config["held_out_enabled"]:
        raise RuntimeError("Resolution ablation must remain clean and development-only.")
    if args.resolution == 14:
        result_12 = json.loads(
            (ROOT / "results" / "mnist_12x12_pca24_seed42.json").read_text(encoding="utf-8")
        )
        required = config["reference_validation_accuracy"] + config["meaningful_accuracy_improvement"]
        if result_12["validation"]["accuracy"] < required:
            raise RuntimeError("12x12 PCA-24 did not pass the predeclared gate; 14x14 is prohibited.")

    print(f"[RES START] resolution={args.resolution}x{args.resolution} pca=24", flush=True)
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_resolution_development(
        config, args.resolution
    )
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")

    reducer = TrainingOnlyPCAReducer(24).fit(train_x)
    reducer_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca24_train_only_reducer.npz"
    reducer.save(reducer_path)
    reduced_train, reduced_val = reducer.transform(train_x), reducer.transform(val_x)
    checkpoint_path = ROOT / "checkpoints" / f"mnist_{args.resolution}x{args.resolution}_pca24_blocks4_seed42.pt"
    run = train_resolution_variant(
        config, args.resolution, reduced_train, train_y, reduced_val, val_y, checkpoint_path
    )
    history = run.pop("history")
    raw_validation = reducer.pca.transform(val_x)
    run.update({
        "explained_variance": float(np.sum(reducer.pca.explained_variance_ratio_)),
        "reducer": str(reducer_path.relative_to(ROOT)),
        "pca_fit_partition": "training only",
        "validation_clipping_fraction": float(np.mean(
            (raw_validation < reducer.minimum) | (raw_validation > reducer.maximum)
        )),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "split_seed": config["split_seed"], "model_seed": config["seed"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
    })
    result_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca24_seed42.json"
    result_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    history_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca24_seed42_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary_path = ROOT / "results" / f"mnist_{args.resolution}x{args.resolution}_pca24_seed42_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "resolution", "raw_features", "pca_features", "explained_variance",
            "validation_accuracy", "validation_macro_f1", "validation_ce",
            "train_accuracy", "best_epoch", "stopping_epoch", "convergence_established",
            "qubits", "reupload_blocks", "circuit_depth", "parameters", "runtime_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "resolution": run["resolution"], "raw_features": run["raw_features"],
            "pca_features": run["pca_features"], "explained_variance": run["explained_variance"],
            "validation_accuracy": run["validation"]["accuracy"],
            "validation_macro_f1": run["validation"]["macro_f1"],
            "validation_ce": run["validation"]["loss"], "train_accuracy": run["train"]["accuracy"],
            "best_epoch": run["best_epoch"], "stopping_epoch": run["stopping_epoch"],
            "convergence_established": run["convergence_established"], "qubits": run["qubits"],
            "reupload_blocks": run["reupload_blocks"], "circuit_depth": run["circuit_depth"],
            "parameters": run["parameters"], "runtime_seconds": run["runtime_seconds"],
        })
    print(
        f"[RES COMPLETE] resolution={args.resolution}x{args.resolution} "
        f"best_epoch={run['best_epoch']} val_acc={run['validation']['accuracy']:.4f}", flush=True,
    )


if __name__ == "__main__":
    main()
