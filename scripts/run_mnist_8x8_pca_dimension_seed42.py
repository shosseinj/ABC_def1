"""Run one controlled clean MNIST PCA-dimension arm."""
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development
from experiments.mnist.pca_training import train_pca_variant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dim", type=int, choices=(24, 32), required=True)
    args = parser.parse_args()
    config = json.loads(
        (ROOT / "configs" / "mnist_8x8_pca_dimension_seed42.json").read_text(encoding="utf-8")
    )
    if config["attacks_enabled"] or config["defenses_enabled"] or config["held_out_enabled"]:
        raise RuntimeError("PCA dimension ablation must remain clean and development-only.")

    print(f"[PCA START] dim={args.pca_dim}", flush=True)
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    split = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != split["train_indices"] or val_ids.tolist() != split["validation_indices"]:
        raise RuntimeError("Development split differs from the frozen split.")

    reducer_path = ROOT / "results" / f"mnist_8x8_pca{args.pca_dim}_train_only_reducer.npz"
    checkpoint_path = ROOT / "checkpoints" / f"mnist_8x8_pca{args.pca_dim}_blocks4_seed42.pt"
    recovery_path = checkpoint_path.with_name(checkpoint_path.stem + "_latest.pt")
    if recovery_path.exists():
        if not reducer_path.exists():
            raise RuntimeError("Recovery checkpoint exists without its fitted reducer.")
        reducer = TrainingOnlyPCAReducer.load(reducer_path)
        prior_runtime = max(0.0, recovery_path.stat().st_mtime - recovery_path.stat().st_ctime)
    else:
        reducer = TrainingOnlyPCAReducer(args.pca_dim).fit(train_x)
        reducer.save(reducer_path)
        prior_runtime = 0.0
    reduced_train = reducer.transform(train_x)
    reduced_val = reducer.transform(val_x)
    run = train_pca_variant(
        config, args.pca_dim, reduced_train, train_y, reduced_val, val_y, checkpoint_path,
        recovery_path=recovery_path, prior_runtime_seconds=prior_runtime,
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
        "split_seed": config["split_seed"],
        "model_seed": config["seed"],
        "held_out_accessed": False,
        "attacks_run": False,
        "defenses_run": False,
        "runtime_note": "Segmented wall time; interrupted legacy segment recovered from checkpoint timestamps."
        if run["runtime_segmented"] else "Single uninterrupted training process.",
    })
    result_path = ROOT / "results" / f"mnist_8x8_pca{args.pca_dim}_blocks4_seed42.json"
    result_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    history_path = ROOT / "results" / f"mnist_8x8_pca{args.pca_dim}_blocks4_seed42_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(
        f"[PCA COMPLETE] dim={args.pca_dim} best_epoch={run['best_epoch']} "
        f"val_acc={run['validation']['accuracy']:.4f}", flush=True,
    )


if __name__ == "__main__":
    main()
