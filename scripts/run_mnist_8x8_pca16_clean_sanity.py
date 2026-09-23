"""Run only the seed-42 clean 8x8 representation sanity experiment."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import TrainingOnlyPCAReducer, load_mnist_8x8_development
from experiments.mnist.training_8x8 import train_clean_sanity


def main():
    config = json.loads((ROOT / "configs" / "mnist_8x8_pca16_clean_sanity.json").read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"]:
        raise RuntimeError("This entry point is clean-only.")
    train_x, val_x, train_y, val_y, train_ids, val_ids = load_mnist_8x8_development(config)
    old_manifest = json.loads((ROOT / "results" / "mnist_4x4_split_indices.json").read_text(encoding="utf-8"))
    if train_ids.tolist() != old_manifest["train_indices"] or val_ids.tolist() != old_manifest["validation_indices"]:
        raise RuntimeError("8x8 experiment does not reproduce the frozen development split.")
    reducer = TrainingOnlyPCAReducer(config["input_features"]).fit(train_x)
    reducer_path = ROOT / "results" / "mnist_8x8_pca16_train_only_reducer.npz"
    reducer.save(reducer_path)
    reduced_train = reducer.transform(train_x)
    reduced_val = reducer.transform(val_x)
    checkpoint = ROOT / "checkpoints" / "mnist_8x8_pca16_clean_seed_42.pt"
    run = train_clean_sanity(config, reduced_train, train_y, reduced_val, val_y, checkpoint)
    result = {
        "seed": config["sanity_seed"], "split_seed": config["split_seed"],
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
        "downsampled_input_dimension": 64, "resulting_input_dimension": config["input_features"],
        "n_qubits": config["n_qubits"], "circuit_depth": run["model"].circuit_depth(),
        "parameter_count": run["model"].trainable_parameter_count(),
        "pca_explained_variance_ratio_sum": float(np.sum(reducer.pca.explained_variance_ratio_)),
        "reducer": str(reducer_path.relative_to(ROOT)), "checkpoint": str(checkpoint.relative_to(ROOT)),
        "best_epoch": run["best_epoch"], "training_runtime_seconds": run["runtime_seconds"],
        "validation_loss": run["validation_loss"], "validation_accuracy": run["validation_accuracy"],
        "validation_macro_f1": run["validation_macro_f1"], "per_class_accuracy": run["per_class_accuracy"],
        "confusion_matrix": run["confusion_matrix"],
    }
    (ROOT / "results" / "mnist_8x8_pca16_clean_seed42.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (ROOT / "results" / "mnist_8x8_pca16_clean_seed42_history.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{key: value for key, value in row.items() if key not in ("per_class_accuracy", "confusion_matrix")} for row in run["history"]]
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
