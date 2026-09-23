"""Train and freeze the seed-42 native-event N-MNIST convolutional SNN."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import (
    environment_metadata, sha256, stratified_train_validation_indices, train, write_history, write_json,
)


def main():
    from tonic.datasets import NMNIST

    config_path = ROOT / "configs" / "nmnist_snn_clean_seed42.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"] or config["qsnn_enabled"]:
        raise RuntimeError("This command is restricted to clean SNN training.")
    dataset = NMNIST(save_to=str(ROOT / config["data_root"]), train=True)
    train_ids, validation_ids = stratified_train_validation_indices(
        dataset.targets, config["validation_per_class"], config["split_seed"]
    )
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    split_path = results_dir / "nmnist_snn_clean_seed42_split.json"
    split = {"dataset": "N-MNIST", "source_partition": "official training",
             "official_training_size": len(dataset), "split_seed": config["split_seed"],
             "train_indices": train_ids.tolist(), "validation_indices": validation_ids.tolist(),
             "train_class_counts": np.bincount(np.asarray(dataset.targets)[train_ids], minlength=10).tolist(),
             "validation_class_counts": np.bincount(np.asarray(dataset.targets)[validation_ids], minlength=10).tolist(),
             "official_test_instantiated": False}
    write_json(split_path, split)
    checkpoint_path = checkpoints_dir / "nmnist_snn_clean_seed42_best.pt"
    print(f"input={config['input_representation']} architecture={config['architecture']}", flush=True)
    _, result = train(config, dataset, train_ids, validation_ids, checkpoint_path)
    history = result.pop("history")
    result.update({"status": "FROZEN_FOR_FINAL_TEST", "dataset": "N-MNIST", "seed": config["seed"],
                   "train_samples": len(train_ids), "validation_samples": len(validation_ids),
                   "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
                   "split": str(split_path.relative_to(ROOT)), "split_sha256": sha256(split_path),
                   "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                   "checkpoint_sha256": sha256(checkpoint_path), "environment": environment_metadata(),
                   "official_test_accessed": False, "attacks_run": False, "defenses_run": False})
    write_history(results_dir / "nmnist_snn_clean_seed42_history.csv", history)
    write_json(results_dir / "nmnist_snn_clean_seed42_validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
