"""Evaluate the frozen N-MNIST SNN once on the official test partition."""
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import FramedNMNIST, evaluate, make_loader, sha256, write_json
from models.nmnist_snn import NMNISTConvSNN


def main():
    from tonic.datasets import NMNIST

    result_path = ROOT / "results" / "nmnist_snn_clean_seed42_test.json"
    if result_path.exists():
        raise RuntimeError("Final N-MNIST test result already exists; refusing a repeated test access.")
    config_path = ROOT / "configs" / "nmnist_snn_clean_seed42.json"
    validation_path = ROOT / "results" / "nmnist_snn_clean_seed42_validation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checkpoint_path = ROOT / validation["checkpoint"]
    if validation["status"] != "FROZEN_FOR_FINAL_TEST":
        raise RuntimeError("Checkpoint is not frozen for final test.")
    for path, expected in ((config_path, validation["config_sha256"]),
                           (checkpoint_path, validation["checkpoint_sha256"])):
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact hash mismatch: {path.name}")
    dataset = NMNIST(save_to=str(ROOT / config["data_root"]), train=False)
    loader = make_loader(FramedNMNIST(dataset, config["temporal_bins"]),
                         config["batch_size"], False, config["seed"])
    model = NMNISTConvSNN(config["lif_decay"]).to(config["device"])
    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, loader, torch.device(config["device"]), True)
    result = {"status": "FINAL_FROZEN", "dataset": "N-MNIST", "seed": config["seed"],
              "test": test, "best_validation": validation["best_validation"],
              "best_epoch": validation["best_epoch"], "parameters": validation["parameters"],
              "training_runtime_seconds": validation["training_runtime_seconds"],
              "checkpoint": validation["checkpoint"], "checkpoint_sha256": validation["checkpoint_sha256"],
              "input_representation": config["input_representation"], "temporal_steps": config["temporal_bins"],
              "architecture": config["architecture"], "neuron_type": config["neuron_type"],
              "optimizer": config["optimizer"], "learning_rate": config["learning_rate"],
              "batch_size": config["batch_size"], "maximum_epochs": config["epochs"],
              "test_access_utc": datetime.now(timezone.utc).isoformat(), "official_test_access_count": 1,
              "attacks_run": False, "defenses_run": False, "qsnn_modified": False}
    write_json(result_path, result)
    with (ROOT / "results" / "nmnist_snn_clean_seed42_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["best_validation_accuracy", "test_accuracy", "macro_f1",
                                                    "best_epoch", "parameters", "training_runtime_seconds"])
        writer.writeheader()
        writer.writerow({"best_validation_accuracy": validation["best_validation"]["accuracy"],
                         "test_accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
                         "best_epoch": validation["best_epoch"], "parameters": validation["parameters"],
                         "training_runtime_seconds": validation["training_runtime_seconds"]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
