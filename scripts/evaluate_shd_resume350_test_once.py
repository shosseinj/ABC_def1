"""Evaluate five frozen SHD resume350 checkpoints once on the official test set."""
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import select_device, set_determinism, sha256, write_json
from experiments.shd.snn_baseline import evaluate, make_tensor_loader, preprocess_partition
from models.shd_snn import SHDRecurrentSNN

SEEDS = [42, 123, 777, 2026, 6543]


def mean_sample_sd(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def preflight():
    validations = []
    for seed in SEEDS:
        result_path = ROOT / "results" / f"shd_snn_seed{seed}_resume350_test.json"
        if result_path.exists():
            raise RuntimeError(f"Seed {seed} resume350 test artifact exists; refusing repeated access.")
        validation_path = ROOT / "results" / f"shd_snn_seed{seed}_resume350_validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        checkpoint_path = ROOT / validation["checkpoint"]
        config_path = ROOT / validation["config"]
        if validation["seed"] != seed or sha256(checkpoint_path) != validation["checkpoint_sha256"]:
            raise RuntimeError(f"Seed {seed} checkpoint ownership or hash mismatch.")
        if sha256(config_path) != validation["config_sha256"]:
            raise RuntimeError(f"Seed {seed} config hash mismatch.")
        validations.append(validation)
    return validations


def evaluate_one(validation, frames, labels):
    seed = validation["seed"]
    config = json.loads((ROOT / validation["config"]).read_text(encoding="utf-8"))
    set_determinism(seed)
    device = select_device(config["device"])
    model = SHDRecurrentSNN(config["input_channels"], config["hidden_size"],
                            config["n_classes"], config["lif_decay"]).to(device)
    checkpoint = torch.load(ROOT / validation["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    loader = make_tensor_loader(frames, labels, np.arange(len(labels)),
                                config["batch_size"], False, seed)
    test = evaluate(model, loader, device, config["n_classes"], True)
    result = {
        "status": "FINAL_FROZEN", "dataset": "SHD", "seed": seed,
        "validation_accuracy": validation["new_best_validation_accuracy"], "test": test,
        "best_epoch": validation["new_best_epoch"], "stopping_epoch": validation["stopping_epoch"],
        "parameters": validation["parameters"], "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "test_access_utc": datetime.now(timezone.utc).isoformat(), "official_test_access_count": 1,
        "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(ROOT / "results" / f"shd_snn_seed{seed}_resume350_test.json", result)
    print(f"seed={seed} validation={result['validation_accuracy']:.4f} "
          f"test={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}", flush=True)
    return result


def main():
    from tonic.datasets import SHD

    validations = preflight()
    base_config = json.loads((ROOT / "configs" / "shd_snn_multiseed.json").read_text(encoding="utf-8"))
    print("[FINAL TEST] five resume350 checkpoints verified; loading official SHD test partition", flush=True)
    dataset = SHD(save_to=str(ROOT / base_config["data_root"]), train=False)
    frames, labels, preprocessing = preprocess_partition(dataset, base_config, "official_test_resume350")
    preprocessing["test_h5_sha256"] = sha256(ROOT / base_config["data_root"] / "SHD" / "shd_test.h5")
    write_json(ROOT / "results" / "shd_snn_resume350_test_preprocessing.json", preprocessing)
    results = [evaluate_one(validation, frames, labels) for validation in validations]
    summary = {
        "status": "COMPLETED", "dataset": "SHD", "seeds": SEEDS,
        "validation_accuracy": mean_sample_sd(result["validation_accuracy"] for result in results),
        "test_accuracy": mean_sample_sd(result["test"]["accuracy"] for result in results),
        "macro_f1": mean_sample_sd(result["test"]["macro_f1"] for result in results),
        "per_class_accuracy": {str(label): mean_sample_sd(
            result["test"]["per_class_accuracy"][str(label)] for result in results) for label in range(20)},
        "best_epoch_per_seed": {str(result["seed"]): result["best_epoch"] for result in results},
        "parameters": results[0]["parameters"], "official_test_evaluations_per_seed": 1,
        "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(ROOT / "results" / "shd_snn_resume350_test_summary.json", summary)
    with (ROOT / "results" / "shd_snn_resume350_test_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "test_accuracy", "macro_f1", "best_epoch", "parameters"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"seed": result["seed"], "validation_accuracy": result["validation_accuracy"],
                             "test_accuracy": result["test"]["accuracy"],
                             "macro_f1": result["test"]["macro_f1"], "best_epoch": result["best_epoch"],
                             "parameters": result["parameters"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
