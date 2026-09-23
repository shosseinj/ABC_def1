"""Run the frozen N-MNIST SNN protocol across five deterministic model seeds."""
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

from experiments.nmnist.snn_baseline import (
    FramedNMNIST,
    environment_metadata,
    evaluate,
    make_loader,
    select_device,
    set_determinism,
    sha256,
    stratified_train_validation_indices,
    train,
    write_history,
    write_json,
)
from models.nmnist_snn import NMNISTConvSNN

SEEDS = [42, 123, 777, 2026, 6543]


def mean_sample_sd(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def frozen_config_for_seed(base_config, seed):
    config = dict(base_config)
    config["seed"] = int(seed)
    return config


def train_seed(base_config, dataset, train_ids, validation_ids, split_path, seed):
    results_dir = ROOT / "results"
    checkpoints_dir = ROOT / "checkpoints"
    validation_path = results_dir / f"nmnist_snn_clean_seed{seed}_validation.json"
    checkpoint_path = checkpoints_dir / f"nmnist_snn_clean_seed{seed}_best.pt"
    config_path = results_dir / f"nmnist_snn_clean_seed{seed}_config.json"
    config = frozen_config_for_seed(base_config, seed)
    write_json(config_path, config)

    if validation_path.exists():
        existing = json.loads(validation_path.read_text(encoding="utf-8"))
        if (existing.get("status") == "FROZEN_FOR_FINAL_TEST"
                and existing.get("seed") == seed
                and existing.get("checkpoint_sha256") == sha256(checkpoint_path)):
            existing.update({
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256(config_path),
                "split": str(split_path.relative_to(ROOT)),
                "split_sha256": sha256(split_path),
            })
            write_json(validation_path, existing)
            print(f"[seed={seed}] reusing hash-verified frozen validation checkpoint", flush=True)
            return existing
        raise RuntimeError(f"Seed {seed} has incompatible existing validation artifacts.")

    print(f"[seed={seed}] starting fixed-protocol training", flush=True)
    _, result = train(config, dataset, train_ids, validation_ids, checkpoint_path)
    history = result.pop("history")
    result.update({
        "status": "FROZEN_FOR_FINAL_TEST",
        "dataset": "N-MNIST",
        "seed": seed,
        "split_seed": base_config["split_seed"],
        "train_samples": len(train_ids),
        "validation_samples": len(validation_ids),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "split": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256(split_path),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "environment": environment_metadata(),
        "official_test_accessed": False,
        "attacks_run": False,
        "defenses_run": False,
    })
    write_history(results_dir / f"nmnist_snn_clean_seed{seed}_history.csv", history)
    write_json(validation_path, result)
    return result


def evaluate_seed_once(base_config, test_dataset, validation, seed):
    results_dir = ROOT / "results"
    test_path = results_dir / f"nmnist_snn_clean_seed{seed}_test.json"
    if test_path.exists():
        existing = json.loads(test_path.read_text(encoding="utf-8"))
        if (existing.get("status") == "FINAL_FROZEN"
                and existing.get("seed") == seed
                and existing.get("checkpoint_sha256") == validation["checkpoint_sha256"]):
            print(f"[seed={seed}] final test result already exists; reusing without test re-access", flush=True)
            return existing
        raise RuntimeError(f"Seed {seed} has incompatible existing test artifacts.")

    config = frozen_config_for_seed(base_config, seed)
    set_determinism(seed)
    device = select_device(config["device"])
    checkpoint_path = ROOT / validation["checkpoint"]
    if sha256(checkpoint_path) != validation["checkpoint_sha256"]:
        raise RuntimeError(f"Seed {seed} checkpoint hash changed before final test.")
    loader = make_loader(FramedNMNIST(test_dataset, config["temporal_bins"]),
                         config["batch_size"], False, seed)
    model = NMNISTConvSNN(config["lif_decay"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, loader, device, True)
    result = {
        "status": "FINAL_FROZEN",
        "dataset": "N-MNIST",
        "seed": seed,
        "split_seed": base_config["split_seed"],
        "test": test_metrics,
        "best_validation": validation["best_validation"],
        "best_epoch": validation["best_epoch"],
        "parameters": validation["parameters"],
        "training_runtime_seconds": validation["training_runtime_seconds"],
        "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "input_representation": config["input_representation"],
        "temporal_steps": config["temporal_bins"],
        "architecture": config["architecture"],
        "neuron_type": config["neuron_type"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "maximum_epochs": config["epochs"],
        "test_access_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_access_count": 1,
        "attacks_run": False,
        "defenses_run": False,
        "qsnn_modified": False,
    }
    write_json(test_path, result)
    with (results_dir / f"nmnist_snn_clean_seed{seed}_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "test_accuracy", "macro_f1", "best_epoch",
                  "parameters", "training_runtime_seconds", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "seed": seed,
            "validation_accuracy": validation["best_validation"]["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "best_epoch": validation["best_epoch"],
            "parameters": validation["parameters"],
            "training_runtime_seconds": validation["training_runtime_seconds"],
            "checkpoint_sha256": validation["checkpoint_sha256"],
        })
    print(f"[seed={seed}] final_test_accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}", flush=True)
    return result


def aggregate_results(results):
    parameters = {result["parameters"] for result in results}
    if len(parameters) != 1:
        raise RuntimeError("Parameter count changed across seeds.")
    aggregate = {
        "status": "COMPLETED",
        "seeds": SEEDS,
        "validation_accuracy": mean_sample_sd(
            result["best_validation"]["accuracy"] for result in results),
        "test_accuracy": mean_sample_sd(result["test"]["accuracy"] for result in results),
        "macro_f1": mean_sample_sd(result["test"]["macro_f1"] for result in results),
        "per_class_accuracy": {
            str(label): mean_sample_sd(
                result["test"]["per_class_accuracy"][str(label)] for result in results)
            for label in range(10)
        },
        "best_epoch_per_seed": {str(result["seed"]): result["best_epoch"] for result in results},
        "runtime_seconds_per_seed": {
            str(result["seed"]): result["training_runtime_seconds"] for result in results
        },
        "parameters": parameters.pop(),
        "official_test_evaluations_per_seed": 1,
        "attacks_run": False,
        "qsnn_experiments_run": False,
    }
    return aggregate


def update_readme(aggregate):
    path = ROOT / "readme_jafar.md"
    text = path.read_text(encoding="utf-8")
    old = "| Ours | Our SNN | SNN | Same internal split / preprocessing | TBD | Internal baseline |"
    mean = aggregate["test_accuracy"]["mean"] * 100
    sd = aggregate["test_accuracy"]["sample_sd"] * 100
    new = ("| Ours | Our SNN | SNN | Native-event convolutional LIF SNN; "
           f"5 deterministic seeds | {mean:.2f}% +/- {sd:.2f}% | Internal baseline |")
    if text.count(new) == 1 and old not in text:
        return
    if text.count(old) != 1:
        raise RuntimeError("Expected exactly one unchanged N-MNIST Our SNN row.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main():
    from tonic.datasets import NMNIST

    base_config_path = ROOT / "configs" / "nmnist_snn_clean_seed42.json"
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if base_config["attacks_enabled"] or base_config["defenses_enabled"] or base_config["qsnn_enabled"]:
        raise RuntimeError("This campaign is restricted to the frozen clean SNN protocol.")
    train_dataset = NMNIST(save_to=str(ROOT / base_config["data_root"]), train=True)
    train_ids, validation_ids = stratified_train_validation_indices(
        train_dataset.targets, base_config["validation_per_class"], base_config["split_seed"]
    )
    split_path = ROOT / "results" / "nmnist_snn_multiseed_split.json"
    split = {
        "dataset": "N-MNIST", "source_partition": "official training",
        "official_training_size": len(train_dataset), "split_seed": base_config["split_seed"],
        "train_indices": train_ids.tolist(), "validation_indices": validation_ids.tolist(),
        "train_class_counts": np.bincount(np.asarray(train_dataset.targets)[train_ids], minlength=10).tolist(),
        "validation_class_counts": np.bincount(
            np.asarray(train_dataset.targets)[validation_ids], minlength=10).tolist(),
        "official_test_instantiated_during_training": False,
    }
    write_json(split_path, split)
    validations = [
        train_seed(base_config, train_dataset, train_ids, validation_ids, split_path, seed)
        for seed in SEEDS
    ]
    del train_dataset
    print("[FINAL TEST] all checkpoints frozen; instantiating official test partition", flush=True)
    test_dataset = NMNIST(save_to=str(ROOT / base_config["data_root"]), train=False)
    final_results = [
        evaluate_seed_once(base_config, test_dataset, validation, seed)
        for seed, validation in zip(SEEDS, validations)
    ]
    aggregate = aggregate_results(final_results)
    write_json(ROOT / "results" / "nmnist_snn_multiseed_summary.json", aggregate)
    with (ROOT / "results" / "nmnist_snn_multiseed_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "test_accuracy", "macro_f1",
                  "best_epoch", "runtime_seconds", "parameters"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in final_results:
            writer.writerow({
                "seed": result["seed"],
                "validation_accuracy": result["best_validation"]["accuracy"],
                "test_accuracy": result["test"]["accuracy"],
                "macro_f1": result["test"]["macro_f1"],
                "best_epoch": result["best_epoch"],
                "runtime_seconds": result["training_runtime_seconds"],
                "parameters": result["parameters"],
            })
    update_readme(aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
