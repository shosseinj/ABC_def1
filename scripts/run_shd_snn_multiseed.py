"""Run the frozen five-seed SHD clean SNN baseline and one final test per seed."""
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import (
    environment_metadata, print_runtime_configuration, select_device, set_determinism, sha256,
    write_history, write_json,
)
from experiments.shd.snn_baseline import (
    evaluate, make_tensor_loader, preprocess_partition, stratified_split, train_seed,
)
from models.shd_snn import SHDRecurrentSNN


def mean_sample_sd(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def labels_from_h5(path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["labels"], dtype=np.int64)


def seed_config(base, seed):
    config = dict(base)
    config.pop("seeds")
    config["seed"] = int(seed)
    return config


def train_one(base, dataset, frames, labels, train_ids, validation_ids, split_path, seed):
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    config = seed_config(base, seed)
    config_path = results_dir / f"shd_snn_seed{seed}_config.json"
    history_path = results_dir / f"shd_snn_seed{seed}_history.csv"
    validation_path = results_dir / f"shd_snn_seed{seed}_validation.json"
    checkpoint_path = checkpoints_dir / f"shd_snn_seed{seed}_best.pt"
    write_json(config_path, config)
    print(f"[seed={seed}] starting fixed SHD protocol", flush=True)
    _, result = train_seed(config, frames, labels, train_ids, validation_ids, checkpoint_path)
    history = result.pop("history")
    write_history(history_path, history)
    result.update({
        "status": "FROZEN_FOR_FINAL_TEST", "dataset": "SHD", "seed": seed,
        "split_seed": base["split_seed"], "train_samples": len(train_ids),
        "validation_samples": len(validation_ids), "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path), "split": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256(split_path), "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path), "environment": environment_metadata(),
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False,
        "qsnn_run": False,
    })
    write_json(validation_path, result)
    return result


def test_one(base, frames, labels, validation, seed):
    config = seed_config(base, seed)
    results_dir = ROOT / "results"
    result_path = results_dir / f"shd_snn_seed{seed}_test.json"
    if result_path.exists():
        raise RuntimeError(f"Seed {seed} final test artifact exists; refusing repeated evaluation.")
    checkpoint_path = ROOT / validation["checkpoint"]
    if sha256(checkpoint_path) != validation["checkpoint_sha256"]:
        raise RuntimeError(f"Seed {seed} checkpoint hash mismatch before test.")
    set_determinism(seed)
    device = select_device(config["device"])
    model = SHDRecurrentSNN(config["input_channels"], config["hidden_size"],
                            config["n_classes"], config["lif_decay"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"])
    loader = make_tensor_loader(frames, labels, np.arange(len(labels)), config["batch_size"], False, seed)
    test = evaluate(model, loader, device, config["n_classes"], True)
    result = {
        "status": "FINAL_FROZEN", "dataset": "SHD", "seed": seed,
        "best_validation": validation["best_validation"], "test": test,
        "best_epoch": validation["best_epoch"], "training_runtime_seconds": validation["training_runtime_seconds"],
        "parameters": validation["parameters"], "checkpoint": validation["checkpoint"],
        "checkpoint_sha256": validation["checkpoint_sha256"],
        "test_access_utc": datetime.now(timezone.utc).isoformat(), "official_test_access_count": 1,
        "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(result_path, result)
    with (results_dir / f"shd_snn_seed{seed}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "test_accuracy", "macro_f1", "best_epoch",
                  "runtime_seconds", "parameters", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"seed": seed, "validation_accuracy": validation["best_validation"]["accuracy"],
                         "test_accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
                         "best_epoch": validation["best_epoch"],
                         "runtime_seconds": validation["training_runtime_seconds"],
                         "parameters": validation["parameters"],
                         "checkpoint_sha256": validation["checkpoint_sha256"]})
    print(f"[seed={seed}] test_accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}", flush=True)
    return result


def aggregate(results, seeds):
    parameter_counts = {result["parameters"] for result in results}
    if len(parameter_counts) != 1:
        raise RuntimeError("SHD parameter count changed across seeds.")
    runtimes = [result["training_runtime_seconds"] for result in results]
    return {
        "status": "COMPLETED", "dataset": "SHD", "seeds": seeds,
        "validation_accuracy": mean_sample_sd(result["best_validation"]["accuracy"] for result in results),
        "test_accuracy": mean_sample_sd(result["test"]["accuracy"] for result in results),
        "macro_f1": mean_sample_sd(result["test"]["macro_f1"] for result in results),
        "per_class_accuracy": {str(label): mean_sample_sd(
            result["test"]["per_class_accuracy"][str(label)] for result in results) for label in range(20)},
        "best_epoch_per_seed": {str(result["seed"]): result["best_epoch"] for result in results},
        "runtime_seconds_per_seed": {str(result["seed"]): result["training_runtime_seconds"] for result in results},
        "mean_runtime_seconds": statistics.mean(runtimes), "parameters": parameter_counts.pop(),
        "official_test_evaluations_per_seed": 1, "attacks_run": False,
        "defenses_run": False, "qsnn_experiments_run": False,
    }


def write_aggregate_files(summary, results):
    results_dir = ROOT / "results"
    write_json(results_dir / "shd_snn_multiseed_summary.json", summary)
    with (results_dir / "shd_snn_multiseed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "test_accuracy", "macro_f1", "best_epoch",
                  "runtime_seconds", "parameters"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"seed": result["seed"], "validation_accuracy": result["best_validation"]["accuracy"],
                             "test_accuracy": result["test"]["accuracy"], "macro_f1": result["test"]["macro_f1"],
                             "best_epoch": result["best_epoch"], "runtime_seconds": result["training_runtime_seconds"],
                             "parameters": result["parameters"]})
    lines = ["# SHD SNN Multi-Seed Clean Baseline", "", "| Seed | Validation | Test | Macro-F1 | Best epoch | Runtime |",
             "|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result['seed']} | {result['best_validation']['accuracy']:.4f} | "
                     f"{result['test']['accuracy']:.4f} | {result['test']['macro_f1']:.4f} | "
                     f"{result['best_epoch']} | {result['training_runtime_seconds']:.2f} s |")
    lines.extend(["", f"Validation accuracy: {summary['validation_accuracy']['mean']:.4f} +/- "
                  f"{summary['validation_accuracy']['sample_sd']:.4f}",
                  f"Test accuracy: {summary['test_accuracy']['mean']:.4f} +/- {summary['test_accuracy']['sample_sd']:.4f}",
                  f"Macro-F1: {summary['macro_f1']['mean']:.4f} +/- {summary['macro_f1']['sample_sd']:.4f}",
                  f"Parameters: {summary['parameters']}", "",
                  "No attacks, defenses, or QSNN experiments were run."])
    (results_dir / "shd_snn_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme(summary):
    path = ROOT / "readme_jafar.md"
    text = path.read_text(encoding="utf-8")
    start, end = text.index("## SHD"), text.index("## CIFAR10-DVS")
    section = text[start:end]
    old = "| Ours | Our SNN | SNN | Same internal split / preprocessing | TBD | Internal baseline |"
    mean, sd = summary["test_accuracy"]["mean"] * 100, summary["test_accuracy"]["sample_sd"] * 100
    new = ("| Ours | Our SNN | SNN | Native temporal SHD recurrent LIF SNN; 5 deterministic seeds | "
           f"{mean:.2f}% +/- {sd:.2f}% | Internal baseline |")
    if section.count(old) != 1:
        raise RuntimeError("Expected exactly one SHD Our SNN placeholder row.")
    path.write_text(text[:start] + section.replace(old, new, 1) + text[end:], encoding="utf-8")


def main():
    from tonic.datasets import SHD

    config_path = ROOT / "configs" / "shd_snn_multiseed.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"] or config["qsnn_enabled"]:
        raise RuntimeError("SHD campaign is restricted to clean SNN training.")
    set_determinism(config["seeds"][0])
    device = select_device(config["device"])
    print_runtime_configuration(device)
    print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)
    print(f"representation={config['representation']} architecture={config['architecture']}", flush=True)
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    train_h5 = ROOT / config["data_root"] / "SHD" / "shd_train.h5"
    train_labels = labels_from_h5(train_h5)
    train_ids, validation_ids = stratified_split(train_labels, config["validation_fraction"], config["split_seed"])
    split_path = results_dir / "shd_snn_multiseed_split.json"
    split = {"dataset": "SHD", "source_partition": "official training", "split_seed": config["split_seed"],
             "official_training_size": len(train_labels), "train_indices": train_ids.tolist(),
             "validation_indices": validation_ids.tolist(),
             "train_class_counts": np.bincount(train_labels[train_ids], minlength=20).tolist(),
             "validation_class_counts": np.bincount(train_labels[validation_ids], minlength=20).tolist(),
             "train_h5_sha256": sha256(train_h5), "official_test_instantiated_during_training": False}
    write_json(split_path, split)
    train_dataset = SHD(save_to=str(ROOT / config["data_root"]), train=True)
    train_frames, loaded_labels, preprocessing = preprocess_partition(train_dataset, config, "official_training")
    if not np.array_equal(train_labels, loaded_labels):
        raise RuntimeError("Tonic SHD labels disagree with the source HDF5 labels.")
    write_json(results_dir / "shd_snn_preprocessing.json", preprocessing)
    validations = [train_one(config, train_dataset, train_frames, train_labels, train_ids,
                             validation_ids, split_path, seed) for seed in config["seeds"]]
    del train_dataset, train_frames
    print("[FINAL TEST] all five checkpoints frozen; loading official SHD test partition", flush=True)
    test_dataset = SHD(save_to=str(ROOT / config["data_root"]), train=False)
    test_frames, test_labels, test_preprocessing = preprocess_partition(test_dataset, config, "official_test")
    test_preprocessing["test_h5_sha256"] = sha256(ROOT / config["data_root"] / "SHD" / "shd_test.h5")
    write_json(results_dir / "shd_snn_test_preprocessing.json", test_preprocessing)
    final_results = [test_one(config, test_frames, test_labels, validation, seed)
                     for seed, validation in zip(config["seeds"], validations)]
    summary = aggregate(final_results, config["seeds"])
    write_aggregate_files(summary, final_results)
    update_readme(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
