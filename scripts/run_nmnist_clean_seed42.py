"""Download N-MNIST as needed and train one clean temporal QSNN baseline."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.data import load_nmnist_development
from experiments.nmnist.training import train_clean
from models.nmnist_qsnn import NMNISTTemporalQSNN


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config_path = ROOT / "configs" / "nmnist_clean_seed42.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"] or config["held_out_enabled"]:
        raise RuntimeError("Initial N-MNIST benchmark must remain clean and development-only.")
    model = NMNISTTemporalQSNN(
        config["n_qubits"], config["temporal_reupload_steps"], config["n_classes"]
    )
    print("[REPRESENTATION] original_event_shape=variable_Nx4 fields=x,y,t,p sensor=34x34x2", flush=True)
    print(
        f"[REPRESENTATION] channels={config['event_channels']} qubits={config['n_qubits']} "
        f"encoding=ordered_temporal_RY circuit_depth={model.circuit_depth()} "
        f"parameters={model.trainable_parameter_count()} temporal_bins={config['temporal_bins']} "
        f"polarity_preserved={config['polarity_preserved']}", flush=True,
    )
    train, validation, train_ids, validation_ids, official_train_size = load_nmnist_development(config)
    train_x, train_y, train_counts, train_durations = train
    val_x, val_y, val_counts, val_durations = validation
    results_dir = ROOT / "results"
    checkpoints_dir = ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    split_path = results_dir / "nmnist_clean_seed42_split_indices.json"
    split_path.write_text(json.dumps({
        "source_partition": "official training", "official_training_size": official_train_size,
        "split_seed": config["split_seed"], "train_indices": train_ids.tolist(),
        "validation_indices": validation_ids.tolist(), "held_out_partition_instantiated": False,
    }, indent=2), encoding="utf-8")
    metadata_path = results_dir / "nmnist_clean_seed42_preprocessing.json"
    metadata_path.write_text(json.dumps({
        "input_event_fields": ["x", "y", "t", "p"], "sensor_size": config["sensor_size"],
        "spatial_grid": config["spatial_grid"], "polarity_preserved": True,
        "temporal_bins": config["temporal_bins"], "temporal_order_preserved": True,
        "normalization": "per-sample log1p count divided by log1p sample maximum bin-channel count",
        "train_event_count_summary": {
            "minimum": int(train_counts.min()), "median": float(np.median(train_counts)),
            "maximum": int(train_counts.max()),
        },
        "validation_event_count_summary": {
            "minimum": int(val_counts.min()), "median": float(np.median(val_counts)),
            "maximum": int(val_counts.max()),
        },
        "train_duration_us_summary": {
            "minimum": int(train_durations.min()), "median": float(np.median(train_durations)),
            "maximum": int(train_durations.max()),
        },
        "validation_duration_us_summary": {
            "minimum": int(val_durations.min()), "median": float(np.median(val_durations)),
            "maximum": int(val_durations.max()),
        },
        "learned_preprocessing": False, "static_frames_created": False, "pca_used": False,
    }, indent=2), encoding="utf-8")
    checkpoint_path = checkpoints_dir / "nmnist_temporal_qsnn_seed42.pt"
    run = train_clean(config, train_x, train_y, val_x, val_y, checkpoint_path)
    history = run.pop("history")
    run.update({
        "dataset": "N-MNIST", "model_seed": config["model_seed"],
        "split_seed": config["split_seed"], "train_samples": len(train_ids),
        "validation_samples": len(validation_ids), "event_channels": config["event_channels"],
        "temporal_bins": config["temporal_bins"], "polarity_preserved": True,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "split_indices": str(split_path.relative_to(ROOT)),
        "preprocessing_metadata": str(metadata_path.relative_to(ROOT)),
        "held_out_accessed": False, "attacks_run": False, "defenses_run": False,
    })
    thresholds_pass = (
        run["validation"]["accuracy"] >= config["acceptable_validation_accuracy"]
        and run["validation"]["macro_f1"] >= config["acceptable_validation_macro_f1"]
        and min(run["validation"]["per_class_accuracy"].values())
        >= config["acceptable_minimum_class_accuracy"]
    )
    run["classification"] = "ACCEPTABLE" if thresholds_pass else "NEEDS_IMPROVEMENT"
    result_path = results_dir / "nmnist_clean_seed42.json"
    result_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    with (results_dir / "nmnist_clean_seed42_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (results_dir / "nmnist_clean_seed42_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "best_epoch", "stopping_epoch", "validation_accuracy", "validation_macro_f1",
            "validation_ce", "runtime_seconds", "convergence_established", "parameters", "qubits",
            "checkpoint_sha256", "classification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "best_epoch": run["best_epoch"], "stopping_epoch": run["stopping_epoch"],
            "validation_accuracy": run["validation"]["accuracy"],
            "validation_macro_f1": run["validation"]["macro_f1"],
            "validation_ce": run["validation"]["loss"], "runtime_seconds": run["runtime_seconds"],
            "convergence_established": run["convergence_established"],
            "parameters": run["parameters"], "qubits": run["qubits"],
            "checkpoint_sha256": run["checkpoint_sha256"], "classification": run["classification"],
        })
    print(json.dumps(run, indent=2), flush=True)
    print(f"N-MNIST CLEAN BASELINE: {run['classification']}", flush=True)


if __name__ == "__main__":
    main()
