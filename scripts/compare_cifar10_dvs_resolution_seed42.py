"""Compare 128x128 and 64x64 CIFAR10-DVS event-frame SNN baselines."""
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cifar10_dvs.snn_baseline import (
    environment_metadata, sha256, train, write_history, write_json,
)


def main():
    config_path = ROOT / "configs" / "cifar10_dvs_resolution_comparison_seed42.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"] or config["held_out_enabled"]:
        raise RuntimeError("Resolution comparison is restricted to clean validation training.")
    split_path = ROOT / "results" / "cifar10_dvs_snn_seed42_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_hash = "629571c70b6202629206d7a449a41e02f124efba2cf469f966c19ea4cdf288b8"
    if split.get("split_hash") != expected_hash:
        raise RuntimeError("Frozen CIFAR10-DVS split hash changed; refusing to alter the split.")
    train_ids = np.asarray(split["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(split["validation_indices"], dtype=np.int64)
    if len(train_ids) != 8000 or len(validation_ids) != 1000:
        raise RuntimeError("Frozen split has unexpected train/validation sizes.")

    from tonic.datasets import CIFAR10DVS
    dataset = CIFAR10DVS(save_to=str(ROOT / config["data_root"]))
    results_dir = ROOT / "results"
    checkpoints_dir = ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    summaries = {}
    for resolution, resolution_config in config["resolutions"].items():
        print(f"\nRUN {resolution}", flush=True)
        run_config = dict(config)
        run_config.update({
            "max_epochs": int(resolution_config["max_epochs"]),
            "early_stop_patience": int(resolution_config["early_stop_patience"]),
        })
        checkpoint_path = checkpoints_dir / f"cifar10_dvs_resolution_{resolution}_seed42_best.pt"
        model, result = train(
            run_config, dataset, train_ids, validation_ids, checkpoint_path,
            batch_size=int(config["batch_size"]),
            sensor_size=tuple(resolution_config["sensor_size"]),
            downsample_factor=int(resolution_config["downsample_factor"]),
            cache_frames=True,
        )
        cache_runtime = result["cache_preprocessing_runtime_seconds"]
        history = result.pop("history")
        lr_history = result.pop("lr_history")
        history_path = results_dir / f"cifar10_dvs_resolution_{resolution}_seed42_history.csv"
        lr_path = results_dir / f"cifar10_dvs_resolution_{resolution}_seed42_lr_history.csv"
        write_history(history_path, history)
        lr_path.write_text(
            "epoch,old_lr,new_lr\n" + "\n".join(
                f"{row.get('epoch', '')},{row.get('old_lr', '')},"
                f"{row.get('new_lr', row.get('lr', ''))}" for row in lr_history
            ) + "\n", encoding="utf-8"
        )
        runtimes = np.asarray([row["epoch_runtime_seconds"] for row in history], dtype=float)
        summary = {
            "resolution": resolution,
            "sensor_size": resolution_config["sensor_size"],
            "downsample_factor": resolution_config["downsample_factor"],
            "max_epochs_requested": resolution_config["max_epochs"],
            "stopping_epoch": result["stopping_epoch"],
            "stopping_reason": result["stopping_reason"],
            "best_epoch": result["best_epoch"],
            "best_validation": result["best_validation"],
            "parameters": result["parameters"],
            "mean_epoch_runtime_seconds": float(runtimes.mean()),
            "median_epoch_runtime_seconds": float(np.median(runtimes)),
            "total_training_runtime_seconds": result["training_runtime_seconds"],
            "cache_preprocessing_runtime_seconds": cache_runtime,
            "mean_throughput_samples_per_second": float(
                np.mean([row["throughput_samples_per_second"] for row in history])
            ),
            "peak_gpu_allocated_gb": result["peak_gpu_allocated_gb"],
            "peak_gpu_reserved_gb": result["peak_gpu_reserved_gb"],
            "scheduler_steps": result["scheduler_steps"],
            "scheduler_step_on": run_config["scheduler"]["step_on"],
            "final_lr": result["final_lr"],
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint_path),
            "history": str(history_path.relative_to(ROOT)),
            "lr_history": str(lr_path.relative_to(ROOT)),
        }
        summaries[resolution] = summary
        print(
            f"RUN {resolution} COMPLETE: best_epoch={summary['best_epoch']} "
            f"best_validation_accuracy={summary['best_validation']['accuracy']:.4f} "
            f"mean_epoch_runtime={summary['mean_epoch_runtime_seconds']:.1f}s "
            f"mean_throughput={summary['mean_throughput_samples_per_second']:.1f}samples/s "
            f"peak_GPU_memory={summary['peak_gpu_allocated_gb']:.3f}GB "
            f"scheduler_steps={summary['scheduler_steps']}",
            flush=True,
        )
        del model, history, lr_history, runtimes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a = summaries["128x128"]
    b = summaries["64x64"]
    speedup = a["mean_epoch_runtime_seconds"] / b["mean_epoch_runtime_seconds"]
    validation_difference_pp = 100.0 * (
        b["best_validation"]["accuracy"] - a["best_validation"]["accuracy"]
    )
    accept = speedup >= 1.25 and validation_difference_pp > -2.0
    verdict = "CIFAR10-DVS 64x64: ACCEPT" if accept else "CIFAR10-DVS 64x64: REJECT"
    result = {
        "status": "COMPLETE",
        "dataset": "CIFAR10-DVS",
        "seed": config["model_seed"],
        "split": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256(split_path),
        "held_out_test_accessed": False,
        "existing_128x128_result_reused": False,
        "existing_128x128_reason": (
            "The prior 128x128 run was interrupted and its checkpoint metadata was not compatible "
            "with the observed run log; both resolutions were rerun under one cached protocol."
        ),
        "selection_rule": "64x64 requires >=1.25x mean epoch speedup and <2.0 percentage-point validation drop",
        "128x128": a,
        "64x64": b,
        "speedup": speedup,
        "validation_difference_percentage_points": validation_difference_pp,
        "accepted": accept,
        "environment": environment_metadata(),
    }
    report_path = results_dir / "cifar10_dvs_resolution_comparison_seed42.json"
    write_json(report_path, result)

    print("\nCIFAR10-DVS RESOLUTION CHECK", flush=True)
    print("\n128x128:", flush=True)
    print(f"epoch runtime: {a['mean_epoch_runtime_seconds']:.1f}s mean", flush=True)
    print(f"best validation accuracy: {a['best_validation']['accuracy']:.4f}", flush=True)
    print(f"best validation loss: {a['best_validation']['loss']:.4f}", flush=True)
    print(f"peak GPU memory: {a['peak_gpu_allocated_gb']:.3f}GB allocated", flush=True)
    print(f"parameter count: {a['parameters']}", flush=True)
    print(f"throughput: {a['mean_throughput_samples_per_second']:.1f} samples/s", flush=True)
    print("\n64x64:", flush=True)
    print(f"epoch runtime: {b['mean_epoch_runtime_seconds']:.1f}s mean", flush=True)
    print(f"best validation accuracy: {b['best_validation']['accuracy']:.4f}", flush=True)
    print(f"best validation loss: {b['best_validation']['loss']:.4f}", flush=True)
    print(f"peak GPU memory: {b['peak_gpu_allocated_gb']:.3f}GB allocated", flush=True)
    print(f"parameter count: {b['parameters']}", flush=True)
    print(f"throughput: {b['mean_throughput_samples_per_second']:.1f} samples/s", flush=True)
    print(f"\nspeedup: {speedup:.2f}x", flush=True)
    print(f"validation difference: {validation_difference_pp:+.2f} percentage points", flush=True)
    print(f"scheduler.step(validation_accuracy) calls: {b['scheduler_steps']}", flush=True)
    print(f"\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
