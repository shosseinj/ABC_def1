"""Train clean CIFAR10-DVS convolutional SNN (seed 42) and evaluate held-out test once."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cifar10_dvs.snn_baseline import (
    FramedCIFAR10DVS, environment_metadata, evaluate, make_loader, sha256,
    stratified_split_indices, train, write_history, write_json,
)


def main():
    from tonic.datasets import CIFAR10DVS

    config_path = ROOT / "configs" / "cifar10_dvs_snn_seed42.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["attacks_enabled"] or config["defenses_enabled"] or config["held_out_enabled"]:
        raise RuntimeError("This command is restricted to clean SNN training.")
    dataset = CIFAR10DVS(save_to=str(ROOT / config["data_root"]))
    targets = np.asarray(dataset.targets, dtype=int)
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    split_path = results_dir / "cifar10_dvs_snn_seed42_split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = np.asarray(split["train_indices"], dtype=np.int64)
        validation_ids = np.asarray(split["validation_indices"], dtype=np.int64)
        test_ids = np.asarray(split["test_indices"], dtype=np.int64)
        print(f"reusing frozen split: {split_path} hash={split['split_hash']}", flush=True)
    else:
        train_ids, validation_ids, test_ids = stratified_split_indices(
            targets, config["split_seed"], config["train_split"],
            config["validation_split"], config["test_split"])
        split_payload = {"dataset": "CIFAR10-DVS", "n_samples": len(dataset),
                         "split_seed": config["split_seed"],
                         "train_indices": train_ids.tolist(),
                         "validation_indices": validation_ids.tolist(),
                         "test_indices": test_ids.tolist(),
                         "train_class_counts": np.bincount(targets[train_ids], minlength=10).tolist(),
                         "validation_class_counts": np.bincount(targets[validation_ids],
                                                               minlength=10).tolist(),
                         "test_class_counts": np.bincount(targets[test_ids], minlength=10).tolist()}
        split_payload["split_hash"] = sha256_bytes(json.dumps(split_payload, sort_keys=True).encode())
        write_json(split_path, split_payload)
        split = split_payload
        print(f"saved frozen split: {split_path} hash={split['split_hash']}", flush=True)

    print(f"input={config['representation']} temporal_bins={config['temporal_bins']} "
          f"tensor=[T={config['temporal_bins']},2,128,128] polarity=2ch "
          f"accumulation={config['accumulation']} normalization={config['normalization']}",
          flush=True)
    checkpoint_path = checkpoints_dir / "cifar10_dvs_snn_seed42_best.pt"

    batch_size = int(config["batch_size"])
    batch_reductions = []
    model_result = None
    while True:
        try:
            _, model_result = train(config, dataset, train_ids, validation_ids,
                                    checkpoint_path, batch_size)
            break
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_size <= 2:
                raise
            old = batch_size
            batch_size = max(2, batch_size // 2)
            batch_reductions.append({"from": old, "to": batch_size})
            print(f"CUDA OOM: reducing this experiment batch size {old} -> {batch_size}",
                  flush=True)
    history = model_result.pop("history")
    device = torch.device(config["device"])

    # Final held-out evaluation exactly once on the frozen best checkpoint.
    from torch.utils.data import Subset
    test_set = Subset(FramedCIFAR10DVS(dataset, config["temporal_bins"]), test_ids.tolist())
    test_loader = make_loader(test_set, batch_size, False, config["model_seed"])
    from models.cifar10_dvs_snn import CIFAR10DVSConvSNN
    best_model = CIFAR10DVSConvSNN().to(device)
    frozen = torch.load(checkpoint_path, map_location=device, weights_only=True)
    best_model.load_state_dict(frozen["model_state"])
    test_metrics = evaluate(best_model, test_loader, device, True)

    result = {
        "status": "COMPLETE", "dataset": "CIFAR10-DVS", "seed": config["model_seed"],
        "train_samples": len(train_ids), "validation_samples": len(validation_ids),
        "test_samples": len(test_ids), "batch_size_used": batch_size,
        "batch_size_reductions": batch_reductions,
        "best_epoch": model_result["best_epoch"], "stopping_epoch": model_result["stopping_epoch"],
        "stopping_reason": model_result["stopping_reason"],
        "validation": model_result["best_validation"], "test": test_metrics,
        "parameters": model_result["parameters"],
        "training_runtime_seconds": model_result["training_runtime_seconds"],
        "lr_history": model_result["lr_history"], "final_lr": model_result["final_lr"],
        "scheduler_steps": model_result["scheduler_steps"],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "split": str(split_path.relative_to(ROOT)),
        "split_sha256": sha256(split_path),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "environment": environment_metadata(),
        "attacks_run": False, "defenses_run": False,
        "test_tuning_performed": False,
    }
    write_history(results_dir / "cifar10_dvs_snn_seed42_history.csv", history)
    (results_dir / "cifar10_dvs_snn_seed42_lr_history.csv").write_text(
        "epoch,old_lr,new_lr\n" + "\n".join(
            f"{r.get('epoch', '')},{r.get('old_lr', '')},{r.get('new_lr', r.get('lr', ''))}"
            for r in model_result["lr_history"]) + "\n", encoding="utf-8")
    write_json(results_dir / "cifar10_dvs_snn_seed42_result.json", result)
    print(json.dumps({k: result[k] for k in (
        "best_epoch", "stopping_epoch", "parameters", "final_lr",
        "training_runtime_seconds", "batch_size_used")}, indent=2), flush=True)
    print(f"CIFAR10-DVS SNN CLEAN BASELINE", flush=True)
    print(f"Best validation accuracy: {result['validation']['accuracy']:.4f}", flush=True)
    print(f"Final test accuracy: {result['test']['accuracy']:.4f}", flush=True)
    print(f"Macro-F1: {result['test']['macro_f1']:.4f}", flush=True)
    print(f"Best epoch: {result['best_epoch']}", flush=True)
    print(f"Stopping epoch: {result['stopping_epoch']}", flush=True)
    print(f"Parameters: {result['parameters']}", flush=True)
    print(f"LR reductions: {len(result['lr_history']) - 1}", flush=True)
    print(f"Final LR: {result['final_lr']}", flush=True)
    print(f"Runtime: {result['training_runtime_seconds']:.1f}s", flush=True)
    print("`CIFAR10-DVS SNN CLEAN BASELINE: COMPLETED`", flush=True)


def sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
