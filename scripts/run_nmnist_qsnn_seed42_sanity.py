"""Train the fixed seed-42 N-MNIST QSNN on the frozen SNN development split."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.data import events_to_temporal_channels
from models.nmnist_qsnn import NMNISTCudaQSNN


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "total_bytes": int(total), "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()), "free_bytes": int(free),
    }


def evaluate(model, features, labels, batch_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start:start + batch_size]).to("cuda")
            y = torch.from_numpy(labels[start:start + batch_size]).to("cuda")
            quantum = model.quantum_features(x)
            if not torch.isfinite(quantum).all():
                raise RuntimeError("Non-finite quantum output during evaluation.")
            logits = model.head(quantum)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            total_correct += int((logits.argmax(1) == y).sum())
    return total_loss / len(features), total_correct / len(features)


def main():
    config_path = ROOT / "configs" / "nmnist_qsnn_seed42_sanity.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["seed"] != 42 or any((config["attacks_enabled"], config["defenses_enabled"],
                                    config["architecture_search"], config["official_test_enabled"])):
        raise RuntimeError("This runner is fixed to clean seed-42 validation only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited for this run.")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    initial_memory = gpu_memory()
    runtime = {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
        "quantum_backend": config["backend"], "selected_device": "cuda:0", **initial_memory,
    }
    for key, value in runtime.items():
        print(f"[RUNTIME] {key}={value}", flush=True)

    manifest_path = ROOT / config["split_manifest"]
    if sha256(manifest_path) != config["split_sha256"]:
        raise RuntimeError("Frozen SNN split manifest hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if manifest["source_partition"] != "official training" or manifest["split_seed"] != 42:
        raise RuntimeError("Unexpected split protocol.")
    if len(train_ids) != 55000 or len(validation_ids) != 5000 or np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Frozen split size or disjointness check failed.")

    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / config["data_root"]), train=True)
    if len(dataset) != manifest["official_training_size"]:
        raise RuntimeError("Official N-MNIST training size does not match the manifest.")
    selected = np.concatenate((train_ids, validation_ids))
    features = np.empty((len(selected), config["temporal_bins"], 8), dtype=np.float32)
    labels = np.empty(len(selected), dtype=np.int64)
    event_counts = np.empty(len(selected), dtype=np.int64)
    durations = np.empty(len(selected), dtype=np.int64)
    print(f"[DATA] reducing {len(selected)} official-training samples from frozen manifest", flush=True)
    for position, index in enumerate(selected):
        events, label = dataset[int(index)]
        features[position] = events_to_temporal_channels(
            events, config["temporal_bins"], config["spatial_grid"], config["sensor_size"][:2]
        )
        labels[position] = int(label)
        event_counts[position] = len(events)
        durations[position] = int(events["t"][-1] - events["t"][0]) if len(events) else 0
        if (position + 1) % 5000 == 0:
            print(f"[DATA] reduced={position + 1}/{len(selected)}", flush=True)
    train_x, validation_x = features[:len(train_ids)], features[len(train_ids):]
    train_y, validation_y = labels[:len(train_ids)], labels[len(train_ids):]

    results_dir = ROOT / "results"
    checkpoints_dir = ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    config_snapshot = results_dir / "nmnist_qsnn_seed42_sanity_config.json"
    adapter_path = results_dir / "nmnist_qsnn_seed42_sanity_input_adapter.json"
    runtime_path = results_dir / "nmnist_qsnn_seed42_sanity_runtime.json"
    config_snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")
    adapter = {
        "original_representation": "variable-length native N-MNIST events with fields x,y,t,p on 34x34 sensor",
        "reduced_representation": "per-sample log-normalized event counts in ordered time x spatial-cell x polarity bins",
        "temporal_bins": 4, "spatial_grid": [2, 2], "polarity_handling": "preserved as separate p=0 and p=1 channels",
        "reduced_shape": [4, 8], "final_feature_dimension": 32,
        "temporal_policy": "equal-width bins over each sample observed timestamp range",
        "channel_order": "((cell_y * 2 + cell_x) * 2 + polarity)",
        "normalization": config["normalization"], "learned": False, "static_mnist_image_created": False,
        "source_split": config["split_manifest"], "source_split_sha256": config["split_sha256"],
        "train_samples": len(train_ids), "validation_samples": len(validation_ids),
        "event_count_summary": {"min": int(event_counts.min()), "median": float(np.median(event_counts)), "max": int(event_counts.max())},
        "duration_us_summary": {"min": int(durations.min()), "median": float(np.median(durations)), "max": int(durations.max())},
        "official_test_instantiated": False,
    }
    adapter_path.write_text(json.dumps(adapter, indent=2), encoding="utf-8")

    model = NMNISTCudaQSNN(config["n_qubits"], config["n_blocks"], 10).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_accuracy = -1.0
    best_loss = math.inf
    best_epoch = 0
    checkpoint_path = checkpoints_dir / "nmnist_qsnn_seed42_sanity_best.pt"
    history = []
    gradients_finite = True
    quantum_outputs_finite = True
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        generator = torch.Generator().manual_seed(42 + epoch)
        order = torch.randperm(len(train_x), generator=generator).numpy()
        loss_sum = 0.0
        train_correct = 0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            x = torch.from_numpy(train_x[indices]).to("cuda")
            y = torch.from_numpy(train_y[indices]).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            quantum = model.quantum_features(x)
            if not torch.isfinite(quantum).all():
                quantum_outputs_finite = False
                raise RuntimeError(f"Non-finite quantum output at epoch {epoch}.")
            logits = model.head(quantum)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                gradients_finite = False
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            loss_sum += float(loss) * len(indices)
            train_correct += int((logits.argmax(1) == y).sum())
        validation_loss, validation_accuracy = evaluate(model, validation_x, validation_y, config["batch_size"])
        train_loss = loss_sum / len(train_x)
        train_accuracy = train_correct / len(train_x)
        improved = validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_loss
        )
        if improved:
            best_accuracy, best_loss, best_epoch = validation_accuracy, validation_loss, epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "validation_accuracy": validation_accuracy, "validation_loss": validation_loss,
                        "config": config}, checkpoint_path)
        epoch_runtime = time.perf_counter() - epoch_started
        memory = gpu_memory()
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_accuracy, "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_runtime_seconds": epoch_runtime, "gpu_allocated_bytes": memory["allocated_bytes"],
            "gpu_reserved_bytes": memory["reserved_bytes"],
        }
        history.append(row)
        print("[TRAIN] " + " ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                                      for key, value in row.items()), flush=True)

    elapsed = time.perf_counter() - started
    checkpoint_hash = sha256(checkpoint_path)
    loss_decreased = min(row["train_loss"] for row in history[1:]) < history[0]["train_loss"] if len(history) > 1 else False
    stable = gradients_finite and quantum_outputs_finite and all(math.isfinite(row["train_loss"]) for row in history)
    pass_sanity = stable and loss_decreased and best_accuracy >= 0.15
    peak_memory = int(torch.cuda.max_memory_allocated())
    runtime.update({"runtime_seconds": elapsed, "peak_gpu_allocated_bytes": peak_memory})
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    history_path = results_dir / "nmnist_qsnn_seed42_sanity_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    result = {
        "experiment": "N-MNIST QSNN SEED42 SANITY", "seed": 42,
        "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss, "best_epoch": best_epoch,
        "runtime_seconds": elapsed, "peak_gpu_memory_bytes": peak_memory,
        "trainable_parameters": model.trainable_parameter_count(), "training_stable": stable,
        "loss_decreased": loss_decreased, "clearly_above_random": best_accuracy >= 0.15,
        "verdict": "PASS" if pass_sanity else "NEEDS IMPROVEMENT",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
        "config": str(config_snapshot.relative_to(ROOT)), "input_adapter": str(adapter_path.relative_to(ROOT)),
        "history": str(history_path.relative_to(ROOT)), "runtime_info": str(runtime_path.relative_to(ROOT)),
        "split_manifest": config["split_manifest"], "split_sha256": config["split_sha256"],
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False,
    }
    (results_dir / "nmnist_qsnn_seed42_sanity_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nN-MNIST QSNN SEED42 SANITY", flush=True)
    print(f"Original input: {adapter['original_representation']}", flush=True)
    print(f"Reduced input: shape 4x8, {adapter['reduced_representation']}", flush=True)
    print("Qubits: 8\nVariational blocks: 4", flush=True)
    print(f"Trainable parameters: {model.trainable_parameter_count()}", flush=True)
    print(f"Backend: {config['backend']}\nDevice: cuda:0", flush=True)
    print(f"Best validation accuracy: {best_accuracy:.6f}\nBest validation loss: {best_loss:.6f}", flush=True)
    print(f"Best epoch: {best_epoch}\nRuntime: {elapsed:.2f} seconds", flush=True)
    print(f"Peak GPU memory: {peak_memory} bytes\nTraining stable: {'YES' if stable else 'NO'}", flush=True)
    print(f"\nN-MNIST QSNN SANITY: {result['verdict']}", flush=True)


if __name__ == "__main__":
    main()
