"""Run the prespecified seed-42 N-MNIST QSNN ablations sequentially on CUDA."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
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

SPLIT_SHA256 = "a12176ce117ab9a85dd29d277901f8617ad2b5427dd3f976dbba436942f70198"
RESULTS = ROOT / "results" / "nmnist_qsnn_ablation"
CHECKPOINTS = ROOT / "checkpoints" / "nmnist_qsnn_ablation"
BASELINE_RESULT = ROOT / "results" / "nmnist_qsnn_seed42_sanity_result.json"
BASELINE_HISTORY = ROOT / "results" / "nmnist_qsnn_seed42_sanity_history.csv"
BASELINE_CHECKPOINT = ROOT / "checkpoints" / "nmnist_qsnn_seed42_sanity_best.pt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_record():
    free, total = torch.cuda.mem_get_info()
    return {
        "total_bytes": int(total), "free_bytes": int(free),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def base_config(run, temporal_bins, channels, qubits=8, blocks=4):
    return {
        "run": run, "dataset": "N-MNIST", "data_root": "data/nmnist",
        "split_manifest": "results/nmnist_snn_multiseed_split.json",
        "split_sha256": SPLIT_SHA256, "seed": 42,
        "original_representation": "variable-length native events x,y,t,p on a 34x34 sensor",
        "temporal_bins": temporal_bins, "spatial_polarity_channels": channels,
        "spatial_grid": [2, channels // 4], "polarity_channels": 2,
        "input_features": temporal_bins * channels,
        "normalization": "per-sample log1p count divided by log1p sample maximum bin-channel count",
        "n_qubits": qubits, "n_blocks": blocks, "n_classes": 10,
        "encoding": "temporal-major angle encoding; contiguous time groups assigned to blocks; channels assigned modulo qubits",
        "data_reuploading": True, "variational_gates": ["RY", "RZ"],
        "entanglement": "CNOT ring", "measurements": "local Pauli-Z expectations",
        "output_head": f"Linear({qubits},10)", "optimizer": "Adam",
        "learning_rate": 0.003, "weight_decay": 0.0, "batch_size": 256,
        "epochs": 15, "gradient_clip_norm": 1.0,
        "checkpoint_selection": "highest validation accuracy, then lowest validation loss",
        "backend": "torch.cuda batched analytic state-vector", "device": "cuda:0",
        "attacks_enabled": False, "defenses_enabled": False,
        "official_test_enabled": False, "architecture_search_beyond_protocol": False,
    }


def better(left, right):
    left_key = (-left["best_validation_accuracy"], left["best_validation_loss"],
                left["params"], left["input_features"])
    right_key = (-right["best_validation_accuracy"], right["best_validation_loss"],
                 right["params"], right["input_features"])
    return left if left_key < right_key else right


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
                raise RuntimeError("Non-finite quantum outputs during validation.")
            logits = model.head(quantum)
            total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
            total_correct += int((logits.argmax(1) == y).sum())
    return total_loss / len(features), total_correct / len(features)


def train_run(config, train_x, train_y, validation_x, validation_y):
    run = config["run"]
    print(f"\nRUN: {run}", flush=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = NMNISTCudaQSNN(
        config["n_qubits"], config["n_blocks"], 10,
        config["temporal_bins"], config["spatial_polarity_channels"],
    ).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.0)
    best_accuracy = -1.0
    best_loss = math.inf
    best_epoch = 0
    checkpoint_path = CHECKPOINTS / f"{run}_best.pt"
    history = []
    stable = True
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(42 + epoch)).numpy()
        total_loss = 0.0
        total_correct = 0
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            x = torch.from_numpy(train_x[indices]).to("cuda")
            y = torch.from_numpy(train_y[indices]).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            quantum = model.quantum_features(x)
            if not torch.isfinite(quantum).all():
                stable = False
                raise RuntimeError(f"Non-finite quantum outputs in {run}, epoch {epoch}.")
            logits = model.head(quantum)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                stable = False
                raise RuntimeError(f"Non-finite loss in {run}, epoch {epoch}.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                stable = False
                raise RuntimeError(f"Non-finite gradients in {run}, epoch {epoch}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss) * len(indices)
            total_correct += int((logits.argmax(1) == y).sum())
        validation_loss, validation_accuracy = evaluate(
            model, validation_x, validation_y, config["batch_size"]
        )
        train_loss = total_loss / len(train_x)
        train_accuracy = total_correct / len(train_x)
        if validation_accuracy > best_accuracy or (
                validation_accuracy == best_accuracy and validation_loss < best_loss):
            best_accuracy, best_loss, best_epoch = validation_accuracy, validation_loss, epoch
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch,
                "validation_accuracy": validation_accuracy, "validation_loss": validation_loss,
                "config": config,
            }, checkpoint_path)
        epoch_runtime = time.perf_counter() - epoch_started
        gpu_memory = int(torch.cuda.memory_allocated())
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": best_accuracy, "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_runtime_seconds": epoch_runtime, "gpu_memory_bytes": gpu_memory,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.6f} "
            f"validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.6f} "
            f"best_validation_accuracy={best_accuracy:.6f} learning_rate={optimizer.param_groups[0]['lr']:.6f} "
            f"epoch_runtime={epoch_runtime:.3f} GPU_memory={gpu_memory}", flush=True,
        )
    runtime = time.perf_counter() - started
    result = {
        "run": run, "temporal_bins": config["temporal_bins"],
        "spatial_polarity_channels": config["spatial_polarity_channels"],
        "qubits": config["n_qubits"], "blocks": config["n_blocks"],
        "input_features": config["input_features"], "params": model.trainable_parameter_count(),
        "best_validation_accuracy": best_accuracy, "best_validation_loss": best_loss,
        "best_epoch": best_epoch, "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "training_stable": stable, "convergence": history[-1]["train_loss"] < history[0]["train_loss"],
        "final_training_loss": history[-1]["train_loss"],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint_path),
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False,
        "reused": False,
    }
    write_run_artifacts(config, result, history)
    return result


def write_run_artifacts(config, result, history):
    run = config["run"]
    (RESULTS / f"{run}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULTS / f"{run}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (RESULTS / f"{run}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def reuse_baseline():
    source = json.loads(BASELINE_RESULT.read_text(encoding="utf-8"))
    if source["split_sha256"] != SPLIT_SHA256 or sha256(BASELINE_CHECKPOINT) != source["checkpoint_sha256"]:
        raise RuntimeError("The baseline result is not fully compatible or its checkpoint hash changed.")
    config = base_config("T4_S8_Q8_B4", 4, 8)
    destination = CHECKPOINTS / "T4_S8_Q8_B4_best.pt"
    shutil.copy2(BASELINE_CHECKPOINT, destination)
    with BASELINE_HISTORY.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    normalized_history = [{key: float(value) if key != "epoch" else int(value)
                           for key, value in row.items()} for row in history]
    result = {
        "run": config["run"], "temporal_bins": 4, "spatial_polarity_channels": 8,
        "qubits": 8, "blocks": 4, "input_features": 32, "params": 154,
        "best_validation_accuracy": source["best_validation_accuracy"],
        "best_validation_loss": source["best_validation_loss"], "best_epoch": source["best_epoch"],
        "runtime_seconds": source["runtime_seconds"], "peak_gpu_memory_bytes": source["peak_gpu_memory_bytes"],
        "training_stable": source["training_stable"], "convergence": source["loss_decreased"],
        "final_training_loss": normalized_history[-1]["train_loss"],
        "checkpoint": str(destination.relative_to(ROOT)), "checkpoint_sha256": sha256(destination),
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False,
        "reused": True, "reused_from": str(BASELINE_RESULT.relative_to(ROOT)),
    }
    write_run_artifacts(config, result, normalized_history)
    return result


def make_features(dataset, indices, temporal_bins, channels, label):
    grid = (2, channels // 4)
    features = np.empty((len(indices), temporal_bins, channels), dtype=np.float32)
    labels = np.empty(len(indices), dtype=np.int64)
    print(f"[ADAPTER] {label} bins={temporal_bins} channels={channels} samples={len(indices)}", flush=True)
    for position, index in enumerate(indices):
        events, target = dataset[int(index)]
        features[position] = events_to_temporal_channels(events, temporal_bins, grid, (34, 34))
        labels[position] = int(target)
        if (position + 1) % 10000 == 0:
            print(f"[ADAPTER] {label} reduced={position + 1}/{len(indices)}", flush=True)
    return features, labels


def save_aggregate(rows, temporal_winner, representation_winner, quantum_winner, final):
    columns = [
        "run", "temporal_bins", "spatial_polarity_channels", "qubits", "blocks",
        "input_features", "params", "best_validation_accuracy", "best_validation_loss",
        "best_epoch", "runtime_seconds", "peak_gpu_memory_bytes", "training_stable",
        "convergence", "final_training_loss", "checkpoint_sha256", "reused",
    ]
    csv_path = ROOT / "results" / "nmnist_qsnn_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "study": "N-MNIST QSNN controlled seed-42 validation-only ablation",
        "split_sha256": SPLIT_SHA256, "seed": 42, "runs": rows,
        "temporal_winner": temporal_winner["run"],
        "representation_winner": representation_winner["run"],
        "quantum_capacity_winner": quantum_winner["run"], "final_selected": final["run"],
        "selection_rule": "highest validation accuracy, then lowest validation loss, then lower parameters/input dimension",
        "single_seed_limitation": "Development evidence only; no uncertainty or superiority claim from one seed.",
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False,
    }
    (ROOT / "results" / "nmnist_qsnn_ablation_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    lines = [
        "# N-MNIST QSNN Controlled Ablation", "",
        "Validation-only, seed 42, frozen official-training split. The official test partition was not instantiated.", "",
        "| Run | Temporal Bins | Spatial-Polarity Channels | Qubits | Blocks | Input Features | Params | Best Val Acc | Best Val Loss | Best Epoch | Runtime (s) | Peak GPU Memory (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['temporal_bins']} | {row['spatial_polarity_channels']} | "
            f"{row['qubits']} | {row['blocks']} | {row['input_features']} | {row['params']} | "
            f"{row['best_validation_accuracy']:.4f} | {row['best_validation_loss']:.6f} | "
            f"{row['best_epoch']} | {row['runtime_seconds']:.2f} | {row['peak_gpu_memory_bytes']} |"
        )
    lines.extend([
        "", f"- Temporal winner: `{temporal_winner['run']}`",
        f"- Representation winner: `{representation_winner['run']}`",
        f"- Quantum-capacity winner: `{quantum_winner['run']}`",
        f"- Final selected configuration: `{final['run']}`", "",
        "This single-seed adaptive development study does not establish statistical superiority. "
        "The selected architecture requires prespecified multi-seed validation before any official-test evaluation.",
    ])
    (ROOT / "results" / "nmnist_qsnn_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results" / "nmnist_snn_multiseed_split.json"
    if sha256(manifest_path) != SPLIT_SHA256:
        raise RuntimeError("Frozen split hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if len(train_ids) != 55000 or len(validation_ids) != 5000 or np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Frozen split integrity check failed.")
    runtime = {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
        "backend": "torch.cuda batched analytic state-vector", "selected_device": "cuda:0",
        **memory_record(),
    }
    print("\n".join(f"[RUNTIME] {key}={value}" for key, value in runtime.items()), flush=True)
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data" / "nmnist"), train=True)
    if len(dataset) != 60000:
        raise RuntimeError("Unexpected N-MNIST official-training size.")

    rows = []
    baseline = reuse_baseline()
    rows.append(baseline)
    print("[ABLATION 1/6 COMPLETE] reused fully compatible T4 baseline", flush=True)

    feature_cache = {}
    for position, temporal_bins in enumerate((8, 10), start=2):
        key = (temporal_bins, 8)
        all_x, all_y = make_features(dataset, np.concatenate((train_ids, validation_ids)), *key, f"T{temporal_bins}")
        feature_cache[key] = (all_x[:len(train_ids)], all_y[:len(train_ids)],
                              all_x[len(train_ids):], all_y[len(train_ids):])
        config = base_config(f"T{temporal_bins}_S8_Q8_B4", temporal_bins, 8)
        result = train_run(config, *feature_cache[key])
        rows.append(result)
        print(f"[ABLATION {position}/6 COMPLETE]", flush=True)
    temporal_winner = baseline
    for candidate in rows[1:3]:
        temporal_winner = better(temporal_winner, candidate)
    print(f"TEMPORAL ABLATION WINNER: {temporal_winner['run']}", flush=True)

    temporal_bins = temporal_winner["temporal_bins"]
    key = (temporal_bins, 16)
    all_x, all_y = make_features(dataset, np.concatenate((train_ids, validation_ids)), *key, "S16")
    feature_cache[key] = (all_x[:len(train_ids)], all_y[:len(train_ids)],
                          all_x[len(train_ids):], all_y[len(train_ids):])
    s16 = train_run(base_config(f"T{temporal_bins}_S16_Q8_B4", temporal_bins, 16), *feature_cache[key])
    rows.append(s16)
    print("[ABLATION 4/6 COMPLETE]", flush=True)
    representation_winner = better(temporal_winner, s16)
    print(f"REPRESENTATION ABLATION WINNER: {representation_winner['run']}", flush=True)

    representation_key = (representation_winner["temporal_bins"], representation_winner["spatial_polarity_channels"])
    if representation_key not in feature_cache:
        all_x, all_y = make_features(dataset, np.concatenate((train_ids, validation_ids)),
                                     *representation_key, "representation-winner")
        feature_cache[representation_key] = (
            all_x[:len(train_ids)], all_y[:len(train_ids)], all_x[len(train_ids):], all_y[len(train_ids):]
        )
    q8_b6 = train_run(base_config(
        f"T{representation_key[0]}_S{representation_key[1]}_Q8_B6",
        *representation_key, qubits=8, blocks=6,
    ), *feature_cache[representation_key])
    rows.append(q8_b6)
    print("[ABLATION 5/6 COMPLETE]", flush=True)
    selected_q8 = q8_b6 if (
        q8_b6["best_validation_accuracy"] - representation_winner["best_validation_accuracy"] >= 0.01
    ) else representation_winner
    selected_blocks = selected_q8["blocks"]
    q12 = train_run(base_config(
        f"T{representation_key[0]}_S{representation_key[1]}_Q12_B{selected_blocks}",
        *representation_key, qubits=12, blocks=selected_blocks,
    ), *feature_cache[representation_key])
    rows.append(q12)
    print("[ABLATION 6/6 COMPLETE]", flush=True)
    quantum_winner = better(selected_q8, q12)
    final = rows[0]
    for candidate in rows[1:]:
        final = better(final, candidate)
    save_aggregate(rows, temporal_winner, representation_winner, quantum_winner, final)

    improvement = 100.0 * (final["best_validation_accuracy"] - 0.6956)
    verdict = "IMPROVED" if improvement >= 1.0 else "NO MEANINGFUL IMPROVEMENT"
    print("\nN-MNIST QSNN CONTROLLED ABLATION SUMMARY", flush=True)
    print("\nBaseline:\n4 bins / 8 channels / 8 qubits / 4 blocks\nVal Acc: 69.56%", flush=True)
    print(f"\nTemporal winner:\n{temporal_winner['run']}", flush=True)
    print(f"\nRepresentation winner:\n{representation_winner['run']}", flush=True)
    print(f"\nQuantum-capacity winner:\n{quantum_winner['run']}", flush=True)
    print("\nFINAL SELECTED CONFIG:", flush=True)
    print(f"Temporal bins: {final['temporal_bins']}", flush=True)
    print(f"Spatial-polarity channels: {final['spatial_polarity_channels']}", flush=True)
    print(f"Input features: {final['input_features']}", flush=True)
    print(f"Qubits: {final['qubits']}\nBlocks: {final['blocks']}", flush=True)
    print(f"Trainable parameters: {final['params']}", flush=True)
    print(f"Best validation accuracy: {final['best_validation_accuracy']:.6f}", flush=True)
    print(f"Best validation loss: {final['best_validation_loss']:.6f}", flush=True)
    print(f"Best epoch: {final['best_epoch']}\nRuntime: {final['runtime_seconds']:.2f} seconds", flush=True)
    print(f"Peak GPU memory: {final['peak_gpu_memory_bytes']} bytes", flush=True)
    print(f"\nImprovement over original 69.56%:\n{improvement:+.2f} percentage points", flush=True)
    print(f"\nN-MNIST QSNN ABLATION: {verdict}", flush=True)


if __name__ == "__main__":
    main()
