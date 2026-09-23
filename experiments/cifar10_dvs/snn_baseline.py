import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from models.cifar10_dvs_snn import CIFAR10DVSConvSNN


def events_to_frames(events, temporal_bins=10, sensor_size=(128, 128), downsample_factor=1):
    """Bin native x/y/t/p events into ordered, polarity-separated count frames."""
    target_width, target_height = map(int, sensor_size)
    factor = int(downsample_factor)
    if factor <= 0:
        raise ValueError("downsample_factor must be positive.")
    source_width = target_width * factor
    source_height = target_height * factor
    frames = np.zeros((int(temporal_bins), 2, target_height, target_width), dtype=np.float32)
    if len(events) == 0:
        return frames
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    if np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        x, y, t, p = x[order], y[order], t[order], p[order]
    if x.min() < 0 or x.max() >= source_width or y.min() < 0 or y.max() >= source_height:
        raise ValueError("CIFAR10-DVS coordinates are outside the sensor bounds.")
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("CIFAR10-DVS polarity must be binary.")
    x = x // int(downsample_factor)
    y = y // int(downsample_factor)
    duration = max(int(t[-1]) - int(t[0]) + 1, 1)
    bins = np.minimum(((t - t[0]) * temporal_bins) // duration, temporal_bins - 1)
    np.add.at(frames, (bins, p, y, x), 1.0)
    maximum = float(frames.max())
    if maximum > 0:
        frames = frames / maximum
    return frames.astype(np.float32)


def stratified_split_indices(targets, split_seed, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    targets = np.asarray(targets, dtype=int)
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    rng = np.random.default_rng(int(split_seed))
    train, validation, test = [], [], []
    for label in sorted(np.unique(targets).tolist()):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_test - n_val
        test.extend(indices[:n_test])
        validation.extend(indices[n_test:n_test + n_val])
        train.extend(indices[n_test + n_val:])
        assert len(indices[n_test + n_val:]) == n_train
    train = np.asarray(train, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if len(train) + len(validation) + len(test) != len(targets):
        raise RuntimeError("Invalid CIFAR10-DVS split: counts do not sum.")
    if np.intersect1d(train, validation).size or np.intersect1d(train, test).size \
            or np.intersect1d(validation, test).size:
        raise RuntimeError("CIFAR10-DVS partitions overlap.")
    return train, validation, test


class FramedCIFAR10DVS(Dataset):
    def __init__(self, dataset, temporal_bins, indices=None, sensor_size=(128, 128),
                 downsample_factor=1, cache=False):
        self.dataset = dataset
        self.temporal_bins = int(temporal_bins)
        self.sensor_size = tuple(map(int, sensor_size))
        self.downsample_factor = int(downsample_factor)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.index_to_position = None
        self.frames = None
        self.labels = None
        if cache:
            if self.indices is None:
                self.indices = np.arange(len(dataset), dtype=np.int64)
            self.index_to_position = {
                int(index): position for position, index in enumerate(self.indices)
            }
            height = self.sensor_size[1]
            width = self.sensor_size[0]
            self.frames = np.empty(
                (len(self.indices), self.temporal_bins, 2, height, width), dtype=np.float32
            )
            self.labels = np.empty(len(self.indices), dtype=np.int64)
            for position, index in enumerate(self.indices):
                events, label = self.dataset[int(index)]
                self.frames[position] = events_to_frames(
                    events, self.temporal_bins, self.sensor_size, self.downsample_factor
                )
                self.labels[position] = int(label)

    def __len__(self):
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[int(index)]
        if self.frames is not None:
            position = self.index_to_position[int(index)]
            return torch.from_numpy(self.frames[position]).float(), int(self.labels[position])
        events, label = self.dataset[int(index)]
        frames = events_to_frames(
            events, self.temporal_bins, self.sensor_size, self.downsample_factor
        )
        return torch.from_numpy(frames).float(), int(label)


def set_determinism(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def gpu_report(device):
    info = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        info.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "total_vram_gb": round(total / 1024 ** 3, 3),
            "free_vram_gb": round(free / 1024 ** 3, 3),
            "allocated_vram_gb": round(torch.cuda.memory_allocated(device) / 1024 ** 3, 3),
            "reserved_vram_gb": round(torch.cuda.memory_reserved(device) / 1024 ** 3, 3),
        })
    return info


def print_gpu_report(device, batch_size):
    info = gpu_report(device)
    print(f"GPU name: {info.get('gpu_name', 'N/A')}", flush=True)
    print(f"total VRAM: {info.get('total_vram_gb', 'N/A')} GB", flush=True)
    print(f"free VRAM: {info.get('free_vram_gb', 'N/A')} GB", flush=True)
    print(f"current allocated VRAM: {info.get('allocated_vram_gb', 'N/A')} GB", flush=True)
    print(f"selected batch size: {batch_size}", flush=True)
    return info


def make_loader(dataset, batch_size, shuffle, seed, pin_memory=False):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=0,
        generator=generator, pin_memory=pin_memory,
    )


def evaluate(model, loader, device, details=False, n_classes=10):
    model.eval()
    total_loss = 0.0
    targets, predictions = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    result = {
        "loss": total_loss / len(targets),
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "sample_count": int(len(targets)),
    }
    if details:
        labels = list(range(n_classes))
        result["per_class_accuracy"] = {
            str(label): float(np.mean(predictions[targets == label] == label)) for label in labels
        }
        result["confusion_matrix"] = confusion_matrix(targets, predictions, labels=labels).tolist()
    return result


def _synchronize(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def train(config, dataset, train_indices, validation_indices, checkpoint_path, batch_size,
          sensor_size=(128, 128), downsample_factor=1, cache_frames=True):
    seed = int(config["model_seed"])
    set_determinism(seed)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    print_gpu_report(device, batch_size)
    model = CIFAR10DVSConvSNN(spatial_size=int(sensor_size[0])).to(device)
    feature_size = int(sensor_size[0]) // 8
    print(
        f"architecture: CIFAR10DVSConvSNN conv(2->16)-LIF-pool "
        f"conv(16->32)-LIF-pool conv(32->64)-LIF-pool linear({64 * feature_size * feature_size}->10)",
        flush=True,
    )
    print(f"neuron configuration: LIF decay=0.5 threshold=1.0 surrogate=fast-sigmoid(slope=5)", flush=True)
    print(f"parameter count: {model.trainable_parameter_count()}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    sched_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_cfg["mode"], factor=sched_cfg["factor"],
        patience=sched_cfg["patience"], min_lr=sched_cfg["min_lr"])
    assert sched_cfg["step_on"] == "validation_accuracy"
    cache_start = time.perf_counter()
    train_set = FramedCIFAR10DVS(
        dataset, config["temporal_bins"], train_indices, sensor_size,
        downsample_factor, cache_frames,
    )
    validation_set = FramedCIFAR10DVS(
        dataset, config["temporal_bins"], validation_indices, sensor_size,
        downsample_factor, cache_frames,
    )
    cache_runtime = time.perf_counter() - cache_start
    validation_loader = make_loader(
        validation_set, batch_size, False, seed, pin_memory=device.type == "cuda"
    )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    best_accuracy, best_loss, best_epoch, stale = -1.0, float("inf"), None, 0
    lr_history = [{"epoch": 0, "lr": float(optimizer.param_groups[0]["lr"])}]
    history, started = [], time.perf_counter()
    stopping_reason = "max_epochs"
    scheduler_steps = 0
    peak_allocated = 0.0
    peak_reserved = 0.0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        _synchronize(device)
        epoch_start = time.perf_counter()
        train_loader = make_loader(
            train_set, batch_size, True, seed + epoch, pin_memory=device.type == "cuda"
        )
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        validation = evaluate(model, validation_loader, device)
        _synchronize(device)
        epoch_runtime = time.perf_counter() - epoch_start
        old_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(validation["accuracy"])
        scheduler_steps += 1
        new_lr = float(optimizer.param_groups[0]["lr"])
        if new_lr != old_lr:
            print(f"LR REDUCED: {old_lr} -> {new_lr} epoch={epoch}", flush=True)
            lr_history.append({"epoch": epoch, "old_lr": old_lr, "new_lr": new_lr})
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = \
                validation["accuracy"], validation["loss"], epoch, 0
        else:
            stale += 1
        if device.type == "cuda" and torch.cuda.is_available():
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            allocated = torch.cuda.memory_allocated(device)
        else:
            allocated = 0.0
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "best_validation_accuracy": best_accuracy,
            "learning_rate": new_lr,
            "epoch_runtime_seconds": epoch_runtime,
            "throughput_samples_per_second": seen / epoch_runtime,
            "gpu_allocated_gb": round(allocated / 1024 ** 3, 3),
            "gpu_peak_allocated_gb": round(peak_allocated / 1024 ** 3, 3),
            "gpu_reserved_gb": round(
                torch.cuda.memory_reserved(device) / 1024 ** 3, 3
            ) if device.type == "cuda" else 0.0,
            "gpu_peak_reserved_gb": round(peak_reserved / 1024 ** 3, 3),
        }
        if improved:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(), "best_epoch": epoch,
                "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                "learning_rate": new_lr, "config": config,
            }, checkpoint_path)
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_accuracy={row['train_accuracy']:.4f} "
            f"validation_loss={row['validation_loss']:.4f} "
            f"validation_accuracy={row['validation_accuracy']:.4f} "
            f"best_validation_accuracy={best_accuracy:.4f} "
            f"learning_rate={new_lr:.8f} "
            f"epoch_runtime={epoch_runtime:.1f}s "
            f"throughput={row['throughput_samples_per_second']:.1f}samples/s "
            f"GPU_memory={row['gpu_allocated_gb']:.3f}GB "
            f"peak_GPU_memory={row['gpu_peak_allocated_gb']:.3f}GB",
            flush=True,
        )
        if stale >= int(config["early_stop_patience"]):
            stopping_reason = "early_stopping"
            break
    runtime = time.perf_counter() - started
    assert scheduler_steps >= 1, "scheduler.step was never called"
    frozen = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(frozen["model_state"])
    return model, {
        "best_epoch": best_epoch,
        "stopping_epoch": epoch,
        "stopping_reason": stopping_reason,
        "best_validation": evaluate(model, validation_loader, device, True),
        "parameters": model.trainable_parameter_count(),
        "training_runtime_seconds": runtime,
        "history": history,
        "lr_history": lr_history,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_steps": scheduler_steps,
        "cache_preprocessing_runtime_seconds": cache_runtime,
        "peak_gpu_allocated_gb": round(peak_allocated / 1024 ** 3, 3),
        "peak_gpu_reserved_gb": round(peak_reserved / 1024 ** 3, 3),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata():
    import tonic
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "tonic": tonic.__version__, "numpy": np.__version__}


def write_history(path, history):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
