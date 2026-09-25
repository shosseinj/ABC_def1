"""Representation-matched clean training for DVS-Gesture and CIFAR10-DVS.

This entry point intentionally contains no attack code. Development runs use
validation only; final runs evaluate the frozen test partition exactly once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.improved_event_snn import ImprovedEventConvSNN, ResidualEventConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
CONFIG_PATH = ROOT / "configs/clean_improved.config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATASETS = ("dvs_gesture", "cifar10_dvs")
REPRESENTATIONS = ("binary", "integer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def event_frames(events, representation: str) -> np.ndarray:
    """Create exact T=10 [T,polarity,y,x] occupancy or raw-count grids."""
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    dtype = np.uint8 if representation == "binary" else np.uint16
    result = np.zeros((10, 2, 128, 128), dtype=dtype)
    if not len(events):
        return result
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    timestamps = np.asarray(events["t"], dtype=np.int64)
    polarity = np.asarray(events["p"], dtype=np.int64)
    if np.any(np.diff(timestamps) < 0):
        order = np.argsort(timestamps, kind="stable")
        x, y, timestamps, polarity = (value[order] for value in
                                      (x, y, timestamps, polarity))
    if (x.min() < 0 or x.max() >= 128 or y.min() < 0 or y.max() >= 128 or
            not np.all((polarity == 0) | (polarity == 1))):
        raise ValueError("Invalid coordinate or polarity in raw events")
    duration = max(int(timestamps[-1]) - int(timestamps[0]) + 1, 1)
    bins = np.minimum(((timestamps - timestamps[0]) * 10) // duration, 9)
    if representation == "binary":
        result[bins, polarity, y, x] = 1
    else:
        wide = np.zeros(result.shape, dtype=np.uint32)
        np.add.at(wide, (bins, polarity, y, x), 1)
        if int(wide.max()) > np.iinfo(np.uint16).max:
            raise OverflowError("A temporal cell exceeds uint16 capacity")
        result = wide.astype(np.uint16)
    return result


def output_dir(dataset: str) -> Path:
    name = "dvs_gesture_clean_improved" if dataset == "dvs_gesture" \
        else "cifar10_dvs_clean_improved"
    return ROOT / "Reports/results" / name


def cache_dir(dataset: str) -> Path:
    return ROOT / "Reports/checkpoints/clean_improved_cache" / dataset


def load_native(dataset: str):
    if dataset == "dvs_gesture":
        from tonic.datasets import DVSGesture
        return (DVSGesture(save_to=str(ROOT / "data/dvs_gesture"), train=True),
                DVSGesture(save_to=str(ROOT / "data/dvs_gesture"), train=False))
    from tonic.datasets import CIFAR10DVS
    return (CIFAR10DVS(save_to=str(ROOT / "data/cifar10_dvs")),)


def build_partition_cache(native, dataset: str, partition: str,
                          representation: str):
    directory = cache_dir(dataset)
    directory.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if representation == "binary" else np.uint16
    frames_path = directory / f"{partition}_t10_{representation}_{np.dtype(dtype).name}.npy"
    labels_path = directory / f"{partition}_labels.npy"
    marker_path = directory / f"{partition}_t10_{representation}.json"
    expected_shape = (len(native), 10, 2, 128, 128)
    if frames_path.exists() and labels_path.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        frames = np.load(frames_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        if (tuple(frames.shape) == expected_shape and frames.dtype == dtype and
                len(labels) == len(native) and marker.get("complete") is True):
            return frames, labels, frames_path, labels_path
    frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=dtype,
                                       shape=expected_shape)
    labels = np.empty(len(native), dtype=np.int64)
    maximum = 0
    for index in range(len(native)):
        events, label = native[index]
        frame = event_frames(events, representation)
        frames[index] = frame
        labels[index] = int(label)
        maximum = max(maximum, int(frame.max()))
        if (index + 1) % 250 == 0 or index + 1 == len(native):
            print(f"cache | {dataset} | {partition} | {representation} | "
                  f"{index + 1}/{len(native)}", flush=True)
    frames.flush()
    if labels_path.exists():
        cached_labels = np.load(labels_path, mmap_mode="r")
        if not np.array_equal(np.asarray(cached_labels), labels):
            raise AssertionError(f"Shared label cache mismatch: {labels_path}")
    else:
        np.save(labels_path, labels)
    atomic_json(marker_path, {
        "complete": True, "dataset": dataset, "partition": partition,
        "representation": representation, "shape": expected_shape,
        "dtype": np.dtype(dtype).name, "T": 10, "polarity_channels": 2,
        "spatial_resolution": [128, 128], "normalization": "none",
        "maximum_cell_value": maximum,
    })
    return (np.load(frames_path, mmap_mode="r"),
            np.load(labels_path, mmap_mode="r"), frames_path, labels_path)


def prepare(dataset: str, representation: str):
    native = load_native(dataset)
    out = output_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "dvs_gesture":
        train_frames, train_labels, train_path, _ = build_partition_cache(
            native[0], dataset, "official_train", representation)
        test_frames, test_labels, test_path, _ = build_partition_cache(
            native[1], dataset, "official_test", representation)
        split_path = out / "frozen_split.json"
        if not split_path.exists():
            ids = np.arange(len(train_labels))
            train_ids, validation_ids = train_test_split(
                ids, test_size=CONFIG["dvs_validation_fraction"],
                random_state=CONFIG["split_seed"], stratify=np.asarray(train_labels))
            atomic_json(split_path, {
                "dataset": "DVS-Gesture", "split_seed": CONFIG["split_seed"],
                "official_partition": "Tonic DVSGesture train=True/train=False",
                "train_indices": sorted(map(int, train_ids)),
                "validation_indices": sorted(map(int, validation_ids)),
                "test_indices": list(range(len(test_labels))),
            })
        split = json.loads(split_path.read_text(encoding="utf-8"))
        return (train_frames, train_labels, test_frames, test_labels, split,
                split_path, train_path, test_path, native)
    frames, labels, frame_path, _ = build_partition_cache(
        native[0], dataset, "all", representation)
    source_split = ROOT / "results/cifar10_dvs_snn_seed42_split.json"
    split_path = out / "frozen_split.json"
    if not split_path.exists():
        source = json.loads(source_split.read_text(encoding="utf-8"))
        atomic_json(split_path, source)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return (frames, labels, frames, labels, split, split_path, frame_path,
            frame_path, native)


class FrameSet(Dataset):
    def __init__(self, frames, labels, indices, augment: bool, dataset: str):
        self.frames, self.labels = frames, labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = bool(augment)
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        # Integer-safe cache is cast directly to float32; no normalization.
        frame = torch.from_numpy(np.array(self.frames[index], dtype=np.float32,
                                          copy=True))
        return frame, int(self.labels[index])


def augment_batch(frames: torch.Tensor, dataset: str) -> torch.Tensor:
    """Apply one spatial transform per sample, shared across all time bins."""
    padding = int(CONFIG["translation_pixels"])
    if padding:
        padded = F.pad(frames, (padding, padding, padding, padding))
        shifted = torch.empty_like(frames)
        offsets = torch.randint(0, 2 * padding + 1, (len(frames), 2),
                                device=frames.device)
        for index, (top, left) in enumerate(offsets.tolist()):
            shifted[index] = padded[index, :, :, top:top + 128, left:left + 128]
        frames = shifted
    flip_probability = (CONFIG["cifar_horizontal_flip_probability"]
                        if dataset == "cifar10_dvs" else
                        CONFIG["dvs_horizontal_flip_probability"])
    if flip_probability:
        mask = torch.rand(len(frames), device=frames.device) < flip_probability
        frames[mask] = frames[mask].flip(-1)
    if dataset == "cifar10_dvs" and CONFIG["cifar_rotation_degrees"]:
        # One nearest-neighbor spatial rotation per sample, shared by all T/C.
        radians = ((torch.rand(len(frames), device=frames.device) * 2.0 - 1.0) *
                   float(CONFIG["cifar_rotation_degrees"]) * math.pi / 180.0)
        theta = torch.zeros((len(frames), 2, 3), device=frames.device,
                            dtype=frames.dtype)
        theta[:, 0, 0] = radians.cos(); theta[:, 0, 1] = -radians.sin()
        theta[:, 1, 0] = radians.sin(); theta[:, 1, 1] = radians.cos()
        packed = frames.flatten(1, 2)
        grid = F.affine_grid(theta, packed.shape, align_corners=False)
        frames = F.grid_sample(packed, grid, mode="nearest", padding_mode="zeros",
                               align_corners=False).unflatten(1, (10, 2))
    if dataset == "cifar10_dvs" and CONFIG["cifar_cutout_size"]:
        size = int(CONFIG["cifar_cutout_size"])
        tops = torch.randint(0, 129 - size, (len(frames),), device=frames.device)
        lefts = torch.randint(0, 129 - size, (len(frames),), device=frames.device)
        for index, (top, left) in enumerate(zip(tops.tolist(), lefts.tolist())):
            frames[index, :, :, top:top + size, left:left + size] = 0
    if dataset == "cifar10_dvs" and CONFIG["cifar_event_dropout"]:
        # Drop occupied cells without rescaling surviving Binary/Integer amplitudes.
        keep = torch.rand_like(frames) >= float(CONFIG["cifar_event_dropout"])
        frames = frames * keep
    return frames


def make_loader(frames, labels, indices, batch_size: int, shuffle: bool,
                seed: int, dataset: str, augment: bool = False):
    return DataLoader(FrameSet(frames, labels, indices, augment, dataset),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=bool(CONFIG["amp"])):
            logits = model(frames)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "correct": correct, "samples": total}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one(dataset: str, representation: str, seed: int, mode: str,
              epochs_override: int | None = None):
    (train_frames, train_labels, test_frames, test_labels, split, split_path,
     train_cache, test_cache, _) = prepare(dataset, representation)
    set_seed(seed)
    device = torch.device("cuda")
    n_classes = 11 if dataset == "dvs_gesture" else 10
    channels = (CONFIG["dvs_channels"] if dataset == "dvs_gesture" else
                CONFIG["cifar_channels"])
    pools = (CONFIG["dvs_pools"] if dataset == "dvs_gesture" else
             CONFIG["cifar_pools"])
    pool_after_spike = dataset == "cifar10_dvs"
    if dataset == "cifar10_dvs" and CONFIG["cifar_architecture"] == "residual":
        model = ResidualEventConvSNN(
            n_classes=n_classes,
            stem_channels=CONFIG["cifar_residual_stem_channels"],
            stage_channels=CONFIG["cifar_residual_stage_channels"],
            decay=CONFIG["lif_decay"], threshold=CONFIG["lif_threshold"],
            readout_size=CONFIG["cifar_readout_size"],
            feature_dropout=CONFIG["cifar_feature_dropout"]).to(device)
    else:
        model = ImprovedEventConvSNN(
            n_classes=n_classes, channels=channels,
            decay=CONFIG["lif_decay"], threshold=CONFIG["lif_threshold"],
            readout_size=(CONFIG["cifar_readout_size"]
                          if dataset == "cifar10_dvs" else CONFIG["readout_size"]),
            pool_after_spike=pool_after_spike, pools=pools,
            feature_dropout=(CONFIG["cifar_feature_dropout"]
                             if dataset == "cifar10_dvs" else 0.0),
            normalization=(CONFIG["cifar_normalization"]
                           if dataset == "cifar10_dvs" else "batch"),
            temporal_attention=(CONFIG["cifar_temporal_attention"]
                                if dataset == "cifar10_dvs" else False)).to(device)
    weight_decay = (CONFIG["cifar_weight_decay"] if dataset == "cifar10_dvs"
                    else CONFIG["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                  weight_decay=weight_decay)
    batch_size = int(CONFIG["dvs_batch_size"] if dataset == "dvs_gesture"
                     else CONFIG["cifar_batch_size"])
    epochs = int(epochs_override or (CONFIG["development_epochs"] if mode == "develop"
                                     else CONFIG["max_epochs"]))
    scheduler_per_batch = (dataset == "cifar10_dvs" and
                           CONFIG["cifar_scheduler"] == "OneCycleLR")
    if scheduler_per_batch:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=CONFIG["learning_rate"], epochs=epochs,
            steps_per_epoch=math.ceil(len(split["train_indices"]) / batch_size),
            pct_start=CONFIG["cifar_onecycle_pct_start"], div_factor=10.0,
            final_div_factor=1000.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(CONFIG["amp"]))
    validation_loader = make_loader(
        train_frames, train_labels, split["validation_indices"], batch_size,
        False, seed, dataset)
    out = output_dir(dataset)
    run_name = f"{representation}_seed{seed}"
    run_dir = out / ("development" if mode == "develop" else "")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / f"{run_name}_best.pt"
    latest = run_dir / f"{run_name}_latest.pt"
    result_path = run_dir / f"{run_name}_result.json"
    if mode == "final" and result_path.exists() and checkpoint.exists():
        completed = json.loads(result_path.read_text(encoding="utf-8"))
        if (completed.get("status") == "COMPLETE" and
                completed.get("checkpoint_sha256") == sha256(checkpoint)):
            print(f"{dataset} | {representation} | {seed} | already complete",
                  flush=True)
            return completed
    best_accuracy, best_loss, best_epoch, stale = -1.0, math.inf, 0, 0
    history = []
    start_epoch = 1
    if mode == "final" and latest.exists():
        resumed = torch.load(latest, map_location=device, weights_only=False)
        if resumed.get("config_sha256") == sha256(CONFIG_PATH):
            model.load_state_dict(resumed["model_state"])
            optimizer.load_state_dict(resumed["optimizer_state"])
            scheduler.load_state_dict(resumed["scheduler_state"])
            scaler.load_state_dict(resumed["scaler_state"])
            best_accuracy = float(resumed["best_accuracy"])
            best_loss = float(resumed["best_loss"])
            best_epoch = int(resumed["best_epoch"])
            stale = int(resumed["stale"])
            history = list(resumed["history"])
            start_epoch = int(resumed["epoch"]) + 1
            random.setstate(resumed["python_rng_state"])
            np.random.set_state(resumed["numpy_rng_state"])
            torch.set_rng_state(resumed["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all(resumed["cuda_rng_states"])
            print(f"{dataset} | {representation} | {seed} | resume {start_epoch}",
                  flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        seen = correct = 0
        loss_sum = 0.0
        train_loader = make_loader(
            train_frames, train_labels, split["train_indices"], batch_size,
            True, seed + epoch, dataset, augment=True)
        for frames, labels in train_loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            frames = augment_batch(frames, dataset)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=bool(CONFIG["amp"])):
                temporal_loss = (dataset == "cifar10_dvs" and
                                 CONFIG["cifar_temporal_ensemble_loss"])
                temporal_logits = model(frames, return_sequence=temporal_loss)
                logits = (model.aggregate_logits(temporal_logits)
                          if temporal_loss and hasattr(model, "aggregate_logits") else
                          temporal_logits.mean(dim=1) if temporal_loss else temporal_logits)
                smoothing = (CONFIG["cifar_label_smoothing"]
                             if dataset == "cifar10_dvs" else 0.0)
                if temporal_loss:
                    repeated_labels = labels[:, None].expand(-1, temporal_logits.shape[1])
                    auxiliary_loss = F.cross_entropy(
                        temporal_logits.flatten(0, 1), repeated_labels.flatten(),
                        label_smoothing=smoothing)
                    loss = (F.cross_entropy(logits, labels, label_smoothing=smoothing) +
                            CONFIG["cifar_temporal_auxiliary_weight"] * auxiliary_loss)
                else:
                    loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CONFIG["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            if scheduler_per_batch:
                scheduler.step()
            seen += len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(loss.detach()) * len(labels)
        validation = evaluate(model, validation_loader, device)
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and
                     validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss = validation["accuracy"], validation["loss"]
            best_epoch, stale = epoch, 0
            torch.save({
                "model_state": model.state_dict(), "dataset": dataset,
                "representation": representation, "seed": seed,
                "best_epoch": epoch, "validation": validation,
                "model": {"channels": channels, "n_classes": n_classes,
                          "architecture": (CONFIG["cifar_architecture"]
                                           if dataset == "cifar10_dvs" else "plain"),
                          "residual_stem_channels": CONFIG["cifar_residual_stem_channels"],
                          "residual_stage_channels": CONFIG["cifar_residual_stage_channels"],
                          "decay": CONFIG["lif_decay"],
                          "threshold": CONFIG["lif_threshold"],
                          "readout_size": (CONFIG["cifar_readout_size"]
                                           if dataset == "cifar10_dvs" else
                                           CONFIG["readout_size"]),
                          "pool_after_spike": pool_after_spike,
                          "pools": pools,
                          "feature_dropout": (CONFIG["cifar_feature_dropout"]
                                              if dataset == "cifar10_dvs" else 0.0),
                          "normalization": (CONFIG["cifar_normalization"]
                                            if dataset == "cifar10_dvs" else "batch"),
                          "temporal_attention": (CONFIG["cifar_temporal_attention"]
                                                 if dataset == "cifar10_dvs" else False)},
                "config": CONFIG,
            }, checkpoint)
        else:
            stale += 1
        if not scheduler_per_batch:
            scheduler.step()
        row = {"dataset": dataset, "representation": representation,
               "seed": seed, "epoch": epoch, "train_loss": loss_sum / seen,
               "train_accuracy": correct / seen,
               "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"],
               "best_validation_accuracy": best_accuracy,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if mode == "final":
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(), "epoch": epoch,
                "best_accuracy": best_accuracy, "best_loss": best_loss,
                "best_epoch": best_epoch, "stale": stale, "history": history,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "config_sha256": sha256(CONFIG_PATH),
            }, latest)
        print(f"{dataset} | {representation} | {seed} | {epoch} | "
              f"{row['train_accuracy']:.4f} | {validation['accuracy']:.4f} | "
              f"{best_accuracy:.4f}", flush=True)
        if (mode == "develop" and
                stale >= CONFIG["development_early_stop_patience"]):
            break
        if (mode == "final" and epoch >= CONFIG["minimum_epochs"] and
                stale >= CONFIG["early_stop_patience"]):
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    result = {
        "status": "DEVELOPMENT_COMPLETE" if mode == "develop" else "COMPLETE",
        "dataset": "DVS-Gesture" if dataset == "dvs_gesture" else "CIFAR10-DVS",
        "representation": representation, "seed": seed,
        "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
        "parameters": model.trainable_parameter_count(),
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(CONFIG_PATH), "split_sha256": sha256(split_path),
        "train_cache": str(train_cache.relative_to(ROOT)).replace("\\", "/"),
        "test_cache": str(test_cache.relative_to(ROOT)).replace("\\", "/"),
        "runtime_seconds": time.perf_counter() - started,
        "test_tuning_performed": False, "attacks_run": False,
    }
    if mode == "final":
        test_loader = make_loader(test_frames, test_labels, split["test_indices"],
                                  batch_size, False, seed, dataset)
        result["test"] = evaluate(model, test_loader, device)
        print(f"{dataset} | {representation} | {seed} | "
              f"{result['test']['accuracy']:.4f} | {best_epoch}", flush=True)
    atomic_json(result_path, result)
    with (run_dir / f"{run_name}_training_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return result


def audit_pipeline():
    report = {"status": "PASS", "checks": [], "config": str(CONFIG_PATH)}
    for dataset in DATASETS:
        for representation in REPRESENTATIONS:
            prepared = prepare(dataset, representation)
            train_frames, train_labels, test_frames, test_labels = prepared[:4]
            split, split_path, _, _, native = prepared[4:]
            arrays = [(train_frames, train_labels, native[0], "train")]
            if dataset == "dvs_gesture":
                arrays.append((test_frames, test_labels, native[1], "test"))
            sample_ids = [0, len(arrays[0][0]) // 2, len(arrays[0][0]) - 1]
            for frames, labels, raw, partition in arrays:
                for index in sample_ids if partition == "train" else [0, len(frames)-1]:
                    expected = event_frames(raw[index][0], representation)
                    if not np.array_equal(np.asarray(frames[index]), expected):
                        raise AssertionError(f"cache mismatch: {dataset}/{representation}/{index}")
                    if int(labels[index]) != int(raw[index][1]):
                        raise AssertionError("label mismatch")
            values = np.asarray(train_frames[0])
            if representation == "binary" and not np.all((values == 0) | (values == 1)):
                raise AssertionError("non-binary value")
            if representation == "integer" and (not np.issubdtype(train_frames.dtype,
                                                                   np.integer) or values.min() < 0):
                raise AssertionError("invalid integer grid")
            report["checks"].append({
                "dataset": dataset, "representation": representation,
                "shape": list(train_frames.shape[1:]), "dtype": str(train_frames.dtype),
                "normalization": "none", "split_sha256": sha256(split_path),
                "train_samples": len(split["train_indices"]),
                "validation_samples": len(split["validation_indices"]),
                "test_samples": len(split["test_indices"]),
                "cache_reconstruction_samples": 5 if dataset == "dvs_gesture" else 3,
            })
    path = ROOT / "Reports/results/clean_improved_pipeline_audit.json"
    atomic_json(path, report)
    print(f"pipeline audit PASS | {path}", flush=True)


def aggregate_final():
    lines = []
    for dataset in DATASETS:
        out = output_dir(dataset)
        rows = []
        for representation in REPRESENTATIONS:
            accuracies = []
            for seed in CONFIG["seeds"]:
                path = out / f"{representation}_seed{seed}_result.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"dataset": result["dataset"],
                             "representation": representation, "seed": seed,
                             "best_validation_accuracy": result["best_validation_accuracy"],
                             "test_accuracy": result["test"]["accuracy"],
                             "best_epoch": result["best_epoch"],
                             "parameters": result["parameters"]})
                accuracies.append(100 * result["test"]["accuracy"])
            if len(accuracies) == 3:
                lines.append((dataset, representation, statistics.mean(accuracies),
                              statistics.stdev(accuracies)))
        if rows:
            with (out / "final_clean_accuracy.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    for dataset, representation, mean, sd in lines:
        print(f"{dataset} | {representation}: {mean:.2f} ± {sd:.2f}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "develop", "final", "aggregate"),
                        required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use the required interpreter: {PYTHON}")
    args = parse_args()
    if args.mode == "audit":
        audit_pipeline(); return
    if args.mode == "aggregate":
        aggregate_final(); return
    if args.dataset is None or args.representation is None:
        raise ValueError("--dataset and --representation are required for training")
    train_one(args.dataset, args.representation, args.seed, args.mode, args.epochs)


if __name__ == "__main__":
    main()
