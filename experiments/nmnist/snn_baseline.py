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
from torch.utils.data import DataLoader, Dataset, Subset

from models.nmnist_snn import NMNISTConvSNN


def events_to_frames(events, temporal_bins=10, sensor_size=(34, 34)):
    """Bin native x/y/t/p events into ordered, polarity-separated count frames."""
    width, height = map(int, sensor_size)
    frames = np.zeros((int(temporal_bins), 2, height, width), dtype=np.uint8)
    if len(events) == 0:
        return frames
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    if np.any(np.diff(t) < 0):
        raise ValueError("N-MNIST timestamps must be nondecreasing.")
    if x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height:
        raise ValueError("N-MNIST coordinates are outside the sensor bounds.")
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("N-MNIST polarity must be binary.")
    duration = max(int(t[-1]) - int(t[0]) + 1, 1)
    bins = np.minimum(((t - t[0]) * temporal_bins) // duration, temporal_bins - 1)
    counts = np.zeros_like(frames, dtype=np.uint16)
    np.add.at(counts, (bins, p, y, x), 1)
    return np.minimum(counts, 255).astype(np.uint8)


def stratified_train_validation_indices(targets, validation_per_class, seed):
    targets = np.asarray(targets, dtype=int)
    rng = np.random.default_rng(int(seed))
    train, validation = [], []
    for label in range(10):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        validation.extend(indices[:int(validation_per_class)])
        train.extend(indices[int(validation_per_class):])
    train = np.asarray(train, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    if np.intersect1d(train, validation).size or len(train) + len(validation) != len(targets):
        raise RuntimeError("Invalid N-MNIST train/validation split.")
    return train, validation


class FramedNMNIST(Dataset):
    def __init__(self, dataset, temporal_bins):
        self.dataset = dataset
        self.temporal_bins = int(temporal_bins)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        events, label = self.dataset[index]
        frames = events_to_frames(events, self.temporal_bins)
        return torch.from_numpy(frames).float(), int(label)


def set_determinism(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def select_device(requested):
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for N-MNIST training but is unavailable.")
    return device


def print_runtime_configuration(device):
    print(f"Python executable: {sys.executable}", flush=True)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA version: {torch.version.cuda}", flush=True)
    print(f"GPU name: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'N/A'}", flush=True)
    print(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
    print(f"deterministic algorithms enabled: {torch.are_deterministic_algorithms_enabled()}", flush=True)
    print(f"selected device: {device}", flush=True)


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=0, generator=generator)


def evaluate(model, loader, device, details=False):
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
        result["per_class_accuracy"] = {
            str(label): float(np.mean(predictions[targets == label] == label)) for label in range(10)
        }
        result["confusion_matrix"] = confusion_matrix(targets, predictions, labels=range(10)).tolist()
    return result


def train(config, dataset, train_indices, validation_indices, checkpoint_path):
    seed = int(config["seed"])
    set_determinism(seed)
    device = select_device(config["device"])
    print_runtime_configuration(device)
    model = NMNISTConvSNN(config["lif_decay"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    train_set = Subset(FramedNMNIST(dataset, config["temporal_bins"]), train_indices.tolist())
    validation_set = Subset(FramedNMNIST(dataset, config["temporal_bins"]), validation_indices.tolist())
    validation_loader = make_loader(validation_set, config["batch_size"], False, seed)
    best_accuracy, best_loss, best_epoch, stale = -1.0, float("inf"), None, 0
    history, started = [], time.perf_counter()
    stopping_reason = "max_epochs"
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loader = make_loader(train_set, config["batch_size"], True, seed + epoch)
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
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1": validation["macro_f1"]}
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                        "config": config}, checkpoint_path)
        else:
            stale += 1
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} train_accuracy={row['train_accuracy']:.4f} "
              f"validation_loss={row['validation_loss']:.4f} validation_accuracy={row['validation_accuracy']:.4f} "
              f"best_validation_accuracy={best_accuracy:.4f}", flush=True)
        if epoch >= int(config["minimum_epochs"]) and stale >= int(config["patience"]):
            stopping_reason = "early_stopping"
            break
    runtime = time.perf_counter() - started
    frozen = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(frozen["model_state"])
    return model, {
        "best_epoch": best_epoch, "stopping_epoch": epoch, "stopping_reason": stopping_reason,
        "best_validation": evaluate(model, validation_loader, device, True),
        "parameters": model.trainable_parameter_count(), "training_runtime_seconds": runtime,
        "history": history,
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
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
