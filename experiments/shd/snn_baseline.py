import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.nmnist.snn_baseline import select_device, set_determinism
from models.shd_snn import SHDRecurrentSNN


def events_to_spike_bins(events, time_steps=100, duration_us=1_400_000, input_channels=700):
    """Create fixed-window binary bins while preserving SHD time and channel identity."""
    output = np.zeros((int(time_steps), int(input_channels)), dtype=np.uint8)
    if len(events) == 0:
        return output
    times = np.asarray(events["t"], dtype=np.int64)
    channels = np.asarray(events["x"], dtype=np.int64)
    if np.any(np.diff(times) < 0) or times.min() < 0:
        raise ValueError("SHD spike times must be nonnegative and nondecreasing.")
    if channels.min() < 0 or channels.max() >= input_channels:
        raise ValueError("SHD input channel is outside [0, 699].")
    valid = times <= int(duration_us)
    bins = np.minimum((times[valid] * int(time_steps)) // int(duration_us), int(time_steps) - 1)
    output[bins, channels[valid]] = 1
    return output


def stratified_split(labels, validation_fraction, seed):
    indices = np.arange(len(labels), dtype=np.int64)
    train_ids, validation_ids = train_test_split(
        indices, test_size=float(validation_fraction), random_state=int(seed),
        shuffle=True, stratify=np.asarray(labels, dtype=int),
    )
    return np.sort(train_ids), np.sort(validation_ids)


def preprocess_partition(dataset, config, partition):
    frames = np.empty((len(dataset), config["time_steps"], config["input_channels"]), dtype=np.uint8)
    labels = np.empty(len(dataset), dtype=np.int64)
    event_counts = np.empty(len(dataset), dtype=np.int64)
    clipped_events = 0
    started = time.perf_counter()
    for index in range(len(dataset)):
        events, label = dataset[index]
        frames[index] = events_to_spike_bins(
            events, config["time_steps"], config["duration_us"], config["input_channels"]
        )
        labels[index] = int(label)
        event_counts[index] = len(events)
        clipped_events += int(np.sum(np.asarray(events["t"]) > config["duration_us"]))
        if (index + 1) % 1000 == 0 or index + 1 == len(dataset):
            print(f"[{partition} preprocessing] {index + 1}/{len(dataset)}", flush=True)
    metadata = {
        "partition": partition, "samples": len(dataset), "native_input_channels": 700,
        "native_event_fields": ["t", "x", "p"], "artificial_polarity_ignored": True,
        "duration_us": config["duration_us"], "time_steps": config["time_steps"],
        "bin_width_us": config["duration_us"] / config["time_steps"],
        "tensor_shape_per_sample": [config["time_steps"], config["input_channels"]],
        "representation": "dense binary occupancy", "normalization": "none",
        "clipped_events": clipped_events,
        "event_count_minimum": int(event_counts.min()),
        "event_count_median": float(np.median(event_counts)),
        "event_count_maximum": int(event_counts.max()),
        "preprocessing_seconds": time.perf_counter() - started,
    }
    return frames, labels, metadata


def make_tensor_loader(frames, labels, indices, batch_size, shuffle, seed):
    dataset = TensorDataset(torch.from_numpy(frames), torch.from_numpy(labels).long())
    selected = Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(selected, batch_size=int(batch_size),
                      shuffle=shuffle, num_workers=0, generator=generator)


def evaluate(model, loader, device, n_classes=20, details=False):
    model.eval()
    total_loss = 0.0
    targets, predictions = [], []
    with torch.no_grad():
        for spikes, labels in loader:
            spikes, labels = spikes.float().to(device), labels.to(device)
            logits = model(spikes)
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
            str(label): float(np.mean(predictions[targets == label] == label)) for label in range(n_classes)
        }
        result["confusion_matrix"] = confusion_matrix(
            targets, predictions, labels=range(n_classes)).tolist()
    return result


def train_seed(config, frames, labels, train_ids, validation_ids, checkpoint_path):
    seed = int(config["seed"])
    set_determinism(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(config["device"])
    model = SHDRecurrentSNN(config["input_channels"], config["hidden_size"],
                            config["n_classes"], config["lif_decay"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    validation_loader = make_tensor_loader(
        frames, labels, validation_ids, config["batch_size"], False, seed)
    best_accuracy, best_loss, best_epoch, stale = -1.0, float("inf"), None, 0
    history, started = [], time.perf_counter()
    stopping_reason = "max_epochs"
    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loader = make_tensor_loader(frames, labels, train_ids, config["batch_size"], True, seed + epoch)
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for spikes, targets in train_loader:
            spikes, targets = spikes.float().to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(spikes)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            seen += len(targets)
        validation = evaluate(model, validation_loader, device, config["n_classes"])
        epoch_runtime = time.perf_counter() - epoch_started
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                        "config": config}, checkpoint_path)
        else:
            stale += 1
        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        row = {"seed": seed, "epoch": epoch, "train_loss": total_loss / seen,
               "train_accuracy": correct / seen, "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
               "best_validation_accuracy": best_accuracy, "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_runtime_seconds": epoch_runtime, "gpu_memory_mib": gpu_memory}
        history.append(row)
        print(f"seed={seed} epoch={epoch}/{config['epochs']} train_loss={row['train_loss']:.4f} "
              f"train_accuracy={row['train_accuracy']:.4f} validation_loss={row['validation_loss']:.4f} "
              f"validation_accuracy={row['validation_accuracy']:.4f} "
              f"best_validation_accuracy={best_accuracy:.4f} learning_rate={row['learning_rate']:.6f} "
              f"epoch_runtime={epoch_runtime:.2f}s GPU_memory={gpu_memory:.1f}MiB", flush=True)
        if epoch >= config["minimum_epochs"] and stale >= config["patience"]:
            stopping_reason = "early_stopping"
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model, {"best_epoch": best_epoch, "stopping_epoch": epoch,
                   "stopping_reason": stopping_reason,
                   "best_validation": evaluate(model, validation_loader, device, config["n_classes"], True),
                   "training_runtime_seconds": time.perf_counter() - started,
                   "parameters": model.trainable_parameter_count(), "history": history}
