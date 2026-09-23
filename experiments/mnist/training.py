import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from encoding.quantum import angle_encode
from encoding.ttfs import ttfs_encode
from models.mnist_qsnn import MNIST4x4QSNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_theta(features, time_window):
    times = ttfs_encode(np.asarray(features, dtype=float), time_window)
    return torch.tensor(angle_encode(times, time_window), dtype=torch.float32)


def classification_metrics(targets, logits):
    targets = np.asarray(targets, dtype=int)
    predictions = logits.argmax(1).cpu().numpy()
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "class_accuracy": {
            str(label): float(np.mean(predictions[targets == label] == label))
            for label in range(10)
        },
        "cross_entropy": float(F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long))),
    }


def evaluate_in_batches(model, theta, labels, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            chunks.append(model(theta[start:start + batch_size]).cpu())
    return classification_metrics(labels, torch.cat(chunks))


def train_mnist_model(config, train_features, train_labels, validation_features,
                      validation_labels, checkpoint_path):
    seed = int(config["seed"])
    set_seed(seed)
    train_theta = to_theta(train_features, config["time_window"])
    validation_theta = to_theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNIST4x4QSNN(
        int(config["n_qubits"]), int(config["reupload_blocks"]), int(config["n_classes"])
    )
    with torch.no_grad():
        model.qlayer.weights.uniform_(-0.1, 0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_key = None
    best_state = None
    best_metrics = None
    history = []
    stale_epochs = 0
    started = time.perf_counter()
    batch_size = int(config["batch_size"])

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite MNIST loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_in_batches(
            model, validation_theta, validation_labels, batch_size
        )
        history.append({
            "seed": seed, "epoch": epoch, "training_loss": float(np.mean(losses)),
            "validation_cross_entropy": validation["cross_entropy"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        })
        key = (-validation["accuracy"], validation["cross_entropy"], epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"[MNIST] seed={seed} epoch={epoch}/{config['max_epochs']} "
            f"train_ce={history[-1]['training_loss']:.4f} "
            f"val_ce={validation['cross_entropy']:.4f} "
            f"val_acc={validation['accuracy']:.4f} val_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        if epoch >= int(config["minimum_epochs"]) and stale_epochs >= int(config["patience"]):
            break

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return {
        "model": model, "history": history, "validation": best_metrics,
        "best_epoch": int(best_key[2]),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def resume_mnist_model(config, train_features, train_labels, validation_features,
                       validation_labels, checkpoint_path, history, output_path):
    seed = int(config["seed"])
    resume_epoch = int(config["resume_epoch"])
    set_seed(seed)
    train_theta = to_theta(train_features, config["time_window"])
    validation_theta = to_theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNIST4x4QSNN(
        int(config["n_qubits"]), int(config["reupload_blocks"]), int(config["n_classes"])
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    optimizer_restored = isinstance(payload, dict) and "model_state" in payload
    model.load_state_dict(payload["model_state"] if optimizer_restored else payload)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    if optimizer_restored and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    else:
        optimizer_restored = False
    prior = [row for row in history if int(row["seed"]) == seed]
    if not prior or int(prior[-1]["epoch"]) != resume_epoch:
        raise RuntimeError(f"Seed {seed} history does not end at epoch {resume_epoch}.")
    best_row = min(prior, key=lambda row: (float(row["validation_cross_entropy"]), int(row["epoch"])))
    if int(best_row["epoch"]) != resume_epoch:
        raise RuntimeError(f"Seed {seed} epoch-{resume_epoch} weights are not the best-CE state.")
    best_loss = float(best_row["validation_cross_entropy"])
    best_epoch = resume_epoch
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = evaluate_in_batches(model, validation_theta, validation_labels, int(config["batch_size"]))
    stale_epochs = 0
    extended = list(prior)
    started = time.perf_counter()
    stopping_reason = "max_epoch"
    batch_size = int(config["batch_size"])
    print(
        f"[RESUME] seed={seed} checkpoint={checkpoint_path} loaded_epoch={resume_epoch} "
        f"optimizer_restored={optimizer_restored} best_val_loss={best_loss:.6f}", flush=True,
    )
    for epoch in range(resume_epoch + 1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite MNIST loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_in_batches(model, validation_theta, validation_labels, batch_size)
        row = {
            "seed": seed, "epoch": epoch, "training_loss": float(np.mean(losses)),
            "validation_cross_entropy": validation["cross_entropy"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        extended.append(row)
        if validation["cross_entropy"] < best_loss:
            best_loss = validation["cross_entropy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation
            stale_epochs = 0
            print(f"[BEST] seed={seed} epoch={epoch} val_loss={best_loss:.6f} val_acc={validation['accuracy']:.4f}", flush=True)
        else:
            stale_epochs += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_val_loss": best_loss, "best_epoch": best_epoch,
            "best_model_state": best_state,
        }, output_path.with_name(output_path.stem + "_latest.pt"))
        print(
            f"[TRAIN] seed={seed} epoch={epoch}/{config['max_epochs']} "
            f"train_loss={row['training_loss']:.4f} val_loss={validation['cross_entropy']:.4f} "
            f"val_acc={validation['accuracy']:.4f} patience={stale_epochs}/{config['patience']}",
            flush=True,
        )
        if epoch >= int(config["minimum_early_stop_epoch"]) and stale_epochs >= int(config["patience"]):
            stopping_reason = "early_stop"
            print(f"[EARLY STOP] seed={seed} epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    else:
        epoch = int(config["max_epochs"])
        print(f"[MAX EPOCH] seed={seed} reached {epoch} epochs", flush=True)
    model.load_state_dict(best_state)
    torch.save(best_state, output_path)
    return {
        "model": model, "history": extended, "validation": best_metrics,
        "resumed_epoch": resume_epoch, "stopping_epoch": epoch,
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "optimizer_restored": optimizer_restored, "stopping_reason": stopping_reason,
        "runtime_seconds": float(time.perf_counter() - started),
    }
