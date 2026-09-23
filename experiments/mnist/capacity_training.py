import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from encoding.quantum import angle_encode
from encoding.ttfs import ttfs_encode
from models.mnist_qsnn import MNISTReducedQSNN


def _theta(features, time_window):
    return torch.tensor(
        angle_encode(ttfs_encode(features, time_window), time_window),
        dtype=torch.float32,
    )


def evaluate(model, theta, labels, batch_size, include_details=False):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            chunks.append(model(theta[start:start + batch_size]).cpu())
    logits = torch.cat(chunks)
    targets = np.asarray(labels, dtype=int)
    predictions = logits.argmax(1).numpy()
    metrics = {
        "loss": float(F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long))),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }
    if include_details:
        metrics["per_class_accuracy"] = {
            str(label): float(np.mean(predictions[targets == label] == label))
            for label in range(10)
        }
        metrics["confusion_matrix"] = confusion_matrix(
            targets, predictions, labels=range(10)
        ).tolist()
    return metrics


def train_capacity_variant(config, blocks, train_features, train_labels,
                           validation_features, validation_labels, checkpoint_path):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_theta = _theta(train_features, config["time_window"])
    validation_theta = _theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNISTReducedQSNN(config["n_qubits"], blocks, config["n_classes"])
    with torch.no_grad():
        model.qlayer.weights.uniform_(-0.1, 0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    stale = 0
    started = time.perf_counter()
    stopping_reason = "max_epoch"

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        minibatch_losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for blocks={blocks}, epoch={epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            minibatch_losses.append(float(loss.detach()))
        train_metrics = evaluate(model, train_theta, train_labels, config["batch_size"])
        validation_metrics = evaluate(model, validation_theta, validation_labels, config["batch_size"])
        row = {
            "blocks": blocks, "seed": seed, "epoch": epoch,
            "train_loss": float(np.mean(minibatch_losses)),
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_model_state": best_state, "history": history,
            "runtime_seconds": float(time.perf_counter() - started),
        }, checkpoint_path.with_name(checkpoint_path.stem + "_latest.pt"))
        print(
            f"[TRAIN] blocks={blocks} epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} patience={stale}/{config['patience']}",
            flush=True,
        )
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            stopping_reason = "early_stop"
            break

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint_path)
    train_metrics = evaluate(model, train_theta, train_labels, config["batch_size"], include_details=True)
    validation_metrics = evaluate(
        model, validation_theta, validation_labels, config["batch_size"], include_details=True
    )
    return {
        "blocks": blocks, "circuit_depth": model.circuit_depth(),
        "trainable_parameters": model.trainable_parameter_count(),
        "best_epoch": best_epoch, "stopping_epoch": epoch,
        "stopping_reason": stopping_reason, "train": train_metrics,
        "validation": validation_metrics, "runtime_seconds": float(time.perf_counter() - started),
        "history": history,
    }
