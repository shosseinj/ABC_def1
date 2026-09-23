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


def _evaluate(model, theta, labels, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(theta), batch_size):
            chunks.append(model(theta[start:start + batch_size]).cpu())
    logits = torch.cat(chunks)
    targets = np.asarray(labels, dtype=int)
    predictions = logits.argmax(1).numpy()
    return {
        "validation_loss": float(F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long))),
        "validation_accuracy": float(np.mean(predictions == targets)),
        "validation_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "per_class_accuracy": {
            str(label): float(np.mean(predictions[targets == label] == label))
            for label in range(10)
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=range(10)).tolist(),
    }


def resume_clean_training(config, train_features, train_labels, validation_features,
                          validation_labels, checkpoint_path, prior_history,
                          output_path):
    seed = int(config["seed"])
    resume_epoch = int(config["resume_epoch"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_theta = _theta(train_features, config["time_window"])
    validation_theta = _theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNISTReducedQSNN(config["n_qubits"], config["reupload_blocks"], config["n_classes"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    rich_checkpoint = isinstance(payload, dict) and "model_state" in payload
    model.load_state_dict(payload["model_state"] if rich_checkpoint else payload)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    optimizer_restored = rich_checkpoint and "optimizer_state" in payload
    if optimizer_restored:
        optimizer.load_state_dict(payload["optimizer_state"])

    if rich_checkpoint and "history" in payload:
        prior_history = payload["history"]
    if not prior_history or int(prior_history[-1]["epoch"]) != resume_epoch:
        raise RuntimeError(f"Training history must end at epoch {resume_epoch}.")
    if rich_checkpoint and int(payload.get("epoch", -1)) != resume_epoch:
        raise RuntimeError(f"Checkpoint epoch must equal {resume_epoch}.")
    initial_metrics = _evaluate(model, validation_theta, validation_labels, config["batch_size"])
    recorded_loss = float(payload.get("best_validation_loss", prior_history[-1]["validation_loss"]))
    if not np.isclose(initial_metrics["validation_loss"], recorded_loss, atol=1e-6):
        raise RuntimeError("Loaded checkpoint does not reproduce its recorded validation loss.")

    best_loss = recorded_loss
    best_epoch = int(payload.get("best_epoch", resume_epoch))
    best_state = copy.deepcopy(payload.get("best_model_state", model.state_dict()))
    best_metrics = initial_metrics
    history = list(prior_history)
    stale = 0
    started = time.perf_counter()
    stopping_reason = "max_epoch"

    for epoch in range(resume_epoch + 1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = _evaluate(model, validation_theta, validation_labels, config["batch_size"])
        row = {
            "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_loss": metrics["validation_loss"],
            "validation_accuracy": metrics["validation_accuracy"],
            "validation_macro_f1": metrics["validation_macro_f1"],
        }
        history.append(row)
        if metrics["validation_loss"] < best_loss:
            best_loss = metrics["validation_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            stale = 0
            print(
                f"[BEST] epoch={epoch} val_loss={best_loss:.4f} "
                f"val_acc={metrics['validation_accuracy']:.4f} "
                f"macro_f1={metrics['validation_macro_f1']:.4f}", flush=True,
            )
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_model_state": best_state, "history": history,
        }, output_path.with_name(output_path.stem + "_latest.pt"))
        print(
            f"[TRAIN] epoch={epoch}/{config['max_epochs']} train_loss={row['train_loss']:.4f} "
            f"val_loss={metrics['validation_loss']:.4f} val_acc={metrics['validation_accuracy']:.4f} "
            f"macro_f1={metrics['validation_macro_f1']:.4f} patience={stale}/{config['patience']}",
            flush=True,
        )
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            stopping_reason = "early_stop"
            print(
                f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch} "
                f"best_val_loss={best_loss:.4f}", flush=True,
            )
            break
    else:
        epoch = int(config["max_epochs"])
        print(f"[MAX EPOCH] reached {epoch}", flush=True)

    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state, "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
        "best_model_state": best_state, "history": history,
    }, output_path)
    return {
        "history": history, "initial_metrics": initial_metrics,
        "best_metrics": best_metrics, "best_epoch": best_epoch,
        "stopping_epoch": epoch, "stopping_reason": stopping_reason,
        "optimizer_restored": bool(optimizer_restored),
        "runtime_seconds": float(time.perf_counter() - started),
    }
