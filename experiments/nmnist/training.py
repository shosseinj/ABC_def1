import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from models.nmnist_qsnn import NMNISTTemporalQSNN


def evaluate(model, features, labels, batch_size, details=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs.append(model(features[start:start + batch_size]).cpu())
    logits = torch.cat(outputs)
    targets = np.asarray(labels, dtype=int)
    predictions = logits.argmax(1).numpy()
    result = {
        "loss": float(F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long))),
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }
    if details:
        result["per_class_accuracy"] = {
            str(label): float(np.mean(predictions[targets == label] == label)) for label in range(10)
        }
        result["confusion_matrix"] = confusion_matrix(targets, predictions, labels=range(10)).tolist()
    return result


def train_clean(config, train_features, train_labels, validation_features,
                validation_labels, checkpoint_path):
    seed = int(config["model_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x = torch.tensor(train_features, dtype=torch.float32)
    validation_x = torch.tensor(validation_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    model = NMNISTTemporalQSNN(
        config["n_qubits"], config["temporal_reupload_steps"], config["n_classes"]
    )
    with torch.no_grad():
        model.qlayer.weights.uniform_(-0.1, 0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    history = []
    stopping_reason = "max_epoch"
    started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed + epoch))
        batch_losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite N-MNIST loss at epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        train_metrics = evaluate(model, train_x, train_labels, config["batch_size"])
        validation_metrics = evaluate(model, validation_x, validation_labels, config["batch_size"])
        row = {
            "epoch": epoch, "train_loss": float(np.mean(batch_losses)),
            "train_accuracy": train_metrics["accuracy"], "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"[TRAIN] epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"macro_f1={validation_metrics['macro_f1']:.4f}", flush=True,
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "best_model_state": best_state, "best_validation_loss": best_loss,
            "best_epoch": best_epoch, "epoch": epoch, "stale": stale, "history": history,
        }, checkpoint_path.with_name(checkpoint_path.stem + "_latest.pt"))
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            stopping_reason = "early_stop"
            break

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoint_path)
    return {
        "best_epoch": best_epoch, "stopping_epoch": epoch,
        "stopping_reason": stopping_reason,
        "convergence_established": stopping_reason == "early_stop",
        "train": evaluate(model, train_x, train_labels, config["batch_size"], True),
        "validation": evaluate(model, validation_x, validation_labels, config["batch_size"], True),
        "runtime_seconds": float(time.perf_counter() - started),
        "parameters": model.trainable_parameter_count(), "qubits": model.n_qubits,
        "circuit_depth": model.circuit_depth(), "history": history,
    }
