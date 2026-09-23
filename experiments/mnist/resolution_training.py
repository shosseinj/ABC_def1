import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mnist.capacity_training import _theta, evaluate
from models.mnist_qsnn import MNISTReducedQSNN


def train_resolution_variant(config, resolution, train_features, train_labels,
                             validation_features, validation_labels, checkpoint_path):
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_theta = _theta(train_features, config["time_window"])
    validation_theta = _theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNISTReducedQSNN(
        config["n_qubits"], config["reupload_blocks"], config["n_classes"],
        input_features=train_features.shape[1],
    )
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
    stopping_reason = "max_epoch"
    started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        minibatch_losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for resolution={resolution}, epoch={epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            minibatch_losses.append(float(loss.detach()))
        train_metrics = evaluate(model, train_theta, train_labels, config["batch_size"])
        validation_metrics = evaluate(model, validation_theta, validation_labels, config["batch_size"])
        row = {
            "resolution": resolution, "seed": seed, "epoch": epoch,
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
            print(
                f"[BEST] resolution={resolution}x{resolution} epoch={epoch} "
                f"val_loss={best_loss:.4f} val_acc={validation_metrics['accuracy']:.4f} "
                f"macro_f1={validation_metrics['macro_f1']:.4f}", flush=True,
            )
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_model_state": best_state, "history": history,
        }, checkpoint_path.with_name(checkpoint_path.stem + "_latest.pt"))
        print(
            f"[TRAIN] resolution={resolution}x{resolution} epoch={epoch} "
            f"train_loss={row['train_loss']:.4f} val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"macro_f1={validation_metrics['macro_f1']:.4f}", flush=True,
        )
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            stopping_reason = "early_stop"
            print(f"[EARLY STOP] resolution={resolution}x{resolution} epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    else:
        epoch = int(config["max_epochs"])
        print(f"[MAX EPOCH] resolution={resolution}x{resolution} reached {epoch}", flush=True)

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return {
        "resolution": resolution, "raw_features": resolution * resolution,
        "pca_features": config["pca_features"], "qubits": config["n_qubits"],
        "reupload_blocks": config["reupload_blocks"],
        "circuit_depth": model.circuit_depth(),
        "parameters": model.trainable_parameter_count(), "best_epoch": best_epoch,
        "stopping_epoch": epoch, "stopping_reason": stopping_reason,
        "convergence_established": stopping_reason == "early_stop",
        "train": evaluate(model, train_theta, train_labels, config["batch_size"], True),
        "validation": evaluate(model, validation_theta, validation_labels, config["batch_size"], True),
        "runtime_seconds": float(time.perf_counter() - started), "history": history,
    }
