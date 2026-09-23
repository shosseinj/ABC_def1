import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mnist.capacity_training import _theta, evaluate
from models.mnist_qsnn import MNISTReducedQSNN


def resume_capacity_variant(config, train_features, train_labels, validation_features,
                            validation_labels, model_checkpoint, recovery_checkpoint,
                            output_checkpoint):
    seed = int(config["seed"])
    blocks = int(config["reupload_blocks"])
    resume_epoch = int(config["resume_epoch"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_theta = _theta(train_features, config["time_window"])
    validation_theta = _theta(validation_features, config["time_window"])
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNISTReducedQSNN(config["n_qubits"], blocks, config["n_classes"])

    model_state = torch.load(model_checkpoint, map_location="cpu", weights_only=True)
    recovery = torch.load(recovery_checkpoint, map_location="cpu", weights_only=True)
    if int(recovery["epoch"]) != resume_epoch or int(recovery["best_epoch"]) != resume_epoch:
        raise RuntimeError("Recovery checkpoint does not describe the selected epoch-250 state.")
    if set(model_state) != set(recovery["model_state"]) or not all(
        torch.equal(model_state[key], recovery["model_state"][key]) for key in model_state
    ):
        raise RuntimeError("Raw and recovery checkpoint model states differ.")
    model.load_state_dict(model_state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    optimizer_restored = "optimizer_state" in recovery
    if optimizer_restored:
        optimizer.load_state_dict(recovery["optimizer_state"])
    history = list(recovery.get("history", []))
    if len(history) != resume_epoch or int(history[-1]["epoch"]) != resume_epoch:
        raise RuntimeError("Complete epoch-250 history is unavailable.")

    initial_validation = evaluate(model, validation_theta, validation_labels, config["batch_size"])
    best_loss = float(recovery["best_validation_loss"])
    if not np.isclose(initial_validation["loss"], best_loss, atol=1e-6):
        raise RuntimeError("Loaded model does not reproduce the recorded best validation loss.")
    best_epoch = int(recovery["best_epoch"])
    best_state = copy.deepcopy(model_state)
    best_metrics = initial_validation
    stale = 0
    stopping_reason = "max_epoch"
    started = time.perf_counter()

    for epoch in range(resume_epoch + 1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        minibatch_losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}.")
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
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation_metrics
            stale = 0
            print(
                f"[BEST] epoch={epoch} val_loss={best_loss:.4f} "
                f"val_acc={validation_metrics['accuracy']:.4f} "
                f"macro_f1={validation_metrics['macro_f1']:.4f}", flush=True,
            )
        else:
            stale += 1
        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
            "best_model_state": best_state, "history": history,
        }, output_checkpoint.with_name(output_checkpoint.stem + "_latest.pt"))
        print(
            f"[TRAIN] blocks=4 epoch={epoch}/{config['max_epochs']} "
            f"train_loss={row['train_loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} val_acc={validation_metrics['accuracy']:.4f} "
            f"macro_f1={validation_metrics['macro_f1']:.4f} patience={stale}/{config['patience']}",
            flush=True,
        )
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            stopping_reason = "early_stop"
            print(f"[EARLY STOP] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    else:
        epoch = int(config["max_epochs"])
        print(f"[MAX EPOCH] reached {epoch}", flush=True)

    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_theta, train_labels, config["batch_size"], True)
    best_metrics = evaluate(model, validation_theta, validation_labels, config["batch_size"], True)
    torch.save({
        "model_state": best_state, "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "best_validation_loss": best_loss, "best_epoch": best_epoch,
        "best_model_state": best_state, "history": history,
    }, output_checkpoint)
    return {
        "optimizer_restored": optimizer_restored, "history": history,
        "best_epoch": best_epoch, "stopping_epoch": epoch,
        "stopping_reason": stopping_reason, "train": train_metrics,
        "validation": best_metrics, "runtime_seconds": float(time.perf_counter() - started),
    }
