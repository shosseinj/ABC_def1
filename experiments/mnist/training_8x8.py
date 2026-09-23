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


def train_clean_sanity(config, train_features, train_labels, validation_features,
                       validation_labels, checkpoint_path):
    seed = int(config["sanity_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_theta = torch.tensor(angle_encode(ttfs_encode(train_features, config["time_window"]), config["time_window"]), dtype=torch.float32)
    validation_theta = torch.tensor(angle_encode(ttfs_encode(validation_features, config["time_window"]), config["time_window"]), dtype=torch.float32)
    train_targets = torch.tensor(train_labels, dtype=torch.long)
    model = MNISTReducedQSNN(config["n_qubits"], config["reupload_blocks"], config["n_classes"])
    with torch.no_grad():
        model.qlayer.weights.uniform_(-0.1, 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    stale = 0
    started = time.perf_counter()

    def evaluate():
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(validation_theta), config["batch_size"]):
                chunks.append(model(validation_theta[start:start + config["batch_size"]]).cpu())
        logits = torch.cat(chunks)
        predictions = logits.argmax(1).numpy()
        return {
            "validation_loss": float(F.cross_entropy(logits, torch.tensor(validation_labels, dtype=torch.long))),
            "validation_accuracy": float(np.mean(predictions == validation_labels)),
            "validation_macro_f1": float(f1_score(validation_labels, predictions, average="macro", zero_division=0)),
            "per_class_accuracy": {str(label): float(np.mean(predictions[np.asarray(validation_labels) == label] == label)) for label in range(10)},
            "confusion_matrix": confusion_matrix(validation_labels, predictions, labels=range(10)).tolist(),
        }

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_theta), generator=torch.Generator().manual_seed(seed + epoch))
        losses = []
        for start in range(0, len(order), config["batch_size"]):
            indices = order[start:start + config["batch_size"]]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_theta[indices]), train_targets[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate()
        train_loss = float(np.mean(losses))
        history.append({"seed": seed, "epoch": epoch, "train_loss": train_loss, **metrics})
        print(f"[TRAIN] seed={seed} epoch={epoch} train_loss={train_loss:.4f} val_loss={metrics['validation_loss']:.4f} val_acc={metrics['validation_accuracy']:.4f}", flush=True)
        if metrics["validation_loss"] < best_loss:
            best_loss = metrics["validation_loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= config["minimum_early_stop_epoch"] and stale >= config["patience"]:
            break

    model.load_state_dict(best_state)
    final_metrics = evaluate()
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return {
        "model": model, "history": history, "best_epoch": best_epoch,
        "runtime_seconds": float(time.perf_counter() - started), **final_metrics,
    }
