"""Train/validation-only diagnosis of clean QSNN model-seed sensitivity."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.data import load_iris_train_validation
from experiments.iris.training import set_seed, to_theta
from models.qsnn import IrisQSNN

SEEDS = (777, 6543)
SPLIT_SEED = 42


def norm(parameters):
    values = [parameter.grad.detach().flatten() for parameter in parameters if parameter.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0


def reproduce_gradients(config, xtrain, ytrain, xval, yval):
    set_seed(int(config["seed"]))
    model = IrisQSNN(4, 4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    tx, vx = to_theta(xtrain, config["time_window"]), to_theta(xval, config["time_window"])
    ty, vy = torch.tensor(ytrain, dtype=torch.long), torch.tensor(yval, dtype=torch.long)
    rows = []
    best_key = None
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); optimizer.zero_grad()
        loss = F.cross_entropy(model(tx), ty); loss.backward()
        qnorm = norm(model.qlayer.parameters()); hnorm = norm(model.head.parameters())
        optimizer.step(); model.eval()
        with torch.no_grad():
            logits = model(vx)
            val_loss = float(F.cross_entropy(logits, vy))
            val_accuracy = float((logits.argmax(1) == vy).float().mean())
        rows.append({"seed": config["seed"], "epoch": epoch,
                     "training_loss": float(loss.detach()), "validation_loss": val_loss,
                     "validation_accuracy": val_accuracy,
                     "quantum_gradient_norm": qnorm, "head_gradient_norm": hnorm})
        key = (-val_accuracy, val_loss, epoch)
        if best_key is None or key < best_key:
            best_key = key
    return rows, {"best_epoch": best_key[2], "best_validation_accuracy": -best_key[0],
                  "best_validation_loss": best_key[1]}


def checkpoint_diagnostics(seed, config, xtrain, ytrain, xval, yval, val_ids):
    model = IrisQSNN(4, 4, 3)
    checkpoint = ROOT / "checkpoints" / f"iris_clean_frozen_seed_{seed}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)); model.eval()
    tx, vx = to_theta(xtrain, config["time_window"]), to_theta(xval, config["time_window"])
    with torch.no_grad():
        train_features = model.quantum_features(tx).numpy()
        val_features = model.quantum_features(vx).numpy()
        logits = model.head(torch.tensor(val_features)).numpy()
    predictions = logits.argmax(1)
    true_margins = []
    pair_margins = logits[:, 1] - logits[:, 2]
    for index, label in enumerate(yval):
        true_margins.append(logits[index, label] - np.max(np.delete(logits[index], label)))
    centroids = {label: train_features[ytrain == label].mean(0) for label in range(3)}
    spreads = {label: float(np.mean(np.linalg.norm(train_features[ytrain == label] - centroids[label], axis=1)))
               for label in range(3)}
    class_metrics = {}
    for label in range(3):
        mask = yval == label
        class_metrics[str(label)] = {
            "accuracy": float(np.mean(predictions[mask] == label)),
            "errors": int(np.sum(predictions[mask] != label)),
            "n": int(mask.sum()),
            "mean_true_margin": float(np.mean(np.asarray(true_margins)[mask])),
            "minimum_true_margin": float(np.min(np.asarray(true_margins)[mask])),
            "mean_class1_minus_class2_logit": float(np.mean(pair_margins[mask])),
        }
    weight = model.head.weight.detach().numpy(); bias = model.head.bias.detach().numpy()
    return {
        "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "confusion_matrix": confusion_matrix(yval, predictions, labels=[0, 1, 2]).tolist(),
        "class_metrics": class_metrics,
        "misclassified_validation_ids": [int(val_ids[index]) for index in np.flatnonzero(predictions != yval)],
        "head_weight": weight.tolist(), "head_bias": bias.tolist(),
        "head_weight_norms": np.linalg.norm(weight, axis=1).tolist(),
        "head_class1_class2_weight_distance": float(np.linalg.norm(weight[1] - weight[2])),
        "head_class1_class2_bias_difference": float(bias[1] - bias[2]),
        "train_feature_centroids": {str(key): value.tolist() for key, value in centroids.items()},
        "train_feature_spread": {str(key): value for key, value in spreads.items()},
        "train_centroid_distance_class1_class2": float(np.linalg.norm(centroids[1] - centroids[2])),
        "class1_class2_separation_to_spread": float(
            np.linalg.norm(centroids[1] - centroids[2]) / max(spreads[1] + spreads[2], 1e-12)
        ),
        "validation_feature_mean": val_features.mean(0).tolist(),
        "validation_feature_sd": val_features.std(0, ddof=1).tolist(),
    }


def main():
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    xtrain, xval, ytrain, yval, _, _, val_ids = load_iris_train_validation(
        SPLIT_SEED, config["test_size"], config["val_size"]
    )
    baseline = json.loads((ROOT / "results" / "clean_qsnn_multiseed_baseline.json").read_text(encoding="utf-8"))
    frozen = {row["seed"]: row for row in baseline["rows"]}
    diagnostics, gradient_rows = [], []
    for seed in SEEDS:
        local = {**config, "seed": seed, "split_seed": SPLIT_SEED,
                 "quantum_temp_enabled": False, "adversarial_training_enabled": False,
                 "consistency_enabled": False}
        gradients, reproduction = reproduce_gradients(local, xtrain, ytrain, xval, yval)
        gradient_rows.extend(gradients)
        expected = frozen[seed]
        if reproduction["best_epoch"] != expected["best_epoch"] or not np.isclose(
                reproduction["best_validation_accuracy"], expected["validation_accuracy"]):
            raise RuntimeError(f"Seed {seed} diagnostic rerun did not reproduce checkpoint selection.")
        item = checkpoint_diagnostics(seed, local, xtrain, ytrain, xval, yval, val_ids)
        selected = gradients[expected["best_epoch"] - 1]
        item.update({
            "selected_epoch": expected["best_epoch"],
            "validation_accuracy": expected["validation_accuracy"],
            "validation_loss": expected["validation_cross_entropy"],
            "final_training_loss": gradients[-1]["training_loss"],
            "final_validation_loss": gradients[-1]["validation_loss"],
            "final_validation_accuracy": gradients[-1]["validation_accuracy"],
            "minimum_validation_loss": min(row["validation_loss"] for row in gradients),
            "selected_quantum_gradient_norm": selected["quantum_gradient_norm"],
            "selected_head_gradient_norm": selected["head_gradient_norm"],
            "mean_quantum_gradient_norm": float(np.mean([row["quantum_gradient_norm"] for row in gradients])),
            "mean_head_gradient_norm": float(np.mean([row["head_gradient_norm"] for row in gradients])),
            "last10_quantum_gradient_norm": float(np.mean([row["quantum_gradient_norm"] for row in gradients[-10:]])),
            "last10_head_gradient_norm": float(np.mean([row["head_gradient_norm"] for row in gradients[-10:]])),
        })
        diagnostics.append(item)

    results = ROOT / "results"
    output = {"validation_only": True, "held_out_accessed": False,
              "attacks_run": False, "defenses_run": False,
              "split_seed": SPLIT_SEED, "primary_seeds": SEEDS,
              "diagnostic_rerun_reproduced_frozen_selection": True,
              "seeds": diagnostics}
    (results / "clean_seed_sensitivity_diagnosis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (results / "clean_seed_sensitivity_gradients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=gradient_rows[0].keys())
        writer.writeheader(); writer.writerows(gradient_rows)
    for item in diagnostics:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
