from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from attacks.training_timing import training_timing_pgd
from defenses.quantum_temp import margin_stability_loss, true_class_margin
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_train_validation
from experiments.iris.phase172 import align_trajectory_epochs, centroid_statistics, class12_gap
from experiments.iris.training import to_theta, train_iris_model
from models.qsnn import IrisQSNN
from scripts.run_phase17_adversarial_training import phase17_config


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SEEDS = (42, 777, 2026)
SAMPLE_ID = 119


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def config_pair(base, seed):
    baseline = phase17_config(
        base, seed=seed, adversarial_training_enabled=False,
        clean_ce_weight=1.0, lambda_adv=0.0,
    )
    defense = phase17_config(
        base, seed=seed, clean_ce_weight=1.0, lambda_adv=0.5,
        lambda_margin=0.5, margin_target=0.1, lambda_js=0.0, lambda_q=0.0,
        adversarial_training_steps=1, adversarial_training_epsilon=0.02,
        adversarial_training_step_size=2.0, adversarial_training_random_start=False,
    )
    return baseline, defense


def epoch_observer(records, sample_index, val_ids):
    def observe(model, epoch, val_loss, val_accuracy, angles, labels):
        model.eval()
        with torch.no_grad():
            features = model.quantum_features(angles)
            logits = model.head(features)
            probabilities = torch.softmax(logits, dim=-1)
            margins = true_class_margin(logits, labels)
            predictions = logits.argmax(1)
        item = sample_index
        records.append({
            "epoch": epoch, "validation_loss": val_loss, "validation_accuracy": val_accuracy,
            "class2_accuracy": float((predictions[labels == 2] == 2).float().mean()),
            "logit_class1": float(logits[item, 1]), "logit_class2": float(logits[item, 2]),
            "logit2_minus_logit1": float(logits[item, 2] - logits[item, 1]),
            "true_margin": float(margins[item]), "prediction": int(predictions[item]),
            "confidence": float(probabilities[item, labels[item]]),
            "all_samples": [{
                "sample_id": int(val_ids[index]), "true_label": int(labels[index]),
                "prediction": int(predictions[index]), "probabilities": probabilities[index].tolist(),
                "logits": logits[index].tolist(), "true_margin": float(margins[index]),
                "logit2_minus_logit1": float(logits[index, 2] - logits[index, 1]),
                "measured_features": features[index].tolist(),
            } for index in range(len(labels))],
        })
    return observe


def final_tensors(model, Xval, yval, T):
    labels = torch.tensor(yval, dtype=torch.long)
    with torch.no_grad():
        features = model.quantum_features(to_theta(Xval, T))
        logits = model.head(features)
        probabilities = torch.softmax(logits, dim=-1)
        margins = true_class_margin(logits, labels)
        predictions = logits.argmax(1)
    return features, logits, probabilities, margins, predictions


def gradient_vector(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        gradient.flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
        for gradient, parameter in zip(gradients, parameters)
    ])


def cosine(left, right):
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator) if denominator > 0 else 0.0


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    Xtrain, Xval, ytrain, yval, _, train_ids, val_ids = load_iris_train_validation(42, base["test_size"], base["val_size"])
    sample_index = int(np.where(val_ids == SAMPLE_ID)[0][0])
    assert int(yval[sample_index]) == 2
    T = float(base["time_window"])

    trajectory_rows, trajectory_json, checkpoint_rows = [], [], []
    final_models = {}
    final_data = {}
    for seed in SEEDS:
        for model_name, config in zip(("baseline", "defense"), config_pair(base, seed)):
            records = []
            reconstructed = CHECKPOINTS / f"iris_qsnn_phase172_{seed}_{model_name}.pt"
            run = train_iris_model(
                config, checkpoint_path=reconstructed, evaluate_test=False,
                epoch_observer=epoch_observer(records, sample_index, val_ids),
            )
            align_trajectory_epochs(records, range(1, int(config["epochs"]) + 1))
            first_wrong = next((row["epoch"] for row in records if row["prediction"] != 2), None)
            recovered = bool(first_wrong and any(row["prediction"] == 2 for row in records if row["epoch"] > first_wrong))
            for row in records:
                compact = {key: value for key, value in row.items() if key != "all_samples"}
                trajectory_rows.append({"seed": seed, "model": model_name, **{key: compact[key] for key in (
                    "epoch", "logit_class1", "logit_class2", "logit2_minus_logit1", "true_margin", "prediction", "confidence"
                )}})
                checkpoint_rows.append({
                    "seed": seed, "model": model_name, "epoch": row["epoch"],
                    "validation_accuracy": row["validation_accuracy"], "validation_loss": row["validation_loss"],
                    "class2_accuracy": row["class2_accuracy"], "sample119_margin": row["true_margin"],
                    "sample119_prediction": row["prediction"], "is_selected_epoch": row["epoch"] == run["metrics"]["best_epoch"],
                })
            trajectory_json.append({
                "seed": seed, "model": model_name, "best_epoch": run["metrics"]["best_epoch"],
                "first_wrong_epoch": first_wrong, "later_recovered": recovered, "epochs": records,
            })
            final_models[(seed, model_name)] = run["model"]
            final_data[(seed, model_name)] = final_tensors(run["model"], Xval, yval, T)

    fragile_rows, margin_rows, centroid_rows = [], [], []
    state_by_sample = {int(sample_id): [] for sample_id in val_ids}
    for seed in SEEDS:
        baseline_features, baseline_logits, baseline_probs, baseline_margins, baseline_preds = final_data[(seed, "baseline")]
        defense_features, defense_logits, defense_probs, defense_margins, defense_preds = final_data[(seed, "defense")]
        for model_name, data in (("baseline", final_data[(seed, "baseline")]), ("defense", final_data[(seed, "defense")])):
            features, logits, probabilities, margins, predictions = data
            for index, sample_id in enumerate(val_ids):
                if yval[index] not in (1, 2):
                    continue
                if model_name == "defense" and baseline_preds[index] == yval[index] and predictions[index] != yval[index]:
                    category = "DEFENSE_FLIPPED"
                elif model_name == "defense" and margins[index] < baseline_margins[index] - 0.10:
                    category = "DEFENSE_MARGIN_DEGRADED"
                elif abs(float(margins[index])) < 0.10:
                    category = "LOW_MARGIN_STABLE"
                else:
                    category = "STABLE"
                fragile_rows.append({
                    "seed": seed, "model": model_name, "sample_id": int(sample_id),
                    "true_label": int(yval[index]), "margin": float(margins[index]),
                    "prediction": int(predictions[index]), "fragility_category": category,
                })
                state_by_sample[int(sample_id)].append((model_name, int(predictions[index]), float(margins[index])))
            class_mask = np.isin(yval, (1, 2))
            gaps = class12_gap(logits[class_mask].numpy(), yval[class_mask])
            class_labels = yval[class_mask]
            for class_id in (1, 2):
                values = margins[torch.tensor(yval == class_id)].numpy()
                class_gaps = gaps[class_labels == class_id]
                margin_rows.append({
                    "seed": seed, "model": model_name, "class": class_id,
                    "mean_margin": float(np.mean(values)), "median_margin": float(np.median(values)),
                    "std_margin": float(np.std(values, ddof=1)), "min_margin": float(np.min(values)),
                    "percentile10_margin": float(np.percentile(values, 10)),
                    "num_gap_lt_0": int(np.sum(class_gaps < 0)),
                    "num_gap_lt_005": int(np.sum(class_gaps < 0.05)),
                    "num_gap_lt_010": int(np.sum(class_gaps < 0.10)),
                })
            stats = centroid_statistics(features.numpy(), yval, features[sample_index].numpy())
            centroid_rows.append({"seed": seed, "model": model_name, **stats})

    seed_sensitive_ids = {
        sample_id for sample_id, states in state_by_sample.items()
        if len({prediction for _, prediction, _ in states}) > 1
    }
    for row in fragile_rows:
        if row["sample_id"] in seed_sensitive_ids and row["fragility_category"] == "STABLE":
            row["fragility_category"] = "SEED_SENSITIVE"
    thresholds = {
        str(threshold): sorted({row["sample_id"] for row in fragile_rows if abs(row["margin"]) < threshold})
        for threshold in (0.05, 0.10, 0.20)
    }

    seed42_baseline = [row for row in fragile_rows if row["seed"] == 42 and row["model"] == "baseline" and row["true_label"] == 2]
    controls = {
        "stable": max(seed42_baseline, key=lambda row: row["margin"])["sample_id"],
        "low_margin_stable": min((row for row in seed42_baseline if row["sample_id"] != SAMPLE_ID and row["margin"] > 0), key=lambda row: row["margin"])["sample_id"],
        "sample119": SAMPLE_ID,
    }

    gradient_rows, pressure_rows = [], []
    for seed in SEEDS:
        model = final_models[(seed, "defense")]
        train_labels = torch.tensor(ytrain, dtype=torch.long)
        train_times = torch.tensor(ttfs_encode(Xtrain, T), dtype=torch.float32)
        adversarial_times = training_timing_pgd(model, train_times, train_labels, 2.0, T=T, steps=1, step_size=2.0)
        train_times_grad = train_times.clone().requires_grad_(True)
        timing_logits = model((torch.pi / 2) * train_times_grad / T)
        per_sample_ce = F.cross_entropy(timing_logits, train_labels, reduction="none")
        timing_gradients = torch.autograd.grad(per_sample_ce.sum(), train_times_grad)[0]
        with torch.no_grad():
            clean_logits = model((torch.pi / 2) * train_times / T)
            adv_logits = model((torch.pi / 2) * adversarial_times / T)
            clean_losses = F.cross_entropy(clean_logits, train_labels, reduction="none")
            adv_losses = F.cross_entropy(adv_logits, train_labels, reduction="none")
            margin_losses = torch.relu(0.1 - true_class_margin(adv_logits, train_labels))
        for class_id in (0, 1, 2):
            mask = train_labels == class_id
            pressure_rows.append({
                "seed": seed, "class": class_id,
                "clean_ce": float(clean_losses[mask].mean()), "adversarial_ce": float(adv_losses[mask].mean()),
                "margin_loss": float(margin_losses[mask].mean()),
                "timing_displacement": float(torch.linalg.vector_norm(adversarial_times[mask] - train_times[mask], dim=1).mean()),
                "timing_gradient_norm": float(torch.linalg.vector_norm(timing_gradients[mask], dim=1).mean()),
            })

        labels = torch.tensor(yval, dtype=torch.long)
        val_times = torch.tensor(ttfs_encode(Xval, T), dtype=torch.float32)
        adv_val = training_timing_pgd(model, val_times, labels, 2.0, T=T, steps=1, step_size=2.0)
        parameters = tuple(model.parameters())
        for control_name, sample_id in controls.items():
            index = int(np.where(val_ids == sample_id)[0][0])
            label = labels[index:index + 1]
            clean_logit = model((torch.pi / 2) * val_times[index:index + 1] / T)
            adv_logit = model((torch.pi / 2) * adv_val[index:index + 1] / T)
            losses = (
                F.cross_entropy(clean_logit, label),
                F.cross_entropy(adv_logit, label),
                margin_stability_loss(adv_logit, label, 0.1),
            )
            vectors = [gradient_vector(loss, parameters) for loss in losses]
            gradient_rows.append({
                "seed": seed, "sample_id": sample_id, "control": control_name,
                "cos_clean_adv": cosine(vectors[0], vectors[1]),
                "cos_clean_margin": cosine(vectors[0], vectors[2]),
                "cos_adv_margin": cosine(vectors[1], vectors[2]),
                "clean_grad_norm": float(torch.linalg.vector_norm(vectors[0])),
                "adv_grad_norm": float(torch.linalg.vector_norm(vectors[1])),
                "margin_grad_norm": float(torch.linalg.vector_norm(vectors[2])),
            })

    write_csv(RESULTS / "iris_phase172_sample119_trajectory.csv", trajectory_rows)
    (RESULTS / "iris_phase172_sample119_trajectory.json").write_text(json.dumps({"runs": trajectory_json}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase172_fragile_samples.csv", fragile_rows)
    (RESULTS / "iris_phase172_fragile_samples.json").write_text(json.dumps({"samples": fragile_rows, "threshold_ids": thresholds, "controls": controls}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase172_margin_summary.csv", margin_rows)
    (RESULTS / "iris_phase172_margin_summary.json").write_text(json.dumps({"rows": margin_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase172_feature_centroids.csv", centroid_rows)
    (RESULTS / "iris_phase172_feature_centroids.json").write_text(json.dumps({"rows": centroid_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase172_gradient_conflict.csv", gradient_rows)
    write_csv(RESULTS / "iris_phase172_checkpoint_diagnostic.csv", checkpoint_rows)
    (RESULTS / "iris_phase172_diagnostics.json").write_text(json.dumps({
        "frozen_configs": {str(seed): dict(zip(("baseline", "defense"), config_pair(base, seed))) for seed in SEEDS},
        "controls": controls, "training_pressure": pressure_rows,
        "test_set_accessed": False, "test_loader_invoked": False,
    }, indent=2), encoding="utf-8")
    print(f"controls={controls} test_set_accessed=False")


if __name__ == "__main__":
    main()
