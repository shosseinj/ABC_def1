from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.iris.attack_evaluation import evaluate_attack_data, load_frozen_model
from experiments.iris.data import load_iris_split_indices, load_iris_splits
from experiments.iris.paired_analysis import (
    CATEGORIES,
    membership_category,
    paired_category,
    summarize_paired_outcomes,
)
from experiments.iris.training import to_theta


RESULTS = ROOT / "results"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def clean_outputs(config, checkpoint, Xtest):
    model = load_frozen_model(config, checkpoint)
    with torch.no_grad():
        features = model.quantum_features(to_theta(Xtest, config["time_window"]))
        logits = model.head(features)
        probabilities = torch.softmax(logits, dim=-1)
    return model, features.numpy(), logits.numpy(), probabilities.numpy()


def category_drift_rows(config_fields, paired_rows):
    output = []
    for category in CATEGORIES:
        members = [row for row in paired_rows if row["paired_category"] == category]
        result = {**config_fields, "category": category, "count": len(members)}
        for metric in ("feature_drift", "logit_drift", "JS"):
            for model in ("baseline", "defense"):
                key = f"{model}_{metric}"
                result[key] = float(np.mean([row[key] for row in members])) if members else None
        output.append(result)
    return output


def main():
    started = time.perf_counter()
    baseline_config = json.loads((ROOT / "configs" / "iris.json").read_text())
    defense_config = json.loads((ROOT / "configs" / "iris_quantum_temp_selected.json").read_text())
    baseline_checkpoint = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    defense_checkpoint = ROOT / defense_config["selected_checkpoint"]
    _, _, Xtest, _, _, ytest, _ = load_iris_splits(
        seed=baseline_config["seed"],
        test_size=baseline_config["test_size"],
        val_size=baseline_config["val_size"],
    )
    _, _, sample_ids = load_iris_split_indices(
        seed=baseline_config["seed"],
        test_size=baseline_config["test_size"],
        val_size=baseline_config["val_size"],
    )
    _, baseline_features, baseline_logits, baseline_probabilities = clean_outputs(
        baseline_config, baseline_checkpoint, Xtest
    )
    _, defense_features, defense_logits, defense_probabilities = clean_outputs(
        defense_config, defense_checkpoint, Xtest
    )
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    defense_predictions = defense_probabilities.argmax(axis=1)

    membership_rows = []
    for index, sample_id in enumerate(sample_ids):
        baseline_correct = bool(baseline_predictions[index] == ytest[index])
        defense_correct = bool(defense_predictions[index] == ytest[index])
        category = membership_category(baseline_correct, defense_correct)
        baseline_sorted = np.sort(baseline_probabilities[index])
        defense_sorted = np.sort(defense_probabilities[index])
        membership_rows.append({
            "sample_id": int(sample_id),
            "test_position": index,
            "true_label": int(ytest[index]),
            "baseline_pred": int(baseline_predictions[index]),
            "defense_pred": int(defense_predictions[index]),
            "baseline_correct": baseline_correct,
            "defense_correct": defense_correct,
            "membership": category,
            "baseline_clean_confidence": float(baseline_probabilities[index].max()),
            "defense_clean_confidence": float(defense_probabilities[index].max()),
            "baseline_margin": float(baseline_sorted[-1] - baseline_sorted[-2]),
            "defense_margin": float(defense_sorted[-1] - defense_sorted[-2]),
            "baseline_clean_logits": baseline_logits[index].tolist(),
            "defense_clean_logits": defense_logits[index].tolist(),
            "baseline_measured_features": baseline_features[index].tolist(),
            "defense_measured_features": defense_features[index].tolist(),
            "clean_feature_change": float(np.linalg.norm(defense_features[index] - baseline_features[index])),
        })
    sets = {
        category: [row["sample_id"] for row in membership_rows if row["membership"] == category]
        for category in ("common_correct", "baseline_only_correct", "defense_only_correct", "both_wrong")
    }
    write_csv(
        RESULTS / "iris_phase152_clean_membership.csv",
        [{key: row[key] for key in (
            "sample_id", "true_label", "baseline_pred", "defense_pred",
            "baseline_correct", "defense_correct", "membership",
            "baseline_clean_confidence", "defense_clean_confidence",
        )} for row in membership_rows],
    )
    (RESULTS / "iris_phase152_clean_membership.json").write_text(
        json.dumps({"sets": sets, "samples": membership_rows}, indent=2), encoding="utf-8"
    )

    common_ids = set(sets["common_correct"])
    configurations = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        for attack in ("random_jitter", "classical_timing", "temp_drift_reference"):
            configurations.append((attack, fraction, 0.10, 20))
        for tau in (0.01, 0.05, 0.10):
            configurations.append(("temp_drift_gradient", fraction, tau, 40))

    summaries = []
    classical_details = []
    gradient_details = []
    drift_rows = []
    for attack, fraction, tau, iterations in configurations:
        epsilon = fraction * baseline_config["time_window"]
        kwargs = {
            "tau": tau,
            "iterations": iterations,
            "step_size": epsilon / (10 if attack == "temp_drift_gradient" else 5),
            "restarts": 3,
            "split": "test",
            "sample_ids": sample_ids,
            "return_samples": True,
        }
        baseline = evaluate_attack_data(
            baseline_config, baseline_checkpoint, attack, epsilon, Xtest, ytest, **kwargs
        )
        defense = evaluate_attack_data(
            defense_config, defense_checkpoint, attack, epsilon, Xtest, ytest, **kwargs
        )
        baseline_by_id = {row["sample_id"]: row for row in baseline["samples"]}
        defense_by_id = {row["sample_id"]: row for row in defense["samples"]}
        paired = []
        for sample_id in sorted(common_ids):
            base = baseline_by_id[sample_id]
            defended = defense_by_id[sample_id]
            row = {
                "attack": attack,
                "epsilon": fraction,
                "tau": tau,
                "sample_id": sample_id,
                "true_label": base["true_label"],
                "baseline_clean_pred": base["clean_prediction"],
                "defense_clean_pred": defended["clean_prediction"],
                "baseline_attacked_pred": base["attacked_prediction"],
                "defense_attacked_pred": defended["attacked_prediction"],
                "baseline_success": base["attack_success"],
                "defense_success": defended["attack_success"],
                "paired_category": paired_category(base["attack_success"], defended["attack_success"]),
                "baseline_feature_drift": base["measured_feature_drift"],
                "defense_feature_drift": defended["measured_feature_drift"],
                "baseline_logit_drift": base["logit_drift"],
                "defense_logit_drift": defended["logit_drift"],
                "baseline_JS": base["prediction_js"],
                "defense_JS": defended["prediction_js"],
            }
            paired.append(row)
        config_fields = {"attack": attack, "epsilon": fraction, "tau": tau}
        summaries.append({**config_fields, **summarize_paired_outcomes(paired)})
        drift_rows.extend(category_drift_rows(config_fields, paired))
        if attack == "classical_timing" and fraction in (0.05, 0.10):
            order = {"rescued_by_defense": 0, "broken_by_defense": 1, "both_fail": 2, "both_robust": 3}
            classical_details.extend(sorted(paired, key=lambda row: (order[row["paired_category"]], row["sample_id"])))
        if attack == "temp_drift_gradient" and fraction == 0.10:
            gradient_details.extend(paired)
        print(f"paired attack={attack} epsilon={fraction:.2f} tau={tau:.2f}")

    write_csv(RESULTS / "iris_phase152_paired_attack_summary.csv", summaries)
    (RESULTS / "iris_phase152_paired_attack_summary.json").write_text(
        json.dumps({
            "seed_policy": "seed 42 plus test position for both models",
            "primary_set": "common_clean_correct",
            "sets": sets,
            "summaries": summaries,
            "total_runtime_seconds": time.perf_counter() - started,
        }, indent=2), encoding="utf-8"
    )
    write_csv(RESULTS / "iris_phase152_classical_pgd_samples.csv", classical_details)
    write_csv(RESULTS / "iris_phase152_gradient_temp_drift_samples.csv", gradient_details)
    write_csv(RESULTS / "iris_phase152_drift_by_category.csv", drift_rows)


if __name__ == "__main__":
    main()
