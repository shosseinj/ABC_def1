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
from experiments.iris.multiseed import (
    FROZEN_DEFENSE,
    PHASE153_SEEDS,
    aggregate_seed_summaries,
    apply_frozen_defense,
    sample_mean_sd,
)
from experiments.iris.paired_analysis import CATEGORIES, paired_category, summarize_paired_outcomes
from experiments.iris.training import to_theta, train_iris_model


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SPLIT_SEED = 42


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def clean_predictions(config, checkpoint, Xtest):
    model = load_frozen_model(config, checkpoint)
    with torch.no_grad():
        return model(to_theta(Xtest, config["time_window"])).argmax(1).numpy()


def train_pair(base, seed):
    common = {**base, "seed": seed, "split_seed": SPLIT_SEED}
    baseline_config = {
        **common,
        "quantum_temp_enabled": False,
        "lambda_q": 0.0,
        "lambda_pred": 0.0,
        "consistency_enabled": False,
        "normalize_quantum_loss": False,
    }
    defense_config = apply_frozen_defense(common)
    output = {}
    for name, config in (("baseline", baseline_config), ("quantum_temp", defense_config)):
        checkpoint = CHECKPOINTS / f"iris_phase153_seed_{seed}_{name}.pt"
        run = train_iris_model(config, checkpoint, evaluate_test=True)
        summary = {
            "seed": seed,
            "split_seed": SPLIT_SEED,
            "model": name,
            "config": config,
            "metrics": run["metrics"],
            "validation_metrics": run["validation_metrics"],
            "loss_summary": run["loss_summary"],
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        }
        (RESULTS / f"iris_phase153_seed_{seed}_{name}_training.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        output[name] = (config, checkpoint, run)
    return output


def main():
    started = time.perf_counter()
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    _, _, Xtest, _, _, ytest, _ = load_iris_splits(
        seed=SPLIT_SEED, test_size=base["test_size"], val_size=base["val_size"]
    )
    test_ids = load_iris_split_indices(
        seed=SPLIT_SEED, test_size=base["test_size"], val_size=base["val_size"]
    )[2]
    clean_rows = []
    per_seed_rows = []
    identity = {}
    drift_seed_rows = []

    configurations = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        for attack in ("random_jitter", "classical_timing", "temp_drift_reference"):
            configurations.append((attack, fraction, 0.10, 20))
        for tau in (0.01, 0.05, 0.10):
            configurations.append(("temp_drift_gradient", fraction, tau, 40))

    for seed in PHASE153_SEEDS:
        pair = train_pair(base, seed)
        baseline_config, baseline_checkpoint, baseline_run = pair["baseline"]
        defense_config, defense_checkpoint, defense_run = pair["quantum_temp"]
        baseline_pred = clean_predictions(baseline_config, baseline_checkpoint, Xtest)
        defense_pred = clean_predictions(defense_config, defense_checkpoint, Xtest)
        baseline_correct = baseline_pred == ytest
        defense_correct = defense_pred == ytest
        common_mask = baseline_correct & defense_correct
        common_ids = set(test_ids[common_mask].tolist())
        baseline_ids = test_ids[baseline_correct].tolist()
        defense_ids = test_ids[defense_correct].tolist()
        clean_rows.append({
            "seed": seed,
            "baseline_accuracy": baseline_run["metrics"]["accuracy"],
            "defense_accuracy": defense_run["metrics"]["accuracy"],
            "Delta_accuracy": defense_run["metrics"]["accuracy"] - baseline_run["metrics"]["accuracy"],
            "baseline_macro_F1": baseline_run["metrics"]["macro_f1"],
            "defense_macro_F1": defense_run["metrics"]["macro_f1"],
            "baseline_clean_correct": len(baseline_ids),
            "defense_clean_correct": len(defense_ids),
            "N_common": len(common_ids),
            "baseline_clean_correct_ids": json.dumps(baseline_ids),
            "defense_clean_correct_ids": json.dumps(defense_ids),
            "common_clean_correct_ids": json.dumps(sorted(common_ids)),
            "baseline_best_validation_accuracy": baseline_run["metrics"]["best_val_accuracy"],
            "defense_best_validation_accuracy": defense_run["metrics"]["best_val_accuracy"],
            "baseline_training_CE": baseline_run["loss_summary"]["classification_loss"],
            "defense_training_CE": defense_run["loss_summary"]["classification_loss"],
            "defense_quantum_loss": defense_run["loss_summary"]["quantum_loss"],
            "defense_weighted_quantum_loss": defense_run["loss_summary"]["weighted_quantum_loss"],
            "defense_R_q": defense_run["loss_summary"]["r_q"],
            "baseline_checkpoint": str(baseline_checkpoint.relative_to(ROOT)),
            "defense_checkpoint": str(defense_checkpoint.relative_to(ROOT)),
        })

        for attack, fraction, tau, iterations in configurations:
            epsilon = fraction * base["time_window"]
            kwargs = {
                "tau": tau,
                "iterations": iterations,
                "step_size": epsilon / (10 if attack == "temp_drift_gradient" else 5),
                "restarts": 3,
                "split": "test",
                "sample_ids": test_ids,
                "return_samples": True,
            }
            baseline = evaluate_attack_data(
                baseline_config, baseline_checkpoint, attack, epsilon, Xtest, ytest, **kwargs
            )
            defense = evaluate_attack_data(
                defense_config, defense_checkpoint, attack, epsilon, Xtest, ytest, **kwargs
            )
            baseline_samples = {row["sample_id"]: row for row in baseline["samples"]}
            defense_samples = {row["sample_id"]: row for row in defense["samples"]}
            paired = []
            current_categories = {}
            for sample_id in sorted(common_ids):
                left, right = baseline_samples[sample_id], defense_samples[sample_id]
                category = paired_category(left["attack_success"], right["attack_success"])
                current_categories[sample_id] = category
                paired.append({"paired_category": category})
                key = (sample_id, attack, fraction, tau)
                counts = identity.setdefault(key, {category: 0 for category in CATEGORIES})
                counts[category] += 1
                counts["times_common_correct"] = counts.get("times_common_correct", 0) + 1
                for model, sample in (("baseline", left), ("defense", right)):
                    counts.setdefault(f"{model}_feature", []).append(sample["measured_feature_drift"])
                    counts.setdefault(f"{model}_logit", []).append(sample["logit_drift"])
                    counts.setdefault(f"{model}_js", []).append(sample["prediction_js"])
            summary = summarize_paired_outcomes(paired)
            per_seed_rows.append({"seed": seed, "attack": attack, "epsilon": fraction, "tau": tau, **summary})
            for category in CATEGORIES:
                members = [
                    sample_id for sample_id in common_ids
                    if current_categories[sample_id] == category
                ]
                drift_row = {"seed": seed, "attack": attack, "epsilon": fraction, "tau": tau, "category": category, "count": len(members)}
                for model in ("baseline", "defense"):
                    for metric in ("feature", "logit", "js"):
                        values = [identity[(sample_id, attack, fraction, tau)][f"{model}_{metric}"][-1] for sample_id in members]
                        drift_row[f"{model}_{metric}_mean"] = float(np.mean(values)) if values else None
                drift_seed_rows.append(drift_row)
            print(f"seed={seed} attack={attack} epsilon={fraction:.2f} tau={tau:.2f}")

    aggregate = aggregate_seed_summaries(per_seed_rows)
    identity_rows = []
    for (sample_id, attack, epsilon, tau), counts in identity.items():
        identity_rows.append({
            "sample_id": sample_id,
            "attack": attack,
            "epsilon": epsilon,
            "tau": tau,
            "times_common_correct": counts["times_common_correct"],
            "times_rescued": counts["rescued_by_defense"],
            "times_broken": counts["broken_by_defense"],
            "times_both_fail": counts["both_fail"],
            "times_both_robust": counts["both_robust"],
        })

    drift_groups = {}
    for row in drift_seed_rows:
        key = (row["attack"], row["epsilon"], row["tau"], row["category"])
        drift_groups.setdefault(key, []).append(row)
    drift_summary = []
    for (attack, epsilon, tau, category), members in drift_groups.items():
        row = {"attack": attack, "epsilon": epsilon, "tau": tau, "category": category, "num_seeds": len(members), "total_samples": sum(item["count"] for item in members)}
        for model in ("baseline", "defense"):
            for metric in ("feature", "logit", "js"):
                values = [item[f"{model}_{metric}_mean"] for item in members if item[f"{model}_{metric}_mean"] is not None]
                mean, sd = sample_mean_sd(values)
                row[f"{model}_{metric}_mean"] = mean
                row[f"{model}_{metric}_SD"] = sd
        drift_summary.append(row)

    delta_mean, delta_sd = sample_mean_sd([row["Delta_accuracy"] for row in clean_rows])
    clean_payload = {
        "seeds": list(PHASE153_SEEDS),
        "split_seed": SPLIT_SEED,
        "frozen_defense": FROZEN_DEFENSE,
        "Delta_accuracy_mean": delta_mean,
        "Delta_accuracy_SD": delta_sd,
        "Delta_accuracy_min": min(row["Delta_accuracy"] for row in clean_rows),
        "Delta_accuracy_max": max(row["Delta_accuracy"] for row in clean_rows),
        "improved_seeds": sum(row["Delta_accuracy"] > 0 for row in clean_rows),
        "unchanged_seeds": sum(row["Delta_accuracy"] == 0 for row in clean_rows),
        "degraded_seeds": sum(row["Delta_accuracy"] < 0 for row in clean_rows),
        "runs": clean_rows,
    }
    write_csv(RESULTS / "iris_phase153_clean_per_seed.csv", clean_rows)
    (RESULTS / "iris_phase153_clean_per_seed.json").write_text(json.dumps(clean_payload, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase153_paired_per_seed.csv", per_seed_rows)
    (RESULTS / "iris_phase153_paired_per_seed.json").write_text(json.dumps({"runs": per_seed_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase153_multiseed_attack_summary.csv", aggregate)
    (RESULTS / "iris_phase153_multiseed_attack_summary.json").write_text(json.dumps({"runs": aggregate, "total_runtime_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase153_sample_reproducibility.csv", identity_rows)
    (RESULTS / "iris_phase153_sample_reproducibility.json").write_text(json.dumps({"runs": identity_rows}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase153_drift_summary.csv", drift_summary)


if __name__ == "__main__":
    main()
