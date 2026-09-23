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
import torch.nn.functional as F

from attacks.training_timing import training_timing_pgd
from defenses.quantum_temp import jensen_shannon, margin_stability_loss
from encoding.ttfs import ttfs_encode
from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_split_indices, load_iris_splits
from experiments.iris.paired_analysis import paired_category, summarize_paired_outcomes
from experiments.iris.training import to_theta, train_iris_model
from models.qsnn import IrisQSNN


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SPLIT_SEED = 42
SEEDS = (42, 777, 2026)
EPSILON_FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def phase17_config(base, **values):
    return {
        **base,
        "split_seed": SPLIT_SEED,
        "quantum_temp_enabled": False,
        "lambda_q": 0.0,
        "consistency_enabled": False,
        "adversarial_training_enabled": True,
        "adversarial_training_epsilon": 0.02,
        "adversarial_training_steps": 1,
        "adversarial_training_step_size": None,
        "adversarial_training_random_start": False,
        "clean_ce_weight": 0.5,
        "lambda_adv": 0.5,
        "lambda_js": 0.0,
        "lambda_margin": 0.0,
        "margin_target": 0.1,
        **values,
    }


def attack_summary(config, checkpoint, Xval, yval, sample_ids):
    attacks = []
    samples = {}
    T = float(config["time_window"])
    for fraction in EPSILON_FRACTIONS:
        epsilon = fraction * T
        for attack in ("random_jitter", "classical_timing"):
            result = evaluate_attack_data(
                config, checkpoint, attack, epsilon, Xval, yval,
                iterations=20, step_size=epsilon / 5, split="validation",
                sample_ids=sample_ids, return_samples=True,
            )
            attacks.append(result["summary"])
            samples[f"{attack}:{fraction}"] = result["samples"]
    pgd = [row for row in attacks if row["attack"] == "classical_timing"]
    random = [row for row in attacks if row["attack"] == "random_jitter"]
    return {
        "mean_random_asr": float(np.mean([row["asr"] for row in random])),
        "mean_pgd_asr": float(np.mean([row["asr"] for row in pgd])),
        "mean_pgd_attacked_accuracy": float(np.mean([row["attacked_accuracy"] for row in pgd])),
        "mean_pgd_attacked_margin": float(np.mean([row["mean_attacked_true_margin"] for row in pgd])),
        "mean_pgd_logit_drift": float(np.mean([row["mean_logit_drift"] for row in pgd])),
        "mean_pgd_prediction_js": float(np.mean([row["mean_prediction_js"] for row in pgd])),
        "validation_attacks": attacks,
        "validation_attack_samples": samples,
    }


def candidate(name, config, Xval, yval, sample_ids):
    checkpoint = CHECKPOINTS / f"iris_qsnn_phase17_{name}.pt"
    cache = RESULTS / f"phase17_{name}_training_summary.json"
    if cache.is_file() and checkpoint.is_file():
        row = json.loads(cache.read_text(encoding="utf-8"))
        if row["config"] == config and row["test_evaluated"] is False:
            return add_classwise_metrics(row)
    run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
    robustness = attack_summary(config, checkpoint, Xval, yval, sample_ids)
    row = {
        "name": name,
        "config": config,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "test_evaluated": run["test_evaluated"],
        "clean_validation_accuracy": run["validation_metrics"]["accuracy"],
        "clean_validation_macro_f1": run["validation_metrics"]["macro_f1"],
        **run["loss_summary"],
        **robustness,
    }
    cache.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return add_classwise_metrics(row)


def add_classwise_metrics(row):
    classwise = []
    for key, samples in row["validation_attack_samples"].items():
        attack, fraction = key.split(":")
        for class_id in range(int(row["config"]["n_classes"])):
            members = [sample for sample in samples if sample["true_label"] == class_id]
            clean_correct = [sample for sample in members if sample["clean_correct"]]
            classwise.append({
                "attack": attack,
                "epsilon_fraction": float(fraction),
                "class": class_id,
                "support": len(members),
                "clean_accuracy": sum(sample["clean_correct"] for sample in members) / len(members) if members else 0.0,
                "attacked_accuracy": sum(sample["attacked_prediction"] == class_id for sample in members) / len(members) if members else 0.0,
                "asr": sum(sample["attack_success"] for sample in clean_correct) / len(clean_correct) if clean_correct else 0.0,
            })
    row["classwise_validation"] = classwise
    return row


def select_candidate(rows, baseline):
    eligible = [
        row for row in rows
        if row["clean_validation_accuracy"] >= baseline["clean_validation_accuracy"] - 0.02 - 1e-12
        and np.isfinite(row["total_loss"])
    ]
    ranked_pool = eligible or rows
    selected = sorted(ranked_pool, key=lambda row: (
        round(row["mean_pgd_asr"], 12),
        -round(row["mean_pgd_attacked_margin"], 12),
        row["config"]["adversarial_training_steps"],
        row["r_aux"],
    ))[0]
    return selected, bool(eligible)


def write_ablation(stem, rows):
    (RESULTS / f"{stem}.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    fields = [
        "name", "clean_validation_accuracy", "clean_validation_macro_f1",
        "mean_random_asr", "mean_pgd_asr", "mean_pgd_attacked_accuracy",
        "mean_pgd_attacked_margin", "mean_pgd_logit_drift", "mean_pgd_prediction_js",
        "classification_loss", "adversarial_loss", "js_loss", "margin_loss",
        "weighted_clean_loss", "weighted_adversarial_loss", "weighted_js_loss",
        "weighted_margin_loss", "total_loss", "r_adv", "r_js", "r_margin", "r_aux",
    ]
    with (RESULTS / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def paired_validation(baseline, selected):
    rows = []
    for fraction in EPSILON_FRACTIONS:
        key = f"classical_timing:{fraction}"
        baseline_samples = {row["sample_id"]: row for row in baseline["validation_attack_samples"][key]}
        selected_samples = {row["sample_id"]: row for row in selected["validation_attack_samples"][key]}
        paired = []
        for sample_id in sorted(baseline_samples.keys() & selected_samples.keys()):
            left, right = baseline_samples[sample_id], selected_samples[sample_id]
            if left["clean_correct"] and right["clean_correct"]:
                paired.append({
                    "sample_id": sample_id,
                    "true_label": left["true_label"],
                    "paired_category": paired_category(left["attack_success"], right["attack_success"]),
                })
        rows.append({"epsilon_fraction": fraction, **summarize_paired_outcomes(paired)})
    return rows


def gradient_analysis(configurations, Xval, yval):
    rows = []
    for name, checkpoint, config in configurations:
        T = float(config["time_window"])
        model = IrisQSNN(int(config["n_qubits"]), int(config["n_layers"]), int(config["n_classes"]))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        labels = torch.tensor(yval, dtype=torch.long)
        times = torch.tensor(ttfs_encode(Xval, T), dtype=torch.float32)
        adversarial_times = training_timing_pgd(
            model, times, labels, config.get("adversarial_training_epsilon", 0.02) * T,
            T=T, steps=int(config.get("adversarial_training_steps", 1)),
        )
        clean_logits = model(to_theta(Xval, T))
        adversarial_logits = model((torch.pi / 2) * adversarial_times / T)
        losses = {
            "clean_ce": F.cross_entropy(clean_logits, labels),
            "adversarial_ce": F.cross_entropy(adversarial_logits, labels),
            "js": jensen_shannon(clean_logits, adversarial_logits),
            "margin": margin_stability_loss(adversarial_logits, labels, config.get("margin_target", 0.1)),
        }
        weights = {
            "clean_ce": config.get("clean_ce_weight", 1.0),
            "adversarial_ce": config.get("lambda_adv", 0.0),
            "js": config.get("lambda_js", 0.0),
            "margin": config.get("lambda_margin", 0.0),
        }
        circuit = tuple(model.qlayer.parameters())
        classifier = tuple(model.head.parameters())
        for loss_name, loss in losses.items():
            gradients = torch.autograd.grad(loss, circuit + classifier, retain_graph=True, allow_unused=True)
            circuit_gradients = [gradient.flatten() for gradient in gradients[:len(circuit)] if gradient is not None]
            classifier_gradients = [gradient.flatten() for gradient in gradients[len(circuit):] if gradient is not None]
            rows.append({
                "configuration": name,
                "loss": loss_name,
                "raw_loss": float(loss.detach()),
                "weight": float(weights[loss_name]),
                "weighted_loss": float((weights[loss_name] * loss).detach()),
                "circuit_gradient_norm": float(torch.linalg.vector_norm(torch.cat(circuit_gradients))) if circuit_gradients else 0.0,
                "classifier_gradient_norm": float(torch.linalg.vector_norm(torch.cat(classifier_gradients))) if classifier_gradients else 0.0,
            })
    with (RESULTS / "iris_phase17_gradient_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    _, Xval, _, _, yval, _, _ = load_iris_splits(SPLIT_SEED, base["test_size"], base["val_size"])
    _, val_ids, _ = load_iris_split_indices(SPLIT_SEED, base["test_size"], base["val_size"])
    baseline_config = phase17_config(base, adversarial_training_enabled=False, clean_ce_weight=1.0, lambda_adv=0.0)
    baseline = candidate("baseline", baseline_config, Xval, yval, val_ids)

    step_rows = []
    for steps in (1, 3, 5):
        row = candidate(f"adv_ce_pgd{steps}", phase17_config(base, adversarial_training_steps=steps), Xval, yval, val_ids)
        step_rows.append(row)
        print(f"PGD-{steps}: clean={row['clean_validation_accuracy']:.3f} mean_PGD_ASR={row['mean_pgd_asr']:.3f}")
    selected_step_row, step_selection_eligible = select_candidate(step_rows, baseline)
    selected_steps = selected_step_row["config"]["adversarial_training_steps"]
    write_ablation("iris_phase17_pgd_steps_ablation", step_rows)

    objective_rows = [selected_step_row]
    variants = (
        ("adv_ce_js", {"lambda_js": 0.5}),
        ("adv_ce_margin", {"lambda_margin": 0.5}),
        ("adv_ce_js_margin", {"lambda_js": 0.5, "lambda_margin": 0.5}),
    )
    for name, values in variants:
        config = phase17_config(
            base, adversarial_training_steps=selected_steps,
            clean_ce_weight=1.0, lambda_adv=1.0, **values,
        )
        row = candidate(name, config, Xval, yval, val_ids)
        objective_rows.append(row)
        print(f"{name}: clean={row['clean_validation_accuracy']:.3f} mean_PGD_ASR={row['mean_pgd_asr']:.3f}")
    selected, objective_selection_eligible = select_candidate(objective_rows, baseline)
    write_ablation("iris_phase17_objective_ablation", objective_rows)
    gradient_analysis([
        ("baseline", ROOT / baseline["checkpoint"], baseline["config"]),
        ("adv_ce", ROOT / selected_step_row["checkpoint"], selected_step_row["config"]),
        ("selected", ROOT / selected["checkpoint"], selected["config"]),
    ], Xval, yval)

    baseline_pgd = {row["epsilon_fraction"]: row["asr"] for row in baseline["validation_attacks"] if row["attack"] == "classical_timing"}
    selected_pgd = {row["epsilon_fraction"]: row["asr"] for row in selected["validation_attacks"] if row["attack"] == "classical_timing"}
    improved_budgets = sum(selected_pgd[key] < baseline_pgd[key] for key in baseline_pgd)
    initial_gate = (
        objective_selection_eligible
        and selected["mean_pgd_asr"] < baseline["mean_pgd_asr"]
        and improved_budgets >= 2
    )
    selected_config = {
        **selected["config"],
        "selection_data": "validation_only",
        "selected_checkpoint": selected["checkpoint"],
        "improved_pgd_budgets": improved_budgets,
        "step_selection_eligible": step_selection_eligible,
        "objective_selection_eligible": objective_selection_eligible,
        "initial_validation_gate": initial_gate,
    }
    (ROOT / "configs" / "iris_phase17_selected.json").write_text(json.dumps(selected_config, indent=2), encoding="utf-8")

    multiseed = []
    paired = {}
    if initial_gate:
        for seed in SEEDS:
            seed_baseline = candidate(
                f"multiseed_{seed}_baseline", {**baseline_config, "seed": seed}, Xval, yval, val_ids,
            )
            seed_selected = candidate(
                f"multiseed_{seed}_selected", {**selected["config"], "seed": seed}, Xval, yval, val_ids,
            )
            multiseed.extend([
                {"seed": seed, "model": "baseline", **seed_baseline},
                {"seed": seed, "model": "selected", **seed_selected},
            ])
            paired[str(seed)] = paired_validation(seed_baseline, seed_selected)
    (RESULTS / "iris_phase17_multiseed_validation.json").write_text(
        json.dumps({"runs": multiseed, "paired": paired}, indent=2), encoding="utf-8"
    )
    fields = ["seed", "model", "clean_validation_accuracy", "clean_validation_macro_f1", "mean_random_asr", "mean_pgd_asr", "mean_pgd_attacked_accuracy", "mean_pgd_attacked_margin", "mean_pgd_logit_drift", "mean_pgd_prediction_js"]
    with (RESULTS / "iris_phase17_multiseed_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in multiseed)

    seed_directions = {}
    for seed in SEEDS if multiseed else ():
        left = next(row for row in multiseed if row["seed"] == seed and row["model"] == "baseline")
        right = next(row for row in multiseed if row["seed"] == seed and row["model"] == "selected")
        paired_rows = paired[str(seed)]
        seed_directions[str(seed)] = {
            "clean_preserved": right["clean_validation_accuracy"] >= left["clean_validation_accuracy"] - 0.02 - 1e-12,
            "positive_pgd_paired_net_gain": sum(row["net_gain"] for row in paired_rows) > 0,
            "mean_pgd_improved": right["mean_pgd_asr"] < left["mean_pgd_asr"],
            "no_epsilon_collapse": all(
                selected_row["asr"] <= baseline_row["asr"] + 1.0 / max(baseline_row["asr_denominator"], 1)
                for baseline_row, selected_row in zip(
                    [row for row in left["validation_attacks"] if row["attack"] == "classical_timing"],
                    [row for row in right["validation_attacks"] if row["attack"] == "classical_timing"],
                )
            ),
        }
    replication_gate = bool(seed_directions) and (
        sum(row["positive_pgd_paired_net_gain"] for row in seed_directions.values()) >= 2
        and all(row["clean_preserved"] and row["no_epsilon_collapse"] for row in seed_directions.values())
        and np.mean([row["mean_pgd_asr"] for row in multiseed if row["model"] == "selected"])
        < np.mean([row["mean_pgd_asr"] for row in multiseed if row["model"] == "baseline"])
    )
    gate = {
        "status": "validation_gate_passed_final_test_required" if replication_gate else "not_run_validation_gate_failed",
        "test_set_accessed": False,
        "initial_validation_gate": initial_gate,
        "multiseed_replication_gate": replication_gate,
        "seed_directions": seed_directions,
        "baseline_validation": baseline,
        "selected_validation": selected,
        "selected_configuration": selected_config,
        "development_runtime_seconds": time.perf_counter() - started,
        "final_test_runs": [],
    }
    (RESULTS / "iris_phase17_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"selected={selected['name']} steps={selected_steps} initial_gate={initial_gate} replication_gate={replication_gate}")
    if replication_gate:
        raise RuntimeError("Validation gate passed; frozen final-test evaluation is not yet implemented.")


if __name__ == "__main__":
    main()
