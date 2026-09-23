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
from experiments.iris.phase171 import classwise_accuracy, phase171_gate
from experiments.iris.paired_analysis import paired_category, summarize_paired_outcomes
from experiments.iris.training import to_theta
from models.qsnn import IrisQSNN
from scripts.run_phase17_adversarial_training import candidate, phase17_config


RESULTS = ROOT / "results"
SEEDS = (42, 777, 2026)
FRACTIONS = (0.01, 0.02, 0.05, 0.10)
LAMBDAS = (0.10, 0.25, 0.50, 0.75, 1.00)


def pgd_rows(row):
    return {item["epsilon_fraction"]: item for item in row["validation_attacks"] if item["attack"] == "classical_timing"}


def clean_records(row, Xval, yval, val_ids):
    config = row["config"]
    model = IrisQSNN(config["n_qubits"], config["n_layers"], config["n_classes"])
    model.load_state_dict(torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        features = model.quantum_features(to_theta(Xval, config["time_window"]))
        logits = model.head(features)
        probabilities = torch.softmax(logits, dim=-1)
        labels = torch.tensor(yval, dtype=torch.long)
        margins = true_class_margin(logits, labels)
        predictions = logits.argmax(1)
    return [{
        "sample_id": int(val_ids[index]),
        "true_label": int(yval[index]),
        "prediction": int(predictions[index]),
        "clean_correct": bool(predictions[index] == labels[index]),
        "confidence": float(probabilities[index, labels[index]]),
        "margin": float(margins[index]),
        "measured_features": features[index].tolist(),
    } for index in range(len(yval))]


def paired_by_epsilon(baseline, defense):
    output = []
    for fraction in FRACTIONS:
        key = f"classical_timing:{fraction}"
        left = {row["sample_id"]: row for row in baseline["validation_attack_samples"][key]}
        right = {row["sample_id"]: row for row in defense["validation_attack_samples"][key]}
        rows = []
        for sample_id in sorted(left.keys() & right.keys()):
            if left[sample_id]["clean_correct"] and right[sample_id]["clean_correct"]:
                rows.append({"paired_category": paired_category(left[sample_id]["attack_success"], right[sample_id]["attack_success"])})
        output.append({"epsilon": fraction, **summarize_paired_outcomes(rows)})
    return output


def classwise_margins(lambda_adv, seed, row):
    output = []
    for class_id in range(3):
        clean_source = row["validation_attack_samples"]["classical_timing:0.01"]
        clean_members = [sample for sample in clean_source if sample["true_label"] == class_id]
        result = {
            "lambda_adv": lambda_adv, "seed": seed, "class": class_id,
            "clean_margin": float(np.mean([sample["clean_true_margin"] for sample in clean_members])),
            "clean_positive_fraction": float(np.mean([sample["clean_true_margin"] > 0 for sample in clean_members])),
        }
        for fraction in FRACTIONS:
            members = [sample for sample in row["validation_attack_samples"][f"classical_timing:{fraction}"] if sample["true_label"] == class_id]
            tag = int(fraction * 100)
            result[f"attacked_margin_{tag}"] = float(np.mean([sample["attacked_true_margin"] for sample in members]))
            result[f"attacked_positive_fraction_{tag}"] = float(np.mean([sample["attacked_true_margin"] > 0 for sample in members]))
            result[f"margin_drop_{tag}"] = result["clean_margin"] - result[f"attacked_margin_{tag}"]
        output.append(result)
    return output


def gradient_by_class(lambda_adv, row, Xval, yval):
    config = row["config"]
    model = IrisQSNN(config["n_qubits"], config["n_layers"], config["n_classes"])
    model.load_state_dict(torch.load(ROOT / row["checkpoint"], map_location="cpu", weights_only=True))
    T = float(config["time_window"])
    labels = torch.tensor(yval, dtype=torch.long)
    times = torch.tensor(ttfs_encode(Xval, T), dtype=torch.float32)
    adversarial = training_timing_pgd(model, times, labels, 0.02 * T, T=T, steps=1, step_size=0.02 * T, random_start=False)
    clean_logits = model(to_theta(Xval, T))
    adversarial_logits = model((torch.pi / 2) * adversarial / T)
    circuit = tuple(model.qlayer.parameters())
    classifier = tuple(model.head.parameters())
    output = []
    for class_id in range(3):
        mask = labels == class_id
        losses = {
            "clean_ce": F.cross_entropy(clean_logits[mask], labels[mask]),
            "adversarial_ce": F.cross_entropy(adversarial_logits[mask], labels[mask]),
            "margin": margin_stability_loss(adversarial_logits[mask], labels[mask], 0.1),
        }
        weights = {"clean_ce": 1.0, "adversarial_ce": lambda_adv, "margin": 0.5}
        for name, loss in losses.items():
            gradients = torch.autograd.grad(loss, circuit + classifier, retain_graph=True, allow_unused=True)
            q = [gradient.flatten() for gradient in gradients[:len(circuit)] if gradient is not None]
            h = [gradient.flatten() for gradient in gradients[len(circuit):] if gradient is not None]
            output.append({
                "lambda_adv": lambda_adv, "seed": 42, "class": class_id, "loss": name,
                "raw_loss": float(loss), "weight": weights[name], "weighted_loss": float(weights[name] * loss),
                "circuit_gradient_norm": float(torch.linalg.vector_norm(torch.cat(q))) if q else 0.0,
                "classifier_gradient_norm": float(torch.linalg.vector_norm(torch.cat(h))) if h else 0.0,
            })
    return output


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main():
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    Xtrain, Xval, ytrain, yval, _, _, val_ids = load_iris_train_validation(42, base["test_size"], base["val_size"])
    assert np.array_equal(np.bincount(ytrain), np.array([30, 30, 30]))
    baseline_config = phase17_config(base, adversarial_training_enabled=False, clean_ce_weight=1.0, lambda_adv=0.0)
    baseline = candidate("baseline", baseline_config, Xval, yval, val_ids)
    baseline_clean = clean_records(baseline, Xval, yval, val_ids)
    baseline_class = classwise_accuracy([{
        "true_label": row["true_label"], "clean_correct": row["clean_correct"]
    } for row in baseline_clean])

    ablations, full_ablation, seed42_pairs = [], [], {}
    for value in LAMBDAS:
        config = phase17_config(
            base, clean_ce_weight=1.0, lambda_adv=value, lambda_margin=0.5,
            margin_target=0.1, lambda_js=0.0, adversarial_training_steps=1,
            adversarial_training_step_size=2.0, adversarial_training_random_start=False,
        )
        row = candidate(f"phase171_lambda_{str(value).replace('.', 'p')}", config, Xval, yval, val_ids)
        clean = clean_records(row, Xval, yval, val_ids)
        classes = classwise_accuracy([{"true_label": item["true_label"], "clean_correct": item["clean_correct"]} for item in clean])
        pgd = pgd_rows(row)
        pairs = paired_by_epsilon(baseline, row)
        seed42_pairs[value] = pairs
        summary = {
            "lambda_adv": value, "clean_accuracy": row["clean_validation_accuracy"],
            "macro_F1": row["clean_validation_macro_f1"], "class0_acc": classes[0],
            "class1_acc": classes[1], "class2_acc": classes[2],
            "perturbed_validation_accuracy": pgd[0.02]["attacked_accuracy"],
            "PGD_1_ASR": pgd[0.01]["asr"], "PGD_2_ASR": pgd[0.02]["asr"],
            "PGD_5_ASR": pgd[0.05]["asr"], "PGD_10_ASR": pgd[0.10]["asr"],
            "clean_CE": row["classification_loss"], "adversarial_CE": row["adversarial_loss"],
            "weighted_adversarial_CE": row["weighted_adversarial_loss"],
            "margin_loss": row["margin_loss"], "weighted_margin_loss": row["weighted_margin_loss"],
            "total_loss": row["total_loss"], "R_adv": row["r_adv"],
            "R_margin": row["r_margin"], "R_total_aux": row["r_aux"],
        }
        ablations.append(summary)
        full_ablation.append({"summary": summary, "paired": pairs, "clean_samples": clean, "run": row})
        print(f"lambda_adv={value:.2f} clean={summary['clean_accuracy']:.3f} class2={classes[2]:.3f} PGD5={summary['PGD_5_ASR']:.3f} PGD10={summary['PGD_10_ASR']:.3f}")

    eligible = []
    for item in full_ablation:
        summary, row = item["summary"], item["run"]
        class_drops = [baseline_class[index] - summary[f"class{index}_acc"] for index in range(3)]
        pair_map = {pair["epsilon"]: pair for pair in item["paired"]}
        if (
            summary["clean_accuracy"] >= baseline["clean_validation_accuracy"] - 0.02 - 1e-12
            and max(class_drops) <= 0.10 + 1e-12
            and (pair_map[0.05]["net_gain"] > 0 or pair_map[0.10]["net_gain"] > 0)
            and pair_map[0.01]["net_gain"] >= 0 and pair_map[0.02]["net_gain"] >= 0
        ):
            eligible.append(item)
    selected_items = sorted(eligible, key=lambda item: (
        np.mean([item["summary"][f"PGD_{epsilon}_ASR"] for epsilon in (1, 2, 5, 10)]),
        item["summary"]["lambda_adv"],
    ))[:2]

    multiseed_rows, multiseed_json, degradation, margins = [], [], [], []
    gradient_rows = []
    candidate_gate_rows = []
    for selected_item in selected_items:
        value = selected_item["summary"]["lambda_adv"]
        seed_checks = []
        for seed in SEEDS:
            seed_baseline = candidate(f"multiseed_{seed}_baseline", {**baseline_config, "seed": seed}, Xval, yval, val_ids)
            config = {**selected_item["run"]["config"], "seed": seed}
            defense = candidate(f"phase171_lambda_{str(value).replace('.', 'p')}_seed_{seed}", config, Xval, yval, val_ids)
            left_clean = clean_records(seed_baseline, Xval, yval, val_ids)
            right_clean = clean_records(defense, Xval, yval, val_ids)
            left_map, right_map = ({row["sample_id"]: row for row in records} for records in (left_clean, right_clean))
            left_classes = classwise_accuracy([{"true_label": row["true_label"], "clean_correct": row["clean_correct"]} for row in left_clean])
            right_classes = classwise_accuracy([{"true_label": row["true_label"], "clean_correct": row["clean_correct"]} for row in right_clean])
            pairs = paired_by_epsilon(seed_baseline, defense)
            pair_map = {pair["epsilon"]: pair for pair in pairs}
            delta_clean = defense["clean_validation_accuracy"] - seed_baseline["clean_validation_accuracy"]
            for pair in pairs:
                multiseed_rows.append({
                    "lambda_adv": value, "seed": seed, "clean_accuracy": defense["clean_validation_accuracy"],
                    "Delta_clean": delta_clean, "class2_accuracy": right_classes[2], **pair,
                })
            for sample_id, left in left_map.items():
                right = right_map[sample_id]
                if left["clean_correct"] and not right["clean_correct"] and left["true_label"] == 2:
                    degradation.append({
                        "seed": seed, "lambda_adv": value, "sample_id": sample_id,
                        "baseline_pred": left["prediction"], "defense_pred": right["prediction"],
                        "baseline_margin": left["margin"], "defense_margin": right["margin"],
                        "baseline_confidence": left["confidence"], "defense_confidence": right["confidence"],
                        "measured_features": right["measured_features"],
                    })
            margins.extend(classwise_margins(value, seed, defense))
            seed_check = {
                "seed": seed, "clean_preserved": delta_clean >= -0.02 - 1e-12,
                "positive_large_epsilon_gain": pair_map[0.05]["net_gain"] > 0 or pair_map[0.10]["net_gain"] > 0,
                "negative_both_small_eps": pair_map[0.01]["net_gain"] < 0 and pair_map[0.02]["net_gain"] < 0,
                "delta_clean": delta_clean, "class2_delta": right_classes[2] - left_classes[2],
            }
            seed_checks.append(seed_check)
            multiseed_json.append({"lambda_adv": value, "seed": seed, "baseline": seed_baseline, "defense": defense, "paired": pairs, "seed_check": seed_check})
        candidate_gate_rows.append({
            "lambda_adv": value, "seeds": seed_checks,
            "mean_clean_delta": float(np.mean([row["delta_clean"] for row in seed_checks])),
            "class2_degradation_reduced": all(row["class2_delta"] >= -0.10 - 1e-12 for row in seed_checks),
        })
        gradient_rows.extend(gradient_by_class(value, selected_item["run"], Xval, yval))

    gate_passed = phase171_gate(candidate_gate_rows)
    frozen = selected_items[0]["run"]["config"] if selected_items else None
    candidate_config = {
        "selection_data": "validation_only", "candidate_count": len(selected_items),
        "candidates": [{**item["run"]["config"], "checkpoint": item["run"]["checkpoint"]} for item in selected_items],
        "carry_forward_candidate": frozen if gate_passed else None,
        "test_set_accessed": False,
    }
    (ROOT / "configs" / "iris_phase171_candidate.json").write_text(json.dumps(candidate_config, indent=2), encoding="utf-8")
    (RESULTS / "iris_phase171_lambda_adv_ablation.json").write_text(json.dumps({"runs": full_ablation}, indent=2), encoding="utf-8")
    write_csv(RESULTS / "iris_phase171_lambda_adv_ablation.csv", ablations, list(ablations[0]))
    (RESULTS / "iris_phase171_multiseed_validation.json").write_text(json.dumps({"runs": multiseed_json}, indent=2), encoding="utf-8")
    table_b_fields = ["lambda_adv", "seed", "clean_accuracy", "Delta_clean", "class2_accuracy", "N_common", "epsilon", "baseline_failures", "defense_failures", "rescued", "broken", "net_gain"]
    write_csv(RESULTS / "iris_phase171_multiseed_validation.csv", multiseed_rows, table_b_fields)
    class2_payload = {"degradation_samples": degradation, "classwise_margins": margins}
    (RESULTS / "iris_phase171_class2_analysis.json").write_text(json.dumps(class2_payload, indent=2), encoding="utf-8")
    table_c_fields = ["seed", "lambda_adv", "sample_id", "baseline_pred", "defense_pred", "baseline_margin", "defense_margin", "baseline_confidence", "defense_confidence", "measured_features"]
    write_csv(RESULTS / "iris_phase171_class2_analysis.csv", degradation, table_c_fields)
    gradient_fields = ["lambda_adv", "seed", "class", "loss", "raw_loss", "weight", "weighted_loss", "circuit_gradient_norm", "classifier_gradient_norm"]
    write_csv(RESULTS / "iris_phase171_gradient_by_class.csv", gradient_rows, gradient_fields)
    gate = {
        "status": "candidate_frozen_for_next_phase" if gate_passed else "negative_multiseed_gate_failed",
        "validation_gate_passed": gate_passed, "test_set_accessed": False,
        "final_test_runs": [], "candidate_checks": candidate_gate_rows,
        "class_balance": {"training_counts": np.bincount(ytrain).tolist(), "diagnostic_run": False},
        "frozen_pgd": {"steps": 1, "epsilon_fraction": 0.02, "step_size_fraction": 0.02, "random_start": False},
        "frozen_margin": {"lambda_margin": 0.5, "margin_target": 0.1},
    }
    (RESULTS / "iris_phase171_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"selected_lambdas={[item['summary']['lambda_adv'] for item in selected_items]} gate={gate_passed} test_set_accessed=False")


if __name__ == "__main__":
    main()
