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

from defenses.quantum_temp import (
    jensen_shannon,
    margin_stability_loss,
    measured_product_fidelity,
)
from encoding.ttfs import ttfs_encode
from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_splits
from experiments.iris.training import to_theta, train_iris_model
from models.qsnn import IrisQSNN
from scripts.run_quantum_temp_phase151 import train_validation_candidate


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
SPLIT_SEED = 42


def slug(value):
    return str(value).replace(".", "p")


def decision_summary(row):
    pgd = [item for item in row["validation_attacks"] if item["attack"] == "classical_timing"]
    random = [item for item in row["validation_attacks"] if item["attack"] == "random_jitter"]
    row["mean_pgd_attacked_margin"] = float(np.mean([item["mean_attacked_true_margin"] for item in pgd]))
    row["mean_pgd_margin_drop"] = float(np.mean([item["mean_margin_drop"] for item in pgd]))
    row["mean_pgd_positive_margin_fraction"] = float(np.mean([item["clean_correct_attacked_positive_margin_fraction"] for item in pgd]))
    row["mean_pgd_logit_drift"] = float(np.mean([item["mean_logit_drift"] for item in pgd]))
    row["mean_pgd_prediction_js"] = float(np.mean([item["mean_prediction_js"] for item in pgd]))
    row["mean_random_prediction_js"] = float(np.mean([item["mean_prediction_js"] for item in random]))
    return row


def candidate(name, config, Xval, yval):
    return decision_summary(train_validation_candidate(name, config, Xval, yval))


def eligible(rows, baseline):
    return [
        row for row in rows
        if row["clean_validation_accuracy"] >= baseline["clean_validation_accuracy"] - 0.02
        and np.isfinite(row["r_aux"])
        and row["clean_validation_macro_f1"] >= 0.80
    ]


def select(rows, baseline):
    valid = eligible(rows, baseline)
    if not valid:
        raise RuntimeError("No numerically stable candidate preserved validation accuracy.")
    return sorted(valid, key=lambda row: (
        round(row["mean_pgd_asr"], 12),
        -round(row["mean_pgd_positive_margin_fraction"], 12),
        -round(row["mean_pgd_attacked_margin"], 12),
        row["r_aux"],
    ))[0]


def write_ablation(stem, rows):
    fields = [
        "name", "classification_loss", "js_loss", "margin_loss", "quantum_loss",
        "weighted_js_loss", "weighted_margin_loss", "weighted_quantum_loss",
        "r_js", "r_margin", "r_q", "r_aux", "clean_validation_accuracy",
        "clean_validation_macro_f1", "perturbed_validation_accuracy", "mean_random_asr",
        "mean_pgd_asr", "mean_pgd_attacked_margin", "mean_pgd_margin_drop",
        "mean_pgd_positive_margin_fraction", "mean_pgd_logit_drift", "mean_pgd_prediction_js",
    ]
    (RESULTS / f"{stem}.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    with (RESULTS / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def gradient_analysis(configurations, Xval, yval):
    rows = []
    clean = to_theta(Xval, 100.0)
    times = torch.tensor(ttfs_encode(Xval, 100.0), dtype=torch.float32)
    perturbed = torch.clamp(times + 2.0, 0, 100)
    perturbed_angles = (torch.pi / 2.0) * perturbed / 100.0
    labels = torch.tensor(yval, dtype=torch.long)
    for name, checkpoint, config in configurations:
        model = IrisQSNN(4, 4, 3)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        clean_features = model.quantum_features(clean)
        perturbed_features = model.quantum_features(perturbed_angles)
        clean_logits = model.head(clean_features)
        perturbed_logits = model.head(perturbed_features)
        losses = {
            "ce": torch.nn.functional.cross_entropy(clean_logits, labels),
            "js": jensen_shannon(clean_logits, perturbed_logits),
            "margin": margin_stability_loss(perturbed_logits, labels, config.get("margin_target", 0.1)),
            "quantum": (1.0 - measured_product_fidelity(clean_features, perturbed_features)).mean(),
        }
        for loss_name, loss in losses.items():
            gradients = torch.autograd.grad(
                loss, (model.qlayer.weights, model.head.weight), retain_graph=True, allow_unused=True
            )
            rows.append({
                "configuration": name,
                "loss": loss_name,
                "raw_loss": float(loss),
                "circuit_gradient_norm": float(torch.linalg.vector_norm(gradients[0])) if gradients[0] is not None else 0.0,
                "classifier_gradient_norm": float(torch.linalg.vector_norm(gradients[1])) if gradients[1] is not None else 0.0,
            })
    with (RESULTS / "iris_phase16_gradient_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def config_for(base, **values):
    return {
        **base,
        "split_seed": SPLIT_SEED,
        "quantum_temp_enabled": True,
        "quantum_temp_epsilon": 0.02,
        "lambda_pred": 0.0,
        "consistency_enabled": False,
        "normalize_quantum_loss": False,
        "lambda_q": 0.0,
        "lambda_js": 0.0,
        "lambda_margin": 0.0,
        "margin_target": 0.1,
        **values,
    }


def main():
    started = time.perf_counter()
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    _, Xval, _, _, yval, _, _ = load_iris_splits(
        seed=SPLIT_SEED, test_size=base["test_size"], val_size=base["val_size"]
    )
    baseline = candidate(
        "phase16_baseline",
        config_for(base, quantum_temp_enabled=False), Xval, yval,
    )
    fidelity = candidate("phase16_fidelity_q5", config_for(base, lambda_q=5.0), Xval, yval)

    js_rows = []
    for value in (0.1, 0.5, 1.0, 2.0, 5.0):
        row = candidate(f"phase16_js_{slug(value)}", config_for(base, lambda_js=value), Xval, yval)
        js_rows.append(row)
        print(f"JS={value} val={row['clean_validation_accuracy']:.3f} PGD={row['mean_pgd_asr']:.3f}")
    js_best = select(js_rows, baseline)
    write_ablation("iris_phase16_js_ablation", js_rows)

    margin_rows = []
    for value in (0.1, 0.5, 1.0, 2.0, 5.0):
        row = candidate(
            f"phase16_margin_{slug(value)}_target_0p1",
            config_for(base, lambda_margin=value, margin_target=0.1), Xval, yval,
        )
        margin_rows.append(row)
        print(f"MARGIN={value} target=.1 val={row['clean_validation_accuracy']:.3f} PGD={row['mean_pgd_asr']:.3f}")
    initial_margin_best = select(margin_rows, baseline)
    margin_lambda = initial_margin_best["config"]["lambda_margin"]
    for target in (0.05, 0.2):
        row = candidate(
            f"phase16_margin_{slug(margin_lambda)}_target_{slug(target)}",
            config_for(base, lambda_margin=margin_lambda, margin_target=target), Xval, yval,
        )
        margin_rows.append(row)
        print(f"MARGIN={margin_lambda} target={target} val={row['clean_validation_accuracy']:.3f} PGD={row['mean_pgd_asr']:.3f}")
    margin_best = select(margin_rows, baseline)
    write_ablation("iris_phase16_margin_ablation", margin_rows)

    combined_rows = []
    for name, quantum in (("js_margin", 0.0), ("js_margin_quantum", 5.0)):
        row = candidate(
            f"phase16_{name}",
            config_for(
                base,
                lambda_js=js_best["config"]["lambda_js"],
                lambda_margin=margin_best["config"]["lambda_margin"],
                margin_target=margin_best["config"]["margin_target"],
                lambda_q=quantum,
            ), Xval, yval,
        )
        combined_rows.append(row)
        print(f"{name} val={row['clean_validation_accuracy']:.3f} PGD={row['mean_pgd_asr']:.3f}")
    write_ablation("iris_phase16_combined_ablation", combined_rows)

    decision_candidates = [js_best, margin_best, *combined_rows]
    selected = select(decision_candidates, baseline)
    baseline_pgd = {row["epsilon_fraction"]: row["asr"] for row in baseline["validation_attacks"] if row["attack"] == "classical_timing"}
    selected_pgd = {row["epsilon_fraction"]: row["asr"] for row in selected["validation_attacks"] if row["attack"] == "classical_timing"}
    improved_budgets = sum(selected_pgd[key] < baseline_pgd[key] for key in baseline_pgd)
    initial_gate = selected["mean_pgd_asr"] < baseline["mean_pgd_asr"] and improved_budgets >= 2
    selected_config = {
        **selected["config"],
        "selection_data": "validation_only",
        "selected_checkpoint": selected["checkpoint"],
        "initial_validation_gate": initial_gate,
        "improved_pgd_budgets": improved_budgets,
    }
    (ROOT / "configs" / "iris_phase16_selected.json").write_text(
        json.dumps(selected_config, indent=2), encoding="utf-8"
    )

    gradient_analysis([
        ("baseline", ROOT / baseline["checkpoint"], baseline["config"]),
        ("fidelity_q5", ROOT / fidelity["checkpoint"], fidelity["config"]),
        ("selected_decision", ROOT / selected["checkpoint"], selected["config"]),
    ], Xval, yval)

    multiseed = []
    if initial_gate:
        for seed in (42, 777, 2026):
            for model_name, template in (("baseline", config_for(base, quantum_temp_enabled=False)), ("selected", selected["config"])):
                config = {**template, "seed": seed, "split_seed": SPLIT_SEED}
                row = candidate(f"phase16_multiseed_{seed}_{model_name}", config, Xval, yval)
                multiseed.append({"seed": seed, "model": model_name, **row})
    with (RESULTS / "iris_phase16_multiseed_validation.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": multiseed}, handle, indent=2)
    if multiseed:
        fields = ["seed", "model", "clean_validation_accuracy", "clean_validation_macro_f1", "perturbed_validation_accuracy", "mean_random_asr", "mean_pgd_asr", "mean_pgd_attacked_margin", "mean_pgd_positive_margin_fraction"]
        with (RESULTS / "iris_phase16_multiseed_validation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in multiseed)
    else:
        (RESULTS / "iris_phase16_multiseed_validation.csv").write_text("seed,model\n", encoding="utf-8")

    paired_by_seed = {}
    if multiseed:
        for seed in (42, 777, 2026):
            left = next(row for row in multiseed if row["seed"] == seed and row["model"] == "baseline")
            right = next(row for row in multiseed if row["seed"] == seed and row["model"] == "selected")
            paired_by_seed[seed] = {
                "clean_preserved": right["clean_validation_accuracy"] >= left["clean_validation_accuracy"] - 0.02,
                "pgd_improved": right["mean_pgd_asr"] < left["mean_pgd_asr"],
            }
    replication_gate = initial_gate and sum(item["pgd_improved"] and item["clean_preserved"] for item in paired_by_seed.values()) >= 2

    final_payload = {
        "status": "not_run_validation_gate_failed" if not replication_gate else "pending_final_test",
        "test_set_accessed": False,
        "baseline_validation": baseline,
        "fidelity_validation": fidelity,
        "selected_validation": selected,
        "selected_configuration": selected_config,
        "initial_validation_gate": initial_gate,
        "multiseed_replication_gate": replication_gate,
        "multiseed_directions": paired_by_seed,
        "total_development_runtime_seconds": time.perf_counter() - started,
        "final_test_runs": [],
    }
    (RESULTS / "iris_phase16_final_test.json").write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    (RESULTS / "iris_phase16_final_test.csv").write_text("status\n" + final_payload["status"] + "\n", encoding="utf-8")
    print(f"selected={selected['name']} initial_gate={initial_gate} replication_gate={replication_gate}")


if __name__ == "__main__":
    main()
