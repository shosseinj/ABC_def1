from pathlib import Path
import csv
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from defenses.quantum_temp import (
    measured_product_fidelity,
    perturb_spike_times,
    quantum_temp_loss,
    timing_to_angle_torch,
)
from encoding.ttfs import ttfs_encode
from experiments.iris.attack_evaluation import evaluate_attack, evaluate_attack_data
from experiments.iris.data import load_iris_splits
from experiments.iris.training import classification_metrics, to_theta, train_iris_model
from models.qsnn import IrisQSNN


RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"


def slug(value):
    return str(value).replace(".", "p")


def fixed_perturbed_validation(model, Xval, yval, config):
    T = float(config["time_window"])
    clean_times = torch.tensor(ttfs_encode(Xval, T), dtype=torch.float32)
    perturbed_times = perturb_spike_times(
        clean_times,
        0.02,
        T,
        torch.Generator().manual_seed(int(config["seed"]) + 1510),
    )
    clean_angles = timing_to_angle_torch(clean_times, T)
    perturbed_angles = timing_to_angle_torch(perturbed_times, T)
    model.eval()
    with torch.no_grad():
        clean_features = model.quantum_features(clean_angles)
        perturbed_features = model.quantum_features(perturbed_angles)
        predictions = model.head(perturbed_features).argmax(1).numpy()
        fidelity = measured_product_fidelity(clean_features, perturbed_features).mean()
    metrics = classification_metrics(yval, predictions)
    return metrics, float(fidelity)


def validation_attack_summary(config, checkpoint, Xval, yval):
    per_budget = []
    T = float(config["time_window"])
    for fraction in (0.01, 0.02, 0.05, 0.10):
        epsilon = fraction * T
        for attack in ("random_jitter", "classical_timing"):
            per_budget.append(
                evaluate_attack_data(
                    config,
                    checkpoint,
                    attack,
                    epsilon,
                    Xval,
                    yval,
                    tau=0.10,
                    iterations=20,
                    step_size=epsilon / 5,
                    split="validation",
                )
            )
    random_rows = [row for row in per_budget if row["attack"] == "random_jitter"]
    pgd_rows = [row for row in per_budget if row["attack"] == "classical_timing"]
    return {
        "mean_random_asr": float(np.mean([row["asr"] for row in random_rows])),
        "mean_pgd_asr": float(np.mean([row["asr"] for row in pgd_rows])),
        "mean_random_accuracy": float(np.mean([row["attacked_accuracy"] for row in random_rows])),
        "mean_pgd_accuracy": float(np.mean([row["attacked_accuracy"] for row in pgd_rows])),
        "per_budget": per_budget,
    }


def train_validation_candidate(name, config, Xval, yval):
    checkpoint = CHECKPOINTS / f"iris_qsnn_phase151_{name}.pt"
    summary_path = RESULTS / f"{name}_training_summary.json"
    if summary_path.is_file() and checkpoint.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached["config"] == config and not cached["test_evaluated"]:
            return cached
    run = train_iris_model(config, checkpoint_path=checkpoint, evaluate_test=False)
    perturbed_metrics, measured_fidelity = fixed_perturbed_validation(
        run["model"], Xval, yval, config
    )
    robustness = validation_attack_summary(config, checkpoint, Xval, yval)
    summary = {
        "name": name,
        "config": config,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "test_evaluated": run["test_evaluated"],
        "clean_validation_accuracy": run["validation_metrics"]["accuracy"],
        "clean_validation_macro_f1": run["validation_metrics"]["macro_f1"],
        "perturbed_validation_accuracy": perturbed_metrics["accuracy"],
        "perturbed_validation_macro_f1": perturbed_metrics["macro_f1"],
        "mean_measured_feature_fidelity": measured_fidelity,
        **run["loss_summary"],
        **{key: value for key, value in robustness.items() if key != "per_budget"},
        "validation_attacks": robustness["per_budget"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def eligible_and_ranked(rows, baseline_accuracy):
    eligible = [
        row for row in rows
        if row["clean_validation_accuracy"] >= baseline_accuracy - 0.02
        and all(math.isfinite(row[key]) for key in ("classification_loss", "r_aux"))
        and row["clean_validation_macro_f1"] >= 0.80
    ]
    if not eligible:
        raise RuntimeError("No stable configuration preserved validation performance.")
    return sorted(
        eligible,
        key=lambda row: (
            -row["clean_validation_accuracy"],
            row["mean_pgd_asr"],
            row["mean_random_asr"],
            -row["perturbed_validation_accuracy"],
            row["config"]["lambda_pred"],
            row["config"]["lambda_q"],
        ),
    )[0]


def write_ablation(stem, rows, fields):
    (RESULTS / f"{stem}.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    with (RESULTS / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def gradient_norms(checkpoint, config, Xval, yval):
    model = IrisQSNN(4, 4, 3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    clean_times = torch.tensor(ttfs_encode(Xval, config["time_window"]), dtype=torch.float32)
    perturbed = perturb_spike_times(
        clean_times, 0.02, config["time_window"],
        torch.Generator().manual_seed(int(config["seed"]) + 1511),
    )
    total, parts = quantum_temp_loss(
        model,
        timing_to_angle_torch(clean_times, config["time_window"]),
        timing_to_angle_torch(perturbed, config["time_window"]),
        torch.tensor(yval, dtype=torch.long),
        float(config["lambda_q"]),
        float(config["lambda_pred"]) if config["consistency_enabled"] else 0.0,
    )
    parameters = (model.qlayer.weights, model.head.weight)
    output = {}
    for name, loss in (
        ("ce", parts["classification"]),
        ("quantum", parts["quantum"]),
        ("consistency", parts["consistency"]),
    ):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        output[name] = {
            "circuit": float(torch.linalg.vector_norm(gradients[0])) if gradients[0] is not None else 0.0,
            "classifier": float(torch.linalg.vector_norm(gradients[1])) if gradients[1] is not None else 0.0,
        }
    output["total"] = float(total.detach())
    return output


def clean_test_metrics(config, checkpoint):
    _, _, Xtest, _, _, ytest, _ = load_iris_splits(
        seed=config["seed"], test_size=config["test_size"], val_size=config["val_size"]
    )
    model = IrisQSNN(4, 4, 3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(to_theta(Xtest, config["time_window"])).argmax(1).numpy()
    return classification_metrics(ytest, predictions)


def main():
    started = time.perf_counter()
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    _, Xval, _, _, yval, _, _ = load_iris_splits(
        seed=base["seed"], test_size=base["test_size"], val_size=base["val_size"]
    )
    baseline_val = train_validation_candidate(
        "phase151_baseline_validation",
        {**base, "quantum_temp_enabled": False, "lambda_q": 0.0, "lambda_pred": 0.0, "consistency_enabled": False},
        Xval,
        yval,
    )
    baseline_accuracy = baseline_val["clean_validation_accuracy"]

    q_rows = []
    for lambda_q in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        config = {
            **base,
            "quantum_temp_enabled": True,
            "quantum_temp_epsilon": 0.02,
            "lambda_q": lambda_q,
            "lambda_pred": 0.0,
            "consistency_enabled": False,
        }
        q_rows.append(train_validation_candidate(f"phase151_q_{slug(lambda_q)}", config, Xval, yval))
        print(f"lambda_q={lambda_q} val={q_rows[-1]['clean_validation_accuracy']:.3f} pgd_asr={q_rows[-1]['mean_pgd_asr']:.3f}")
    q_best = eligible_and_ranked(q_rows, baseline_accuracy)
    write_ablation(
        "iris_quantum_temp_lambda_q_ablation",
        q_rows,
        [
            "name", "classification_loss", "quantum_loss", "weighted_quantum_loss", "r_q",
            "clean_validation_accuracy", "perturbed_validation_accuracy",
            "clean_validation_macro_f1", "perturbed_validation_macro_f1",
            "mean_measured_feature_fidelity", "mean_random_asr", "mean_pgd_asr",
        ],
    )

    pred_rows = []
    for lambda_pred in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        if lambda_pred == 0.0:
            row = q_best
        else:
            config = {
                **base,
                "quantum_temp_enabled": True,
                "quantum_temp_epsilon": 0.02,
                "lambda_q": q_best["config"]["lambda_q"],
                "lambda_pred": lambda_pred,
                "consistency_enabled": True,
            }
            row = train_validation_candidate(
                f"phase151_qp_{slug(q_best['config']['lambda_q'])}_{slug(lambda_pred)}",
                config,
                Xval,
                yval,
            )
        pred_rows.append(row)
        print(f"lambda_pred={lambda_pred} val={row['clean_validation_accuracy']:.3f} pgd_asr={row['mean_pgd_asr']:.3f}")
    final = eligible_and_ranked(pred_rows, baseline_accuracy)
    write_ablation(
        "iris_quantum_temp_lambda_pred_ablation",
        pred_rows,
        [
            "name", "classification_loss", "quantum_loss", "consistency_loss",
            "weighted_quantum_loss", "weighted_consistency_loss", "r_q", "r_pred", "r_aux",
            "clean_validation_accuracy", "perturbed_validation_accuracy",
            "clean_validation_macro_f1", "perturbed_validation_macro_f1",
            "mean_random_asr", "mean_pgd_asr",
        ],
    )

    selected_config = {
        **final["config"],
        "selection_data": "validation_only",
        "selection_priority": [
            "preserve clean validation accuracy", "reduce PGD ASR", "reduce random ASR",
            "improve perturbed validation accuracy", "prefer smaller regularization",
        ],
        "selected_checkpoint": final["checkpoint"],
    }
    selected_path = ROOT / "configs" / "iris_quantum_temp_selected.json"
    selected_path.write_text(json.dumps(selected_config, indent=2), encoding="utf-8")

    representative = {
        "baseline": (ROOT / baseline_val["checkpoint"], baseline_val["config"]),
        "lambda_q_0.1": (ROOT / q_rows[0]["checkpoint"], q_rows[0]["config"]),
        "lambda_q_best": (ROOT / q_best["checkpoint"], q_best["config"]),
        "final_qt_qp": (ROOT / final["checkpoint"], final["config"]),
    }
    gradients = {
        name: gradient_norms(checkpoint, config, Xval, yval)
        for name, (checkpoint, config) in representative.items()
    }

    # Test data is first evaluated here, after the selected configuration is persisted.
    final_models = {
        "baseline": (ROOT / "checkpoints" / "iris_qsnn_best.pt", base),
        "selected_quantum_temp": (ROOT / final["checkpoint"], final["config"]),
    }
    test_metrics = {
        name: clean_test_metrics(config, checkpoint)
        for name, (checkpoint, config) in final_models.items()
    }
    test_rows = []
    T = float(base["time_window"])
    for defense, (checkpoint, config) in final_models.items():
        for fraction in (0.01, 0.02, 0.05, 0.10):
            epsilon = fraction * T
            for attack in ("random_jitter", "classical_timing", "temp_drift_reference"):
                test_rows.append({
                    "defense": defense,
                    **evaluate_attack(
                        config, checkpoint, attack, epsilon, tau=0.10,
                        iterations=20, step_size=epsilon / 5,
                    ),
                })
            for tau in (0.01, 0.05, 0.10):
                test_rows.append({
                    "defense": defense,
                    **evaluate_attack(
                        config, checkpoint, "temp_drift_gradient", epsilon, tau=tau,
                        iterations=40, step_size=epsilon / 10, restarts=3,
                    ),
                })
        print(f"final_test_evaluated={defense}")

    payload = {
        "schema_version": 1,
        "selection_data": "validation_only",
        "test_evaluation_started_after_selected_config_was_written": True,
        "baseline_validation": baseline_val,
        "selected_lambda_q": q_best["config"]["lambda_q"],
        "selected_lambda_pred": final["config"]["lambda_pred"],
        "selection_reason": "lexicographic validation-only protocol recorded in selected config",
        "selected_configuration": selected_config,
        "gradient_norms": gradients,
        "clean_test_metrics": test_metrics,
        "test_attack_runs": test_rows,
        "total_runtime_seconds": time.perf_counter() - started,
        "statistical_note": "30 test samples; ASR denominator is expected near 26, so one success is about 3.85 percentage points",
    }
    (RESULTS / "iris_quantum_temp_phase151_final.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (RESULTS / "iris_quantum_temp_phase151_final.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=test_rows[0].keys())
        writer.writeheader()
        writer.writerows(test_rows)


if __name__ == "__main__":
    main()
