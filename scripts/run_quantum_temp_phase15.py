from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack
from experiments.iris.training import train_iris_model


def main():
    started = time.perf_counter()
    base = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    variants = {
        "baseline": {
            **base,
            "quantum_temp_enabled": False,
            "lambda_pred": 0.0,
            "consistency_enabled": False,
        },
        "qt_q": {
            **base,
            "quantum_temp_enabled": True,
            "quantum_temp_epsilon": 0.02,
            "lambda_q": 0.1,
            "lambda_pred": 0.0,
            "consistency_enabled": False,
        },
        "qt_qp": {
            **base,
            "quantum_temp_enabled": True,
            "quantum_temp_epsilon": 0.02,
            "lambda_q": 0.1,
            "lambda_pred": 0.1,
            "consistency_enabled": True,
        },
    }

    checkpoints = {}
    training = {}
    results_dir = ROOT / "results"
    for name, config in variants.items():
        checkpoint = ROOT / "checkpoints" / f"iris_qsnn_phase15_{name}.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint)
        checkpoints[name] = checkpoint
        training[name] = run["metrics"]
        (results_dir / f"iris_quantum_temp_{name}_training.json").write_text(
            json.dumps(
                {"config": config, "metrics": run["metrics"], "defense_history": run["defense_history"]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"trained={name} clean_acc={run['metrics']['accuracy']:.3f} "
            f"best_val={run['metrics']['best_val_accuracy']:.3f}"
        )

    rows = []
    T = float(base["time_window"])
    for defense, config in variants.items():
        checkpoint = checkpoints[defense]
        for fraction in (0.01, 0.02, 0.05, 0.10):
            epsilon = fraction * T
            for attack in ("random_jitter", "classical_timing", "temp_drift_reference"):
                row = evaluate_attack(
                    config,
                    checkpoint,
                    attack,
                    epsilon,
                    tau=0.10,
                    iterations=20,
                    step_size=epsilon / 5,
                )
                rows.append({"defense": defense, **row})
            for tau in (0.01, 0.05, 0.10):
                row = evaluate_attack(
                    config,
                    checkpoint,
                    "temp_drift_gradient",
                    epsilon,
                    tau=tau,
                    iterations=40,
                    step_size=epsilon / 10,
                    restarts=3,
                )
                rows.append({"defense": defense, **row})
            print(f"evaluated={defense} epsilon={fraction:.2f}")

    payload = {
        "schema_version": 1,
        "defense_definition": "measured-quantum-feature fidelity regularization",
        "epsilon_convention": "fraction of T; 0.02 means an absolute timing budget of 2 when T=100",
        "quantum_loss": "1 - product fidelity of per-qubit Z-measurement distributions",
        "consistency_loss": "symmetric KL divergence of class probabilities",
        "split_selection_optimizer_epochs_unchanged": True,
        "asr_definition": "untargeted flips among clean-correct test samples",
        "classical_metric_limitation": "eight-bin ISI TV is coarse for four Iris timings",
        "training_configurations": variants,
        "training_metrics": training,
        "total_runtime_seconds": time.perf_counter() - started,
        "attack_runs": rows,
    }
    (results_dir / "iris_quantum_temp_phase15.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (results_dir / "iris_quantum_temp_phase15.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
