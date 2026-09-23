from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack


def main():
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    T = float(config["time_window"])
    rows = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        baseline = {}
        for attack in ("random_jitter", "classical_timing"):
            baseline[attack] = evaluate_attack(
                config, checkpoint, attack, fraction * T, tau=0.10,
                iterations=20, step_size=fraction * T / 5,
            )
        for tau in (0.01, 0.05, 0.10):
            for attack in ("random_jitter", "classical_timing"):
                rows.append({**baseline[attack], "tau": tau})
            for attack in ("temp_drift_reference", "temp_drift_gradient"):
                row = evaluate_attack(
                    config, checkpoint, attack, fraction * T, tau=tau,
                    iterations=40, step_size=fraction * T / 10, restarts=3,
                )
                rows.append(row)
                print(
                    f"attack={attack} epsilon={fraction:.2f} tau={tau:.2f} "
                    f"asr={row['asr']:.3f} drift={row['trace_distance']:.4f} "
                    f"feasible={row['feasible_attack_fraction']:.3f}"
                )

    results = ROOT / "results"
    payload = {
        "schema_version": 1,
        "asr_definition": "untargeted flips among clean-correct test samples",
        "gradient_objective": "maximize product-state one-minus-fidelity",
        "stealth_surrogate": "mean absolute sorted-ISI displacement divided by T",
        "final_constraint": "exact classical_mismatch delta_cls <= tau",
        "total_runtime_seconds": time.perf_counter() - started,
        "runs": rows,
    }
    (results / "iris_attack_comparison_gradient.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (results / "iris_attack_comparison_gradient.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
