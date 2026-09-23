from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack


def write_results(name, metadata, rows):
    results = ROOT / "results"
    (results / f"{name}.json").write_text(
        json.dumps({**metadata, "runs": rows}, indent=2), encoding="utf-8"
    )
    with (results / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    T = float(config["time_window"])
    rows = []
    for fraction in (0.01, 0.02, 0.05, 0.10):
        for attack in ("random_jitter", "classical_timing", "temp_drift_reference"):
            row = evaluate_attack(
                config, checkpoint, attack, fraction * T, tau=0.10,
                iterations=20, step_size=fraction * T / 5,
            )
            rows.append(row)
            print(
                f"attack={attack} epsilon={fraction:.2f} "
                f"acc={row['attacked_accuracy']:.3f} asr={row['asr']:.3f} "
                f"drift={row['trace_distance']:.4f}"
            )
    write_results(
        "iris_attack_comparison_phase14",
        {
            "schema_version": 1,
            "asr_definition": "untargeted flips among clean-correct test samples",
            "classical_objective": "maximize true-label cross-entropy",
            "constraint": "timing L_inf only; stealth metrics are reported, not optimized",
            "total_runtime_seconds": time.perf_counter() - started,
        },
        rows,
    )


if __name__ == "__main__":
    main()
