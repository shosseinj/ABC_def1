from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack


def main():
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    checkpoint = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    T = float(config["time_window"])
    rows = []
    for epsilon_fraction in (0.01, 0.02, 0.05, 0.10):
        for tau in (0.01, 0.05, 0.10):
            row = evaluate_attack(
                config, checkpoint, "temp_drift_reference", epsilon_fraction * T, tau
            )
            rows.append(row)
            print(
                f"epsilon={epsilon_fraction:.2f} tau={tau:.2f} "
                f"attacked_acc={row['attacked_accuracy']:.3f} asr={row['asr']:.3f}"
            )
    results = ROOT / "results"
    with (results / "iris_attack_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (results / "iris_attack_sweep.json").write_text(
        json.dumps({"optimizer": "randomized_reference", "runs": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
