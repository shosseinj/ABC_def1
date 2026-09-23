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
    epsilon = 0.05 * float(config["time_window"])
    rows = []
    for attack in ("random_jitter", "temp_drift_reference"):
        row = evaluate_attack(config, checkpoint, attack, epsilon, tau=0.10)
        rows.append(row)
        print(json.dumps(row, indent=2))
    clean = {
        **rows[0], "attack": "clean", "epsilon": 0.0, "epsilon_fraction": 0.0,
        "attacked_accuracy": rows[0]["clean_accuracy"], "asr": 0.0,
        "asr_numerator": 0, "trace_distance": 0.0, "fidelity": 1.0,
        "one_minus_fidelity": 0.0, "delta_cls": 0.0,
    }
    output = {"asr_definition": "clean-correct samples changed to an incorrect class", "runs": [clean, *rows]}
    results = ROOT / "results"
    (results / "iris_attack_comparison.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    with (results / "iris_attack_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=clean.keys())
        writer.writeheader()
        writer.writerows(output["runs"])


if __name__ == "__main__":
    main()
