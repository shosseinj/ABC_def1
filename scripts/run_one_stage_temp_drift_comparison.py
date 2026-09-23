"""Validation-only comparison of current/new one-stage TEMP-DRIFT and PGD."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_train_validation

SEEDS = (42, 777, 2026)
EPSILONS = (0.02, 0.05, 0.10)
ATTACKS = ("temp_drift_adaptive", "temp_drift_one_stage", "classical_timing")
SPLIT_SEED = 271


def main():
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    _, validation, _, labels, _, _, ids = load_iris_train_validation(
        seed=SPLIT_SEED, test_size=float(config["test_size"]),
        val_size=float(config["val_size"]),
    )
    T = float(config["time_window"])
    runs = []
    for seed in SEEDS:
        model_config = {**config, "seed": seed, "split_seed": SPLIT_SEED}
        checkpoint = ROOT / "checkpoints" / f"iris_phase21_baseline_split_{SPLIT_SEED}_model_{seed}.pt"
        for epsilon in EPSILONS:
            for attack in ATTACKS:
                row = evaluate_attack_data(
                    model_config, checkpoint, attack, epsilon * T, validation, labels,
                    tau=0.10, iterations=20, step_size=epsilon * T / 5,
                    split="validation", sample_ids=ids,
                )
                runs.append({**row, "model_seed": seed, "split_seed": SPLIT_SEED})
                print(f"seed={seed} epsilon={epsilon:.0%} attack={attack} ASR={row['asr']:.4f} 1-F={row['one_minus_fidelity']:.6f}")

    aggregate = []
    for epsilon in EPSILONS:
        for attack in ATTACKS:
            selected = [row for row in runs if row["epsilon_fraction"] == epsilon and row["attack"] == attack]
            numerator = sum(row["asr_numerator"] for row in selected)
            denominator = sum(row["asr_denominator"] for row in selected)
            aggregate.append({
                "epsilon_fraction": epsilon,
                "attack": attack,
                "asr": numerator / denominator if denominator else 0.0,
                "asr_numerator": numerator,
                "asr_denominator": denominator,
                "one_minus_fidelity": float(np.mean([row["one_minus_fidelity"] for row in selected])),
                "trace_distance": float(np.mean([row["trace_distance"] for row in selected])),
                "exact_feasibility_rate": float(np.mean([row["exact_feasibility_rate"] for row in selected])),
            })

    impacts = {}
    for epsilon in EPSILONS:
        rows = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == epsilon}
        current, new, pgd = rows["temp_drift_adaptive"], rows["temp_drift_one_stage"], rows["classical_timing"]
        asr_delta = new["asr"] - current["asr"]
        gap = max(pgd["one_minus_fidelity"] - current["one_minus_fidelity"], 0.0)
        material_drift_gain = new["one_minus_fidelity"] - current["one_minus_fidelity"] >= 0.05 * gap
        if asr_delta <= -0.05:
            impacts[epsilon] = "WORSE"
        elif asr_delta >= 0.0 and material_drift_gain:
            impacts[epsilon] = "IMPROVED"
        else:
            impacts[epsilon] = "NO MEANINGFUL IMPROVEMENT"
    conclusion = "BENEFICIAL" if any(value == "IMPROVED" for value in impacts.values()) and all(value != "WORSE" for value in impacts.values()) else "NOT BENEFICIAL"
    artifact = {
        "schema_version": 1, "validation_only": True,
        "test_features_accessed": False, "split_seed": SPLIT_SEED,
        "model_seeds": SEEDS, "epsilon_fractions": EPSILONS,
        "candidate_budget": 1600,
        "impact_rule": "IMPROVED requires nondecreasing ASR and closing at least 5% of the current-to-PGD 1-F gap; WORSE requires ASR loss >= 0.05",
        "runs": runs, "aggregate": aggregate,
        "impact": {str(key): value for key, value in impacts.items()},
        "conclusion": conclusion,
    }
    results = ROOT / "results"
    (results / "one_stage_temp_drift_comparison.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with (results / "one_stage_temp_drift_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
        writer.writeheader(); writer.writerows(aggregate)

    print("\nDetailed metrics")
    print("| epsilon | Attack | ASR | 1-Fidelity | Trace Distance | Exact feasibility |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in aggregate:
        print(f"| {row['epsilon_fraction']:.0%} | {row['attack']} | {row['asr']:.4f} | {row['one_minus_fidelity']:.6f} | {row['trace_distance']:.6f} | {row['exact_feasibility_rate']:.4f} |")
    print("\n| epsilon | Current ASR | New ASR | PGD ASR | Current 1-F | New 1-F | PGD 1-F | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for epsilon in EPSILONS:
        rows = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == epsilon}
        current, new, pgd = rows["temp_drift_adaptive"], rows["temp_drift_one_stage"], rows["classical_timing"]
        print(f"| {epsilon:.0%} | {current['asr']:.4f} | {new['asr']:.4f} | {pgd['asr']:.4f} | {current['one_minus_fidelity']:.6f} | {new['one_minus_fidelity']:.6f} | {pgd['one_minus_fidelity']:.6f} | {impacts[epsilon]} |")
    print(f"\nONE-STAGE TEMP-DRIFT UPDATE: {conclusion}")


if __name__ == "__main__":
    main()
