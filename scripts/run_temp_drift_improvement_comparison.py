"""Validation-only comparison of three non-gradient TEMP-DRIFT searches."""
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


MODEL_SEEDS = (42, 777, 2026)
SPLIT_SEED = 271
EPSILON_FRACTIONS = (0.02, 0.05, 0.10)
ATTACKS = ("temp_drift_reference", "temp_drift_improved", "temp_drift_adaptive", "classical_timing")
MEANINGFUL_ASR_DELTA = 0.05


def main():
    config_path = ROOT / "configs" / "iris.json"
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    _, validation, _, validation_labels, _, _, validation_ids = load_iris_train_validation(
        seed=SPLIT_SEED,
        test_size=float(base_config["test_size"]),
        val_size=float(base_config["val_size"]),
    )
    T = float(base_config["time_window"])
    runs = []
    sample_rows = []
    for model_seed in MODEL_SEEDS:
        config = {**base_config, "seed": model_seed, "split_seed": SPLIT_SEED}
        checkpoint = ROOT / "checkpoints" / f"iris_phase21_baseline_split_{SPLIT_SEED}_model_{model_seed}.pt"
        for fraction in EPSILON_FRACTIONS:
            for attack in ATTACKS:
                result = evaluate_attack_data(
                    config,
                    checkpoint,
                    attack,
                    fraction * T,
                    validation,
                    validation_labels,
                    tau=0.10,
                    iterations=20,
                    step_size=fraction * T / 5,
                    split="validation",
                    sample_ids=validation_ids,
                    return_samples=True,
                )
                summary = {**result["summary"], "model_seed": model_seed, "split_seed": SPLIT_SEED}
                runs.append(summary)
                for sample in result["samples"]:
                    sample_rows.append({
                        "model_seed": model_seed,
                        "split_seed": SPLIT_SEED,
                        "epsilon_fraction": fraction,
                        "attack": attack,
                        **sample,
                    })
                print(
                    f"seed={model_seed} epsilon={fraction:.0%} attack={attack} "
                    f"ASR={summary['asr']:.4f} exact_feasible={summary['exact_feasibility_rate']:.4f}"
                )

    aggregate = []
    for fraction in EPSILON_FRACTIONS:
        for attack in ATTACKS:
            selected = [row for row in runs if row["epsilon_fraction"] == fraction and row["attack"] == attack]
            successes = sum(row["asr_numerator"] for row in selected)
            denominator = sum(row["asr_denominator"] for row in selected)
            aggregate.append({
                "epsilon_fraction": fraction,
                "attack": attack,
                "asr": successes / denominator if denominator else 0.0,
                "asr_numerator": successes,
                "asr_denominator": denominator,
                "one_minus_fidelity": float(np.mean([row["one_minus_fidelity"] for row in selected])),
                "trace_distance": float(np.mean([row["trace_distance"] for row in selected])),
                "exact_feasibility_rate": float(np.mean([row["exact_feasibility_rate"] for row in selected])),
            })

    impacts = {}
    for fraction in EPSILON_FRACTIONS:
        by_attack = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == fraction}
        delta = by_attack["temp_drift_adaptive"]["asr"] - by_attack["temp_drift_improved"]["asr"]
        impacts[fraction] = (
            "IMPROVED" if delta >= MEANINGFUL_ASR_DELTA
            else "WORSE" if delta <= -MEANINGFUL_ASR_DELTA
            else "NO MEANINGFUL IMPROVEMENT"
        )
    conclusion = "BENEFICIAL" if any(value == "IMPROVED" for value in impacts.values()) and all(value != "WORSE" for value in impacts.values()) else "NOT BENEFICIAL"

    results_dir = ROOT / "results"
    artifact = {
        "schema_version": 1,
        "validation_only": True,
        "test_features_accessed": False,
        "split_seed": SPLIT_SEED,
        "model_seeds": MODEL_SEEDS,
        "epsilon_fractions": EPSILON_FRACTIONS,
        "tau": 0.10,
        "temp_drift_candidate_budget": 1600,
        "impact_threshold_absolute_asr": MEANINGFUL_ASR_DELTA,
        "runs": runs,
        "aggregate": aggregate,
        "impact": {str(key): value for key, value in impacts.items()},
        "conclusion": conclusion,
    }
    (results_dir / "temp_drift_improvement_comparison.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    with (results_dir / "temp_drift_improvement_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)
    with (results_dir / "temp_drift_improvement_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(sample_rows)

    print("\nDetailed metrics")
    print("| epsilon | Attack | ASR | 1-Fidelity | Trace Distance | Exact feasibility |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in aggregate:
        print(f"| {row['epsilon_fraction']:.0%} | {row['attack']} | {row['asr']:.4f} | {row['one_minus_fidelity']:.6f} | {row['trace_distance']:.6f} | {row['exact_feasibility_rate']:.4f} |")
    print("\n| epsilon | Original ASR | Current ASR | New ASR | PGD ASR | Current 1-F | New 1-F | PGD 1-F | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for fraction in EPSILON_FRACTIONS:
        by_attack = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == fraction}
        original = by_attack["temp_drift_reference"]
        current = by_attack["temp_drift_improved"]
        new = by_attack["temp_drift_adaptive"]
        pgd = by_attack["classical_timing"]
        print(f"| {fraction:.0%} | {original['asr']:.4f} | {current['asr']:.4f} | {new['asr']:.4f} | {pgd['asr']:.4f} | {current['one_minus_fidelity']:.6f} | {new['one_minus_fidelity']:.6f} | {pgd['one_minus_fidelity']:.6f} | {impacts[fraction]} |")
    print(f"\nNEW TEMP-DRIFT SEARCH: {conclusion}")


if __name__ == "__main__":
    main()
