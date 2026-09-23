"""Validation-only comparison of adaptive, two-stage TEMP-DRIFT, and frozen PGD."""
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
ATTACKS = ("temp_drift_adaptive", "temp_drift_two_stage", "classical_timing")
MEANINGFUL_ASR_DELTA = 0.05


def main():
    base_config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    _, validation, _, labels, _, _, sample_ids = load_iris_train_validation(
        seed=SPLIT_SEED,
        test_size=float(base_config["test_size"]),
        val_size=float(base_config["val_size"]),
    )
    T = float(base_config["time_window"])
    runs = []
    for model_seed in MODEL_SEEDS:
        config = {**base_config, "seed": model_seed, "split_seed": SPLIT_SEED}
        checkpoint = ROOT / "checkpoints" / f"iris_phase21_baseline_split_{SPLIT_SEED}_model_{model_seed}.pt"
        for fraction in EPSILON_FRACTIONS:
            for attack in ATTACKS:
                row = evaluate_attack_data(
                    config, checkpoint, attack, fraction * T, validation, labels,
                    tau=0.10, iterations=20, step_size=fraction * T / 5,
                    split="validation", sample_ids=sample_ids,
                )
                runs.append({**row, "model_seed": model_seed, "split_seed": SPLIT_SEED})
                print(
                    f"seed={model_seed} epsilon={fraction:.0%} attack={attack} "
                    f"ASR={row['asr']:.4f} 1-F={row['one_minus_fidelity']:.6f}"
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
                "stage1_successful_attacks": sum(row["stage1_successful_attacks"] for row in selected),
                "stage1_successes_preserved": sum(row["stage1_successes_preserved"] for row in selected),
            })

    impacts = {}
    for fraction in EPSILON_FRACTIONS:
        rows = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == fraction}
        current = rows["temp_drift_adaptive"]
        two_stage = rows["temp_drift_two_stage"]
        asr_delta = two_stage["asr"] - current["asr"]
        drift_increased = two_stage["one_minus_fidelity"] > current["one_minus_fidelity"]
        if asr_delta <= -MEANINGFUL_ASR_DELTA:
            impacts[fraction] = "WORSE"
        elif asr_delta >= 0.0 and drift_increased:
            impacts[fraction] = "IMPROVED"
        else:
            impacts[fraction] = "NO MEANINGFUL IMPROVEMENT"
    conclusion = "BENEFICIAL" if any(value == "IMPROVED" for value in impacts.values()) and all(value != "WORSE" for value in impacts.values()) else "NOT BENEFICIAL"

    results = ROOT / "results"
    artifact = {
        "schema_version": 1,
        "validation_only": True,
        "test_features_accessed": False,
        "split_seed": SPLIT_SEED,
        "model_seeds": MODEL_SEEDS,
        "epsilon_fractions": EPSILON_FRACTIONS,
        "candidate_budget": 1600,
        "stage1_budget": 1200,
        "stage2_reserved_budget": 400,
        "impact_rule": "IMPROVED requires nondecreasing ASR and increased 1-F; WORSE requires ASR loss >= 0.05",
        "runs": runs,
        "aggregate": aggregate,
        "impact": {str(key): value for key, value in impacts.items()},
        "conclusion": conclusion,
    }
    (results / "two_stage_temp_drift_comparison.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with (results / "two_stage_temp_drift_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)

    print("\nDetailed metrics")
    print("| epsilon | Attack | ASR | 1-Fidelity | Trace Distance | Exact feasibility | Stage-1 successes preserved |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for row in aggregate:
        preserved = f"{row['stage1_successes_preserved']}/{row['stage1_successful_attacks']}" if row["attack"] == "temp_drift_two_stage" else "n/a"
        print(f"| {row['epsilon_fraction']:.0%} | {row['attack']} | {row['asr']:.4f} | {row['one_minus_fidelity']:.6f} | {row['trace_distance']:.6f} | {row['exact_feasibility_rate']:.4f} | {preserved} |")
    print("\n| epsilon | Current ASR | Two-Stage ASR | PGD ASR | Current 1-F | Two-Stage 1-F | PGD 1-F | Impact |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for fraction in EPSILON_FRACTIONS:
        rows = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == fraction}
        current, two_stage, pgd = rows["temp_drift_adaptive"], rows["temp_drift_two_stage"], rows["classical_timing"]
        print(f"| {fraction:.0%} | {current['asr']:.4f} | {two_stage['asr']:.4f} | {pgd['asr']:.4f} | {current['one_minus_fidelity']:.6f} | {two_stage['one_minus_fidelity']:.6f} | {pgd['one_minus_fidelity']:.6f} | {impacts[fraction]} |")
    print(f"\nTWO-STAGE TEMP-DRIFT: {conclusion}")


if __name__ == "__main__":
    main()
