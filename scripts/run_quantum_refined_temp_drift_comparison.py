"""Frozen validation-only comparison for quantum-refined TEMP-DRIFT."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_train_validation


SPLIT_SEED = 42
EPSILON_FRACTIONS = (0.02, 0.05, 0.10)
ATTACKS = ("temp_drift_adaptive", "temp_drift_quantum_refined", "classical_timing")


def main():
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "results" / "final_clean_protocol.json").read_text(encoding="utf-8"))
    _, validation, _, labels, _, _, sample_ids = load_iris_train_validation(
        seed=SPLIT_SEED, test_size=float(config["test_size"]), val_size=float(config["val_size"]),
    )
    T = float(config["time_window"])
    runs, samples = [], []
    for checkpoint_spec in protocol["selected_checkpoints"]:
        model_seed = int(checkpoint_spec["seed"])
        run_config = {**config, "seed": model_seed, "split_seed": SPLIT_SEED}
        checkpoint = ROOT / checkpoint_spec["checkpoint"]
        for fraction in EPSILON_FRACTIONS:
            for attack in ATTACKS:
                result = evaluate_attack_data(
                    run_config, checkpoint, attack, fraction * T, validation, labels,
                    tau=0.10, iterations=20, step_size=fraction * T / 5,
                    split="validation", sample_ids=sample_ids, return_samples=True,
                )
                runs.append({**result["summary"], "model_seed": model_seed})
                samples.extend({**row, "model_seed": model_seed,
                                "epsilon_fraction": fraction, "attack": attack}
                               for row in result["samples"])
                print(f"seed={model_seed} epsilon={fraction:.0%} attack={attack} "
                      f"ASR={result['summary']['asr']:.4f}")

    aggregate, paired = [], []
    for fraction in EPSILON_FRACTIONS:
        for attack in ATTACKS:
            selected = [row for row in runs if row["epsilon_fraction"] == fraction
                        and row["attack"] == attack]
            aggregate.append({
                "epsilon_fraction": fraction, "attack": attack,
                "asr_numerator": sum(row["asr_numerator"] for row in selected),
                "asr_denominator": sum(row["asr_denominator"] for row in selected),
                "one_minus_fidelity": float(np.mean([row["one_minus_fidelity"] for row in selected])),
                "trace_distance": float(np.mean([row["trace_distance"] for row in selected])),
                "exact_feasibility_rate": float(np.mean([row["exact_feasibility_rate"] for row in selected])),
                "stage1_successful_attacks": sum(row["stage1_successful_attacks"] for row in selected),
                "stage1_successes_preserved": sum(row["stage1_successes_preserved"] for row in selected),
            })
        keyed = {(row["model_seed"], row["sample_id"], row["attack"]): row for row in samples
                 if row["epsilon_fraction"] == fraction and row["clean_correct"]}
        current, refined = [], []
        counts = {"rescued": 0, "broken": 0, "both_success": 0, "both_fail": 0}
        keys = {(seed, sample_id) for seed, sample_id, attack in keyed if attack == ATTACKS[0]}
        for seed, sample_id in keys:
            left = keyed[(seed, sample_id, ATTACKS[0])]
            right = keyed[(seed, sample_id, ATTACKS[1])]
            if left["attack_success"] and right["attack_success"]:
                counts["both_success"] += 1; current.append(left); refined.append(right)
            elif right["attack_success"]:
                counts["rescued"] += 1
            elif left["attack_success"]:
                counts["broken"] += 1
            else:
                counts["both_fail"] += 1
        paired.append({"epsilon_fraction": fraction, **counts,
                       "current_one_minus_fidelity": float(np.mean([x["input_one_minus_fidelity"] for x in current])) if current else None,
                       "refined_one_minus_fidelity": float(np.mean([x["input_one_minus_fidelity"] for x in refined])) if refined else None,
                       "current_trace_distance": float(np.mean([x["input_trace_distance"] for x in current])) if current else None,
                       "refined_trace_distance": float(np.mean([x["input_trace_distance"] for x in refined])) if refined else None})

    for row in aggregate:
        row["asr"] = row["asr_numerator"] / row["asr_denominator"] if row["asr_denominator"] else 0.0
    conditions = []
    for pair in paired:
        rows = {row["attack"]: row for row in aggregate
                if row["epsilon_fraction"] == pair["epsilon_fraction"]}
        refined = rows[ATTACKS[1]]
        current = rows[ATTACKS[0]]
        conditions.append(pair["broken"] == 0 and refined["exact_feasibility_rate"] == 1.0
                          and refined["stage1_successful_attacks"] == refined["stage1_successes_preserved"]
                          and pair["both_success"] > 0
                          and refined["one_minus_fidelity"] >= current["one_minus_fidelity"]
                          and refined["trace_distance"] >= current["trace_distance"]
                          and pair["refined_one_minus_fidelity"] >= pair["current_one_minus_fidelity"]
                          and pair["refined_trace_distance"] >= pair["current_trace_distance"])
    strict_improvement = any(pair["both_success"] > 0 and
                             pair["refined_one_minus_fidelity"] > pair["current_one_minus_fidelity"]
                             for pair in paired)
    conclusion = "BENEFICIAL" if all(conditions) and strict_improvement else "NOT BENEFICIAL"
    artifact = {"schema_version": 1, "status": "FROZEN" if conclusion == "BENEFICIAL" else "NOT_FROZEN",
                "validation_only": True, "test_features_accessed": False,
                "split_seed": SPLIT_SEED, "model_seeds": [x["seed"] for x in protocol["selected_checkpoints"]],
                "candidate_budget": 1600, "stage1_budget": 1200, "stage2_budget": 400,
                "attacks": ATTACKS, "epsilon_fractions": EPSILON_FRACTIONS,
                "acceptance_rule": "no paired broken successes; exact feasibility and Stage-1 preservation; nondecreasing aggregate and paired both-success drift at every epsilon; strict paired 1-F improvement at least once",
                "runs": runs, "aggregate": aggregate, "paired": paired, "conclusion": conclusion}
    output = ROOT / "results" / "quantum_refined_temp_drift_validation.json"
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nQUANTUM-DRIFT IMPROVEMENT: {conclusion}")


if __name__ == "__main__":
    main()
