"""One-shot validation-only Gradient Adaptive TEMP-DRIFT comparison."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_train_validation


SPLIT_SEED = 42
EPSILON_FRACTIONS = (0.02, 0.05, 0.10)
ATTACKS = ("classical_timing", "temp_drift_adaptive", "temp_drift_gradient_adaptive")
DISPLAY = {"classical_timing": "PGD", "temp_drift_adaptive": "Adaptive TEMP",
           "temp_drift_gradient_adaptive": "Gradient Adaptive TEMP"}


def main():
    comparison_started = time.perf_counter()
    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "results" / "final_clean_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN":
        raise RuntimeError("Clean checkpoint protocol is not frozen.")
    _, validation, _, labels, _, _, sample_ids = load_iris_train_validation(
        seed=SPLIT_SEED, test_size=float(config["test_size"]), val_size=float(config["val_size"]),
    )
    T = float(config["time_window"])
    runs, samples = [], []
    total_qnode_evaluations = 0
    gradient_runtime = 0.0
    gradient_samples = 0
    for checkpoint_spec in protocol["selected_checkpoints"]:
        checkpoint = ROOT / checkpoint_spec["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != checkpoint_spec["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        model_seed = int(checkpoint_spec["seed"])
        print(f"[SEED START] seed={model_seed}", flush=True)
        run_config = {**config, "seed": model_seed, "split_seed": SPLIT_SEED}
        for fraction in EPSILON_FRACTIONS:
            print(f"[EPS START] seed={model_seed} eps={fraction:.0%}", flush=True)
            epsilon_results = {}
            for attack in ATTACKS:
                result = evaluate_attack_data(
                    run_config, checkpoint, attack, fraction * T, validation, labels,
                    tau=0.10, iterations=20, step_size=fraction * T / 5,
                    split="validation", sample_ids=sample_ids, return_samples=True,
                )
                runs.append({**result["summary"], "model_seed": model_seed})
                epsilon_results[attack] = result["summary"]
                total_qnode_evaluations += result["summary"]["qnode_evaluations"]
                if attack == "temp_drift_gradient_adaptive":
                    gradient_runtime += result["summary"]["runtime_seconds"]
                    gradient_samples += len(validation)
                samples.extend({**row, "model_seed": model_seed,
                                "epsilon_fraction": fraction, "attack": attack}
                               for row in result["samples"])
                print(f"seed={model_seed} epsilon={fraction:.0%} {DISPLAY[attack]} "
                      f"ASR={result['summary']['asr']:.4f} "
                      f"1-F={result['summary']['one_minus_fidelity']:.6f}", flush=True)
            print(
                f"[EPS COMPLETE] seed={model_seed} eps={fraction:.0%} "
                f"PGD_ASR={epsilon_results[ATTACKS[0]]['asr']:.4f} "
                f"Adaptive_ASR={epsilon_results[ATTACKS[1]]['asr']:.4f} "
                f"Gradient_ASR={epsilon_results[ATTACKS[2]]['asr']:.4f}",
                flush=True,
            )
        print(f"[SEED COMPLETE] seed={model_seed}", flush=True)

    aggregate = []
    for fraction in EPSILON_FRACTIONS:
        fraction_samples = [row for row in samples
                            if row["epsilon_fraction"] == fraction and row["clean_correct"]]
        keyed = {(row["model_seed"], row["sample_id"], row["attack"]): row
                 for row in fraction_samples}
        units = sorted({(row["model_seed"], row["sample_id"]) for row in fraction_samples})
        for attack in ATTACKS:
            selected_runs = [row for row in runs if row["epsilon_fraction"] == fraction
                             and row["attack"] == attack]
            selected_samples = [keyed[(seed, sample_id, attack)] for seed, sample_id in units]
            successes = [row for row in selected_samples if row["attack_success"]]
            unique = sum(
                bool(keyed[(seed, sample_id, attack)]["attack_success"])
                and not any(keyed[(seed, sample_id, other)]["attack_success"]
                            for other in ATTACKS if other != attack)
                for seed, sample_id in units
            )
            numerator = len(successes)
            aggregate.append({
                "epsilon_fraction": fraction, "attack": attack,
                "asr": numerator / len(units) if units else 0.0,
                "successful_attacks": numerator, "clean_correct_count": len(units),
                "unique_successful_attacks": int(unique),
                "one_minus_fidelity": float(np.mean([row["input_one_minus_fidelity"] for row in selected_samples])),
                "trace_distance": float(np.mean([row["input_trace_distance"] for row in selected_samples])),
                "successful_one_minus_fidelity": float(np.mean([row["input_one_minus_fidelity"] for row in successes])) if successes else None,
                "successful_trace_distance": float(np.mean([row["input_trace_distance"] for row in successes])) if successes else None,
                "exact_feasibility_rate": float(np.mean([row["exact_feasibility_rate"] for row in selected_runs])),
            })

    checkpoint_noninferior = True
    for fraction in EPSILON_FRACTIONS:
        for checkpoint_spec in protocol["selected_checkpoints"]:
            selected = {row["attack"]: row for row in runs
                        if row["epsilon_fraction"] == fraction
                        and row["model_seed"] == checkpoint_spec["seed"]}
            checkpoint_noninferior &= (selected[ATTACKS[2]]["asr_numerator"] >=
                                       selected[ATTACKS[1]]["asr_numerator"])
    criteria = []
    strict_drift_gain = False
    winners = {}
    for fraction in EPSILON_FRACTIONS:
        rows = {row["attack"]: row for row in aggregate
                if row["epsilon_fraction"] == fraction}
        adaptive, gradient = rows[ATTACKS[1]], rows[ATTACKS[2]]
        criteria.append(gradient["successful_attacks"] >= adaptive["successful_attacks"]
                        and gradient["one_minus_fidelity"] >= adaptive["one_minus_fidelity"]
                        and gradient["trace_distance"] >= adaptive["trace_distance"]
                        and gradient["exact_feasibility_rate"] == 1.0)
        strict_drift_gain |= (gradient["one_minus_fidelity"] > adaptive["one_minus_fidelity"]
                              and gradient["trace_distance"] > adaptive["trace_distance"])
        best_asr = max(row["asr"] for row in rows.values())
        finalists = [row for row in rows.values() if row["asr"] == best_asr]
        winners[fraction] = DISPLAY[max(finalists, key=lambda row: row["one_minus_fidelity"])["attack"]]
    conclusion = "BENEFICIAL" if all(criteria) and checkpoint_noninferior and strict_drift_gain else "NOT BENEFICIAL"
    artifact = {
        "schema_version": 1, "status": "FROZEN" if conclusion == "BENEFICIAL" else "NOT_FROZEN",
        "validation_only": True, "test_features_accessed": False,
        "split_seed": SPLIT_SEED, "model_seeds": [item["seed"] for item in protocol["selected_checkpoints"]],
        "epsilon_fractions": EPSILON_FRACTIONS, "tau": 0.10,
        "gradient_adaptive_budget": {"total_candidates": 1600, "initial_candidates": 600,
                                     "gradient_updates": 1000, "retained_elites": 12},
        "acceptance_rule": "no pooled or checkpoint-level ASR loss; nondecreasing overall 1-F and trace at every epsilon; strict joint drift gain at least once; exact feasibility 1.0",
        "runs": runs, "aggregate": aggregate,
        "checkpoint_asr_noninferior": bool(checkpoint_noninferior),
        "winner": {str(key): value for key, value in winners.items()}, "conclusion": conclusion,
    }
    output = ROOT / "results" / "gradient_adaptive_temp_drift_validation.json"
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n| epsilon | Attack | ASR | successful | 1-Fidelity | Trace Distance | exact feasibility | unique successes |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregate:
        print(f"| {row['epsilon_fraction']:.0%} | {DISPLAY[row['attack']]} | {row['asr']:.4f} | "
              f"{row['successful_attacks']} | {row['one_minus_fidelity']:.6f} | "
              f"{row['trace_distance']:.6f} | {row['exact_feasibility_rate']:.4f} | "
              f"{row['unique_successful_attacks']} |")
    print("\n| epsilon | PGD ASR | Adaptive TEMP ASR | Gradient TEMP ASR | PGD 1-F | Adaptive TEMP 1-F | Gradient TEMP 1-F | Best ASR | Best Drift |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for fraction in EPSILON_FRACTIONS:
        rows = {row["attack"]: row for row in aggregate if row["epsilon_fraction"] == fraction}
        best_asr = DISPLAY[max(rows.values(), key=lambda row: row["asr"])["attack"]]
        best_drift = DISPLAY[max(rows.values(), key=lambda row: row["one_minus_fidelity"])["attack"]]
        print(f"| {fraction:.0%} | {rows[ATTACKS[0]]['asr']:.4f} | {rows[ATTACKS[1]]['asr']:.4f} | "
               f"{rows[ATTACKS[2]]['asr']:.4f} | {rows[ATTACKS[0]]['one_minus_fidelity']:.6f} | "
               f"{rows[ATTACKS[1]]['one_minus_fidelity']:.6f} | {rows[ATTACKS[2]]['one_minus_fidelity']:.6f} | "
               f"{best_asr} | {best_drift} |")
    total_elapsed = time.perf_counter() - comparison_started
    print(f"\ntotal_qnode_evaluations={total_qnode_evaluations}")
    print(f"average_gradient_runtime_per_sample={gradient_runtime / gradient_samples:.3f}s")
    print(f"total_elapsed_runtime={total_elapsed:.3f}s")
    print(f"\nGRADIENT ADAPTIVE TEMP-DRIFT: {conclusion}")


if __name__ == "__main__":
    main()
