"""Final held-out comparison on the five manifest-frozen clean checkpoints."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_split_indices, load_iris_splits

EPSILONS = (0.02, 0.05, 0.10)
ATTACKS = ("classical_timing", "temp_drift_adaptive")
EXPECTED_SEEDS = (42, 123, 777, 2026, 6543)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}


def winner(temp, pgd):
    if temp > pgd + 1e-12:
        return "TEMP-DRIFT"
    if pgd > temp + 1e-12:
        return "PGD"
    return "TIE"


def main():
    manifest_path = ROOT / "results" / "final_clean_protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["selected_checkpoints"]
    if manifest.get("status") != "FROZEN" or manifest.get("max_epochs") != 380:
        raise RuntimeError("Final clean protocol is not the frozen 380-epoch protocol.")
    if tuple(entry["seed"] for entry in entries) != EXPECTED_SEEDS:
        raise RuntimeError("Frozen checkpoint seeds do not match the required seed order.")
    expected_paths = {Path(entry["checkpoint"]).name for entry in entries}
    actual_paths = {path.name for path in (ROOT / "checkpoints").iterdir() if path.is_file()}
    if actual_paths != expected_paths:
        raise RuntimeError("Checkpoint directory contains a missing or non-frozen checkpoint.")
    for entry in entries:
        path = ROOT / entry["checkpoint"]
        if not path.is_file() or sha256(path) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen checkpoint verification failed for seed {entry['seed']}.")

    config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    split_seed = 42
    _, _, held_out, _, _, held_out_labels, _ = load_iris_splits(
        seed=split_seed, test_size=config["test_size"], val_size=config["val_size"]
    )
    _, _, held_out_ids = load_iris_split_indices(
        seed=split_seed, test_size=config["test_size"], val_size=config["val_size"]
    )
    T = float(config["time_window"])
    runs, samples = [], []
    for entry in entries:
        seed = int(entry["seed"])
        local = {**config, "seed": seed, "split_seed": split_seed}
        checkpoint = ROOT / entry["checkpoint"]
        for epsilon in EPSILONS:
            for attack in ATTACKS:
                result = evaluate_attack_data(
                    local, checkpoint, attack, epsilon * T, held_out, held_out_labels,
                    tau=0.10, iterations=20, step_size=epsilon * T / 5,
                    split="held_out", sample_ids=held_out_ids, return_samples=True,
                )
                row = {**result["summary"], "model_seed": seed,
                       "checkpoint_sha256": entry["checkpoint_sha256"]}
                runs.append(row)
                samples.extend({"model_seed": seed, "epsilon_fraction": epsilon,
                                "attack": attack, **sample} for sample in result["samples"])
                print(f"seed={seed} epsilon={epsilon:.0%} attack={attack} n={row['clean_correct_count']} ASR={row['asr']:.4f}")

    paired = []
    for seed in EXPECTED_SEEDS:
        for epsilon in EPSILONS:
            cell = [sample for sample in samples if sample["model_seed"] == seed
                    and sample["epsilon_fraction"] == epsilon]
            by_attack = {attack: {sample["sample_id"]: sample for sample in cell
                                  if sample["attack"] == attack} for attack in ATTACKS}
            common_ids = sorted(set(by_attack[ATTACKS[0]]) & set(by_attack[ATTACKS[1]]))
            common_clean = [sample_id for sample_id in common_ids
                            if by_attack[ATTACKS[0]][sample_id]["clean_correct"]
                            and by_attack[ATTACKS[1]][sample_id]["clean_correct"]]
            outcomes = {"pgd_only_successes": 0, "temp_drift_only_successes": 0,
                        "both_successful": 0, "both_robust": 0}
            for sample_id in common_clean:
                pgd = by_attack["classical_timing"][sample_id]["attack_success"]
                temp = by_attack["temp_drift_adaptive"][sample_id]["attack_success"]
                key = ("both_successful" if pgd and temp else "pgd_only_successes" if pgd
                       else "temp_drift_only_successes" if temp else "both_robust")
                outcomes[key] += 1
            paired.append({"model_seed": seed, "epsilon_fraction": epsilon,
                           "common_clean_correct": len(common_clean), **outcomes})

    aggregate = []
    paired_aggregate = []
    for epsilon in EPSILONS:
        by_attack = {}
        for attack in ATTACKS:
            selected = [row for row in runs if row["epsilon_fraction"] == epsilon
                        and row["attack"] == attack]
            by_attack[attack] = {
                "asr": mean_sd([row["asr"] for row in selected]),
                "one_minus_fidelity": mean_sd([row["one_minus_fidelity"] for row in selected]),
                "trace_distance": mean_sd([row["trace_distance"] for row in selected]),
                "exact_feasibility": mean_sd([row["exact_feasibility_rate"] for row in selected]),
            }
        pgd_rows = {row["model_seed"]: row for row in runs
                    if row["epsilon_fraction"] == epsilon and row["attack"] == "classical_timing"}
        temp_rows = {row["model_seed"]: row for row in runs
                     if row["epsilon_fraction"] == epsilon and row["attack"] == "temp_drift_adaptive"}
        aggregate.append({
            "epsilon_fraction": epsilon, "attacks": by_attack,
            "temp_drift_asr_seed_wins": sum(temp_rows[s]["asr"] > pgd_rows[s]["asr"] + 1e-12 for s in EXPECTED_SEEDS),
            "temp_drift_drift_seed_wins": sum(temp_rows[s]["one_minus_fidelity"] > pgd_rows[s]["one_minus_fidelity"] + 1e-12 for s in EXPECTED_SEEDS),
            "asr_winner": winner(by_attack["temp_drift_adaptive"]["asr"]["mean"], by_attack["classical_timing"]["asr"]["mean"]),
            "drift_winner": winner(by_attack["temp_drift_adaptive"]["one_minus_fidelity"]["mean"], by_attack["classical_timing"]["one_minus_fidelity"]["mean"]),
        })
        selected_pairs = [row for row in paired if row["epsilon_fraction"] == epsilon]
        paired_aggregate.append({"epsilon_fraction": epsilon,
                                 **{key: sum(row[key] for row in selected_pairs)
                                    for key in ("common_clean_correct", "pgd_only_successes",
                                                "temp_drift_only_successes", "both_successful", "both_robust")}})

    temp_asr_wins = sum(row["asr_winner"] == "TEMP-DRIFT" for row in aggregate)
    pgd_asr_wins = sum(row["asr_winner"] == "PGD" for row in aggregate)
    temp_drift_wins = sum(row["drift_winner"] == "TEMP-DRIFT" for row in aggregate)
    pgd_drift_wins = sum(row["drift_winner"] == "PGD" for row in aggregate)
    if temp_asr_wins >= 2 and temp_drift_wins >= 2:
        conclusion = "TEMP-DRIFT STRONGER"
    elif pgd_asr_wins >= 2 and pgd_drift_wins >= 2:
        conclusion = "PGD STRONGER"
    else:
        conclusion = "TEMP-DRIFT COMPETITIVE"

    artifact = {
        "schema_version": 1, "evaluation_split": "held_out",
        "held_out_evaluations": 1, "training_run": False, "defense_run": False,
        "manifest": str(manifest_path.relative_to(ROOT)), "checkpoint_verification": "all_five_sha256_match",
        "attacks": {"classical_timing": {"iterations": 20, "step_size": "epsilon/5", "random_start": False},
                    "temp_drift_adaptive": {"candidate_budget": 1600, "tau": 0.10, "gradients": False}},
        "runs": runs, "aggregate": aggregate, "paired_by_seed": paired,
        "paired_aggregate": paired_aggregate, "conclusion": conclusion,
    }
    results = ROOT / "results"
    (results / "final_attack_comparison.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with (results / "final_attack_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model_seed", "epsilon_fraction", "attack", "clean_correct_count", "asr",
                  "successful_attacks", "one_minus_fidelity", "trace_distance", "exact_feasibility_rate",
                  "checkpoint", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in runs)
    with (results / "final_attack_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys()); writer.writeheader(); writer.writerows(samples)

    lines = ["# Final Frozen-Checkpoint Attack Comparison", "",
             "Held-out evaluation using only the five SHA-256-verified checkpoints in `results/final_clean_protocol.json`.", "",
             "## Per-Seed Results", "",
             "| Seed | epsilon | Attack | Clean-correct | Successes | ASR | 1-Fidelity | Trace Distance | Exact feasibility |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        lines.append(f"| {row['model_seed']} | {row['epsilon_fraction']:.0%} | {row['attack']} | {row['clean_correct_count']} | {row['successful_attacks']} | {row['asr']:.4f} | {row['one_minus_fidelity']:.6f} | {row['trace_distance']:.6f} | {row['exact_feasibility_rate']:.4f} |")
    lines += ["", "## Paired Common-Clean-Correct Outcomes", "",
              "| epsilon | Common clean-correct | PGD only | TEMP-DRIFT only | Both successful | Both robust |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in paired_aggregate:
        lines.append(f"| {row['epsilon_fraction']:.0%} | {row['common_clean_correct']} | {row['pgd_only_successes']} | {row['temp_drift_only_successes']} | {row['both_successful']} | {row['both_robust']} |")
    lines += ["", "## Compact Summary", "",
              "| epsilon | PGD ASR mean±SD | TEMP-DRIFT ASR mean±SD | PGD 1-F mean±SD | TEMP-DRIFT 1-F mean±SD | ASR Winner | Drift Winner |",
              "|---:|---:|---:|---:|---:|---|---|"]
    for row in aggregate:
        pgd, temp = row["attacks"]["classical_timing"], row["attacks"]["temp_drift_adaptive"]
        lines.append(f"| {row['epsilon_fraction']:.0%} | {pgd['asr']['mean']:.4f}±{pgd['asr']['sample_sd']:.4f} | {temp['asr']['mean']:.4f}±{temp['asr']['sample_sd']:.4f} | {pgd['one_minus_fidelity']['mean']:.6f}±{pgd['one_minus_fidelity']['sample_sd']:.6f} | {temp['one_minus_fidelity']['mean']:.6f}±{temp['one_minus_fidelity']['sample_sd']:.6f} | {row['asr_winner']} | {row['drift_winner']} |")
    lines += ["", f"**{conclusion}**", ""]
    (results / "final_attack_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-8:]))


if __name__ == "__main__":
    main()
