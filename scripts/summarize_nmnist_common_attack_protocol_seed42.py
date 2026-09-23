"""Create paired summaries from the completed Seed-42 protocol CSV."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/nmnist_common_attack_protocol_seed42"
rows = list(csv.DictReader((OUT / "per_sample_results.csv").open(newline="", encoding="utf-8")))
summary = []
for model in ("SNN", "QSNN"):
    for attack in ("PGD", "TEMP-DRIFT"):
        for eps in (0.02, 0.05, 0.10):
            selected = [r for r in rows if r["model"] == model and r["attack"] == attack and float(r["epsilon_fraction"]) == eps]
            success = np.asarray([r["attack_success"] == "True" for r in selected])
            summary.append({"model": model, "attack": attack, "epsilon_fraction": eps,
                            "n": len(selected), "successes": int(success.sum()),
                            "asr": float(success.mean()),
                            "attacked_accuracy": float(1-success.mean()),
                            "mean_abs_timestamp_drift": float(np.mean([float(r["mean_abs_timestamp_drift"]) for r in selected])),
                            "feasibility_rate": float(np.mean([r["feasible"] == "True" for r in selected]))})
paired = []
for model in ("SNN", "QSNN"):
    for eps in (0.02, 0.05, 0.10):
        p = {int(r["sample_id"]): r["attack_success"] == "True" for r in rows if r["model"] == model and r["attack"] == "PGD" and float(r["epsilon_fraction"]) == eps}
        t = {int(r["sample_id"]): r["attack_success"] == "True" for r in rows if r["model"] == model and r["attack"] == "TEMP-DRIFT" and float(r["epsilon_fraction"]) == eps}
        paired.append({"model": model, "epsilon_fraction": eps, "paired_n": len(p),
                       "pgd_only": sum(p[i] and not t[i] for i in p),
                       "temp_drift_only": sum(not p[i] and t[i] for i in p),
                       "both_fail": sum(p[i] and t[i] for i in p),
                       "both_robust": sum(not p[i] and not t[i] for i in p),
                       "rescued": sum(p[i] and not t[i] for i in p),
                       "broken": sum(not p[i] and t[i] for i in p)})
artifact = {"summary": summary, "paired_attack_outcomes": paired,
            "scientific_note": "These are single-seed protocol-validation results on a common clean-correct denominator; they are not a five-seed robustness claim."}
(OUT / "summary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
print(json.dumps(artifact, indent=2))
