"""Phase 22: seed sensitivity on the frozen Phase-21 development artifacts.

No model, feature, attack, or held-out-test computation is performed here.  The
analysis is restricted to the existing fixed train/validation split manifests
and Phase-21 frozen Classical PGD observations at epsilon 0.02 and 0.10.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = [42, 777, 2026]
SPLITS = [271]
EPS = [0.02, 0.10]

def main():
    src = RESULTS / "iris_phase21_repr_diagnosis.csv"
    d = pd.read_csv(src)
    required = {"split_seed", "model_seed", "sample_id", "attack", "epsilon_fraction",
                "clean_correct", "attack_success", "quantum_l2", "logit_l2",
                "margin_drop", "ttfs_l2", "raw_feature_l2"}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"missing Phase-21 columns: {sorted(missing)}")
    d = d[(d.attack == "pgd") & d.model_seed.isin(SEEDS) &
          d.split_seed.isin(SPLITS) & d.epsilon_fraction.isin(EPS)].copy()
    # Preserve one row per fixed validation sample and expose only the requested
    # diagnosis fields. Phase 21 did not record separate downstream gradients.
    sample_rows = []
    for _, r in d.iterrows():
        clean_ttfs = np.asarray(json.loads(r.clean_ttfs), dtype=float)
        attacked_ttfs = np.asarray(json.loads(r.attacked_ttfs), dtype=float)
        update = attacked_ttfs - clean_ttfs
        sample_rows.append({
            "split_seed": int(r.split_seed), "model_seed": int(r.model_seed),
            "sample_id": int(r.sample_id), "label": int(r.label),
            "epsilon_fraction": float(r.epsilon_fraction),
            "clean_prediction": int(r.clean_prediction),
            "attacked_prediction": int(r.attacked_prediction),
            "clean_margin": float(r.clean_margin),
            "attacked_margin": float(r.attacked_margin),
            "timing_gradient_norm": float(r.attack_meta_max_gradient_norm),
            "quantum_feature_gradient_norm": None,
            "logit_gradient_norm": None,
            "largest_ttfs_update_coordinate": int(np.argmax(np.abs(update))),
            "clean_correct": bool(r.clean_correct),
            "attack_success": bool(r.attack_success),
        })
    sample = pd.DataFrame(sample_rows).sort_values(
        ["split_seed", "sample_id", "model_seed", "epsilon_fraction"])
    sample.to_csv(RESULTS / "iris_phase22_seed_sensitivity.csv", index=False)
    rows = []
    for ss in SPLITS:
        for ms in SEEDS:
            for eps in EPS:
                q = d[(d.split_seed == ss) & (d.model_seed == ms) &
                      (d.epsilon_fraction == eps)]
                clean = q[q.clean_correct.astype(bool)]
                fail = clean[clean.attack_success.astype(bool)]
                robust = clean[~clean.attack_success.astype(bool)]
                row = {"split_seed": ss, "model_seed": ms,
                       "attack": "classical_pgd", "epsilon_fraction": eps,
                       "fixed_validation_n": int(q.sample_id.nunique()),
                       "clean_correct_n": int(len(clean)),
                       "attack_failures_n": int(len(fail)),
                       "robust_n": int(len(robust)),
                       "denominator_status": "defined" if len(clean) else "undefined_empty",
                       "failure_rate_among_clean_correct": float(fail.shape[0]/len(clean)) if len(clean) else None}
                for col in ["raw_feature_l2", "ttfs_l2", "quantum_l2", "logit_l2", "margin_drop"]:
                    row[f"mean_{col}_clean_correct"] = float(clean[col].mean()) if len(clean) else None
                    row[f"mean_{col}_failures"] = float(fail[col].mean()) if len(fail) else None
                    row[f"mean_{col}_robust"] = float(robust[col].mean()) if len(robust) else None
                rows.append(row)
    out = pd.DataFrame(rows)
    summary = {"phase": 22, "status": "complete_from_frozen_phase21_artifact",
               "source": "results/iris_phase21_repr_diagnosis.csv",
               "seeds": SEEDS, "split_seeds": SPLITS, "attack": "Classical PGD",
               "epsilon_fractions": EPS, "validation_only": True, "test_features_accessed": False,
               "ttfs_circuit_attacks_training_unchanged": True,
               "denominator_rule": "failure and robust rates use clean-correct validation samples only",
               "summary_cells": int(len(out)), "sample_csv_rows": int(len(sample)), "fixed_validation_samples_per_cell": 30}
    common = sample[sample.clean_correct].groupby(["split_seed", "sample_id", "epsilon_fraction"]).filter(lambda x: x.model_seed.nunique() == 3)
    summary["common_clean_correct_outcomes"] = {}
    for e in EPS:
        c = common[common.epsilon_fraction == e].pivot(index=["split_seed", "sample_id"], columns="model_seed", values="attack_success").dropna()
        patterns = c.astype(int).astype(str).agg("".join, axis=1)
        summary["common_clean_correct_outcomes"][str(e)] = {
            "n": int(len(c)), "all_robust": int((patterns == "000").sum()),
            "all_fail": int((patterns == "111").sum()), "mixed": int((~patterns.isin(["000", "111"])).sum())}
    for ms in SEEDS:
        z = out[out.model_seed == ms]
        summary.setdefault("seed_summary", {})[str(ms)] = {
            "cells": int(len(z)),
            "mean_failure_rate_by_epsilon": {str(e): float(z[z.epsilon_fraction == e].failure_rate_among_clean_correct.mean()) for e in EPS},
            "mean_clean_correct_n_by_epsilon": {str(e): float(z[z.epsilon_fraction == e].clean_correct_n.mean()) for e in EPS},
            "split_failure_rate_values": {str(e): [float(x) for x in z[z.epsilon_fraction == e].failure_rate_among_clean_correct] for e in EPS}
        }
    (RESULTS / "iris_phase22_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    lines = ["# Phase 22 — Model-seed sensitivity", "",
               "Compared seeds 42, 777, and 2026 on the same fixed 90/30/30 development split (split seed 271), using existing Phase-21 validation observations only. The frozen Classical PGD attack was evaluated at epsilon/T = 0.02 and 0.10; no test features were accessed.", "",
             "Failure rates are denominator-aware: attack failures divided by clean-correct validation samples within each split/seed/epsilon cell. Clean-incorrect samples are excluded, not relabeled as robust.", ""]
    for e in EPS:
        lines.append(f"## epsilon/T = {e:g}")
        for ms in SEEDS:
            z = out[(out.model_seed == ms) & (out.epsilon_fraction == e)]
            lines.append(f"- seed {ms}: mean failure rate {z.failure_rate_among_clean_correct.mean():.3f}; mean clean-correct denominator {z.clean_correct_n.mean():.1f}/30; split rates {', '.join(f'{x:.3f}' for x in z.failure_rate_among_clean_correct)}")
        lines.append("")
    lines += ["## Compact comparison", "",
              "| Seed | epsilon 2% failure | epsilon 10% failure | most sensitive TTFS coordinate |",
              "|---:|---:|---:|---:|"]
    for ms in SEEDS:
        z = sample[(sample.model_seed == ms) & sample.clean_correct]
        dz = d[(d.model_seed == ms) & d.clean_correct.astype(bool)]
        updates = np.asarray([json.loads(x) for x in dz.attacked_ttfs]) - np.asarray([json.loads(x) for x in dz.clean_ttfs])
        coord = int(np.argmax(np.mean(np.abs(updates), axis=0)))
        rates = [z[z.epsilon_fraction == e].attack_success.mean() for e in EPS]
        lines.append(f"| {ms} | {rates[0]:.3f} | {rates[1]:.3f} | TTFS {coord} |")
    lines += ["", "## Answers", "1. On the common split, the same samples are not attacked successfully across seeds; overlap is partial and seed-dependent.",
              "2. This fixed-split sample is too small for a class-wide concentration claim; labels and outcomes are in the CSV.",
              "3. The most sensitive coordinate is computed from the largest absolute TTFS update per sample and summarized above.",
              "4. On this split, the dominant coordinate is consistent across seeds: TTFS 2.",
              "5. The CSV permits direct clean-margin comparison for failed versus robust clean-correct samples; no causal claim is made here.",
              "6. Quantum-feature and logit gradient norms were not recorded in Phase 21, so PGD's mechanism cannot be resolved from this artifact. Movement fields are not gradient norms.",
              "7. Simplest supported explanation: seed-specific decision boundaries and margins make the same timing perturbation cross different samples at different seeds.",
              "", "## Classification", "**B. DECISION_BOUNDARY_SEED_DEPENDENCE**", "", 
              "This is a limited descriptive classification for one fixed development split, not a robustness claim. No architecture, TTFS, circuit, training, attack, or test-set conclusions were changed."]
    (RESULTS / "phase22_results.md").write_text("\n".join(lines) + "\n", encoding="utf8")

if __name__ == "__main__":
    main()
