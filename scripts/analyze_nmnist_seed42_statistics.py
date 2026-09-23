"""Statistical analysis of Seed-42 N-MNIST attack protocol.

Reads the fully audited v3 artifact and produces:
  1. Per-class ASR tables
  2. Paired PGD vs TEMP-DRIFT-v2 McNemar tests
  3. Paired SNN vs QSNN McNemar tests
  4. Successful attack trace identification and overlap
  5. CSV/JSON tables and a Markdown report

All analysis is Seed-42 evidence only.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARTIFACT = Path("results/nmnist_attack_protocol_v3_auditable_seed42")
RECORDS_DIR = ARTIFACT / "records"
OUTPUT_DIR = ARTIFACT / "statistical_analysis"

EPSILONS_OF_INTEREST = [0.0001, 0.0025, 0.01, 0.05, 0.1]
MODELS = ["SNN", "QSNN"]
ATTACKS = ["PGD", "TEMP-DRIFT-v2"]
N_SAMPLES = 100
CLASS_NAMES = [f"Class {i}" for i in range(10)]


def load_records() -> list[dict]:
    """Load all NPZ records into a list of dicts."""
    records = []
    for npz_path in sorted(RECORDS_DIR.rglob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as z:
            rec = {}
            for k in z.files:
                v = z[k]
                if v.ndim == 0:
                    rec[k] = v.item()
                else:
                    rec[k] = v
            records.append(rec)
    return records


def build_index(records: list[dict]) -> dict:
    """Build a nested index: (sample_id, model, attack, epsilon) -> record."""
    idx = {}
    for rec in records:
        key = (rec["sample_id"], rec["model"], rec["attack"], rec["epsilon_fraction"])
        idx[key] = rec
    return idx


# ---------------------------------------------------------------------------
# 1. Per-class ASR
# ---------------------------------------------------------------------------

def compute_per_class_asr(records: list[dict]) -> list[dict]:
    """Per-class ASR for every model/attack/epsilon."""
    rows = []
    for model in MODELS:
        for attack in ATTACKS:
            for eps in [0.0] + EPSILONS_OF_INTEREST:
                class_counts = defaultdict(lambda: {"success": 0, "total": 0})
                for rec in records:
                    if rec["model"] != model or rec["attack"] != attack:
                        continue
                    if not np.isclose(rec["epsilon_fraction"], eps):
                        continue
                    label = int(rec["label"])
                    class_counts[label]["total"] += 1
                    if rec["stored_attack_success"]:
                        class_counts[label]["success"] += 1

                for cls in sorted(class_counts):
                    c = class_counts[cls]
                    n = c["total"]
                    s = c["success"]
                    asr = s / n if n > 0 else 0.0
                    rows.append({
                        "model": model,
                        "attack": attack,
                        "epsilon_fraction": eps,
                        "class": cls,
                        "successes": s,
                        "n": n,
                        "asr": asr,
                        "asr_display": f"{s}/{n} ({asr*100:.1f}%)",
                    })
    return rows


def compute_overall_asr(records: list[dict]) -> list[dict]:
    """Overall ASR (pooled across classes) for every model/attack/epsilon."""
    rows = []
    for model in MODELS:
        for attack in ATTACKS:
            for eps in [0.0] + EPSILONS_OF_INTEREST:
                total = 0
                success = 0
                for rec in records:
                    if rec["model"] != model or rec["attack"] != attack:
                        continue
                    if not np.isclose(rec["epsilon_fraction"], eps):
                        continue
                    total += 1
                    if rec["stored_attack_success"]:
                        success += 1
                asr = success / total if total > 0 else 0.0
                rows.append({
                    "model": model,
                    "attack": attack,
                    "epsilon_fraction": eps,
                    "successes": success,
                    "n": total,
                    "asr": asr,
                    "asr_display": f"{success}/{total} ({asr*100:.1f}%)",
                })
    return rows


# ---------------------------------------------------------------------------
# 2. Paired PGD vs TEMP-DRIFT-v2
# ---------------------------------------------------------------------------

def paired_attack_analysis(records: list[dict], model: str, eps: float) -> dict:
    """Paired PGD vs TEMP-DRIFT-v2 analysis for a given model and epsilon."""
    # Collect per-sample success vectors
    pgd_success = {}
    temp_success = {}
    for rec in records:
        if rec["model"] != model:
            continue
        if not np.isclose(rec["epsilon_fraction"], eps):
            continue
        sid = int(rec["sample_id"])
        if rec["attack"] == "PGD":
            pgd_success[sid] = bool(rec["stored_attack_success"])
        elif rec["attack"] == "TEMP-DRIFT-v2":
            temp_success[sid] = bool(rec["stored_attack_success"])

    common = sorted(set(pgd_success) & set(temp_success))
    n = len(common)

    # Paired outcomes
    pgd_only = sum(1 for s in common if pgd_success[s] and not temp_success[s])
    temp_only = sum(1 for s in common if not pgd_success[s] and temp_success[s])
    both = sum(1 for s in common if pgd_success[s] and temp_success[s])
    neither = sum(1 for s in common if not pgd_success[s] and not temp_success[s])

    pgd_n_success = sum(1 for s in common if pgd_success[s])
    temp_n_success = sum(1 for s in common if temp_success[s])
    pgd_asr = pgd_n_success / n if n > 0 else 0.0
    temp_asr = temp_n_success / n if n > 0 else 0.0

    # McNemar test: exact binomial test on discordant pairs
    # H0: P(PGD succeeds, TEMP fails) = P(PGD fails, TEMP succeeds)
    # Under H0, b ~ Binom(b+c, 0.5) where b=pgd_only, c=temp_only
    discordant = pgd_only + temp_only
    if discordant > 0:
        # Two-sided exact McNemar p-value
        mcnemar_p = stats.binomtest(pgd_only, discordant, 0.5, alternative="two-sided").pvalue
        # 95% CI for difference (pgd_asr - temp_asr)
        # Using exact Clopper-Pearson on discordant proportion
        # diff = (pgd_only - temp_only) / n
        diff = (pgd_only - temp_only) / n
        # Standard error for paired proportion difference
        # se = sqrt((b + c) / n^2) under the null, but for CI we use the observed
        se_diff = np.sqrt((pgd_only + temp_only) / (n ** 2)) if n > 0 else 0.0
        # Wilson-like CI for the difference
        z_crit = 1.96
        ci_low = diff - z_crit * se_diff
        ci_high = diff + z_crit * se_diff
    else:
        mcnemar_p = 1.0
        diff = 0.0
        ci_low = 0.0
        ci_high = 0.0

    return {
        "model": model,
        "epsilon_fraction": eps,
        "n": n,
        "pgd_successes": pgd_n_success,
        "temp_successes": temp_n_success,
        "pgd_asr": pgd_asr,
        "temp_asr": temp_asr,
        "pgd_only": pgd_only,
        "temp_only": temp_only,
        "both_succeed": both,
        "neither_succeed": neither,
        "paired_asr_difference": pgd_asr - temp_asr,
        "difference_direction": "PGD > TEMP" if diff > 0 else ("TEMP > PGD" if diff < 0 else "equal"),
        "mcnemar_exact_p": mcnemar_p,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "discordant_pairs": discordant,
        "pgd_only_samples": sorted([s for s in common if pgd_success[s] and not temp_success[s]]),
        "temp_only_samples": sorted([s for s in common if not pgd_success[s] and temp_success[s]]),
        "both_samples": sorted([s for s in common if pgd_success[s] and temp_success[s]]),
    }


# ---------------------------------------------------------------------------
# 3. Paired SNN vs QSNN
# ---------------------------------------------------------------------------

def paired_model_analysis(records: list[dict], attack: str, eps: float) -> dict:
    """Paired SNN vs QSNN analysis for a given attack and epsilon."""
    snn_success = {}
    qsnn_success = {}
    snn_label = {}
    qsnn_label = {}
    for rec in records:
        if rec["attack"] != attack:
            continue
        if not np.isclose(rec["epsilon_fraction"], eps):
            continue
        sid = int(rec["sample_id"])
        if rec["model"] == "SNN":
            snn_success[sid] = bool(rec["stored_attack_success"])
            snn_label[sid] = int(rec["label"])
        elif rec["model"] == "QSNN":
            qsnn_success[sid] = bool(rec["stored_attack_success"])
            qsnn_label[sid] = int(rec["label"])

    common = sorted(set(snn_success) & set(qsnn_success))
    n = len(common)

    # Paired outcomes (fail = not success, so "robust" = not success)
    snn_only_fails = sum(1 for s in common if not snn_success[s] and qsnn_success[s])
    qsnn_only_fails = sum(1 for s in common if snn_success[s] and not qsnn_success[s])
    both_fail = sum(1 for s in common if not snn_success[s] and not qsnn_success[s])
    both_robust = both_fail  # alias
    both_succeed = sum(1 for s in common if snn_success[s] and qsnn_success[s])

    snn_n_success = sum(1 for s in common if snn_success[s])
    qsnn_n_success = sum(1 for s in common if qsnn_success[s])
    snn_asr = snn_n_success / n if n > 0 else 0.0
    qsnn_asr = qsnn_n_success / n if n > 0 else 0.0

    # McNemar on discordant pairs: (SNN fails, QSNN succeeds) vs (SNN succeeds, QSNN fails)
    discordant = snn_only_fails + qsnn_only_fails
    if discordant > 0:
        mcnemar_p = stats.binomtest(snn_only_fails, discordant, 0.5, alternative="two-sided").pvalue
        diff = (qsnn_n_success - snn_n_success) / n
        se_diff = np.sqrt(discordant / (n ** 2)) if n > 0 else 0.0
        z_crit = 1.96
        ci_low = diff - z_crit * se_diff
        ci_high = diff + z_crit * se_diff
    else:
        mcnemar_p = 1.0
        diff = 0.0
        ci_low = 0.0
        ci_high = 0.0

    return {
        "attack": attack,
        "epsilon_fraction": eps,
        "n": n,
        "snn_successes": snn_n_success,
        "qsnn_successes": qsnn_n_success,
        "snn_asr": snn_asr,
        "qsnn_asr": qsnn_asr,
        "snn_only_fails": snn_only_fails,
        "qsnn_only_fails": qsnn_only_fails,
        "both_succeed": both_succeed,
        "both_robust": both_robust,
        "qsnn_asr_minus_snn_asr": qsnn_asr - snn_asr,
        "mcnemar_exact_p": mcnemar_p,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "discordant_pairs": discordant,
        "snn_only_fail_samples": sorted([s for s in common if not snn_success[s] and qsnn_success[s]]),
        "qsnn_only_fail_samples": sorted([s for s in common if snn_success[s] and not qsnn_success[s]]),
        "both_succeed_samples": sorted([s for s in common if snn_success[s] and qsnn_success[s]]),
    }


# ---------------------------------------------------------------------------
# 4. Successful attack trace identification
# ---------------------------------------------------------------------------

def identify_successful_traces(records: list[dict]) -> dict:
    """Identify all successful attacks and cross-model/attack overlap."""
    success_map = defaultdict(list)  # (model, attack, eps) -> [sample_ids]
    per_sample = defaultdict(list)   # sample_id -> [(model, attack, eps)]

    for rec in records:
        if not rec["stored_attack_success"]:
            continue
        model = rec["model"]
        attack = rec["attack"]
        eps = rec["epsilon_fraction"]
        sid = int(rec["sample_id"])
        label = int(rec["label"])
        success_map[(model, attack, eps)].append({
            "sample_id": sid,
            "label": label,
            "clean_prediction": int(rec["clean_prediction"]),
            "adversarial_prediction": int(rec["adversarial_prediction"]),
            "epsilon_absolute": float(rec["epsilon_absolute"]),
        })
        per_sample[sid].append((model, attack, eps, label))

    # Overlap analysis at 5% and 10%
    overlap = {}
    for eps in [0.05, 0.1]:
        # Which samples are attacked successfully by any attack on any model at this eps?
        all_success_samples = set()
        per_config = {}
        for model in MODELS:
            for attack in ATTACKS:
                key = (model, attack, eps)
                sids = set(r["sample_id"] for r in success_map.get(key, []))
                per_config[key] = sids
                all_success_samples |= sids

        # Cross-attack overlap (within QSNN)
        qsnn_pgd = per_config.get(("QSNN", "PGD", eps), set())
        qsnn_temp = per_config.get(("QSNN", "TEMP-DRIFT-v2", eps), set())
        snn_pgd = per_config.get(("SNN", "PGD", eps), set())
        snn_temp = per_config.get(("SNN", "TEMP-DRIFT-v2", eps), set())

        overlap[eps] = {
            "qsnn_pgd_only": sorted(qsnn_pgd - qsnn_temp),
            "qsnn_temp_only": sorted(qsnn_temp - qsnn_pgd),
            "qsnn_both": sorted(qsnn_pgd & qsnn_temp),
            "snn_pgd_only": sorted(snn_pgd - snn_temp),
            "snn_temp_only": sorted(snn_temp - snn_pgd),
            "snn_both": sorted(snn_pgd & snn_temp),
            "cross_model_pgd": sorted(qsnn_pgd & snn_pgd),
            "cross_model_temp": sorted(qsnn_temp & snn_temp),
            "any_success": sorted(all_success_samples),
        }

    return {
        "success_map": {f"model={k[0]}_attack={k[1]}_eps={k[2]}": v
                        for k, v in success_map.items()},
        "per_sample_successes": {str(k): v for k, v in per_sample.items()},
        "overlap_by_epsilon": overlap,
        "total_successful_records": sum(len(v) for v in success_map.values()),
    }


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None):
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict, path: Path):
    """Write JSON with numpy handling."""
    def default_handler(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(data, indent=2, default=default_handler), encoding="utf-8")


def generate_markdown_report(
    overall_asr: list[dict],
    per_class_asr: list[dict],
    paired_attack_results: list[dict],
    paired_model_results: list[dict],
    trace_info: dict,
) -> str:
    """Generate a concise Markdown statistical report."""
    lines = []
    lines.append("# Seed-42 N-MNIST Statistical Analysis Report")
    lines.append("")
    lines.append("**Evidence basis:** Seed-42 only, 100 common clean-correct samples, fully audited v3 artifact.")
    lines.append("")
    lines.append("**Do NOT overclaim significance with small event counts.**")
    lines.append("")

    # --- Section 1: Overall ASR ---
    lines.append("## 1. Overall Attack Success Rates (ASR)")
    lines.append("")
    lines.append("| Model | Attack | ε=0.0 | ε=0.01% | ε=0.25% | ε=1% | ε=5% | ε=10% |")
    lines.append("|-------|--------|-------|---------|---------|------|------|-------|")
    for model in MODELS:
        for attack in ATTACKS:
            cells = []
            for eps in [0.0] + EPSILONS_OF_INTEREST:
                match = [r for r in overall_asr
                         if r["model"] == model and r["attack"] == attack
                         and np.isclose(r["epsilon_fraction"], eps)]
                if match:
                    r = match[0]
                    cells.append(r["asr_display"])
                else:
                    cells.append("—")
            lines.append(f"| {model} | {attack} | {' | '.join(cells)} |")
    lines.append("")

    # --- Section 2: Per-class ASR at 5% and 10% ---
    lines.append("## 2. Per-Class ASR at ε=5% and ε=10%")
    lines.append("")
    for eps in [0.05, 0.1]:
        lines.append(f"### ε = {eps*100:.0f}%")
        lines.append("")
        lines.append("| Model | Attack | " + " | ".join(CLASS_NAMES) + " | Overall |")
        lines.append("|-------|--------|" + "|".join(["-----"] * 10) + "|---------|")
        for model in MODELS:
            for attack in ATTACKS:
                cells = []
                for cls in range(10):
                    match = [r for r in per_class_asr
                             if r["model"] == model and r["attack"] == attack
                             and np.isclose(r["epsilon_fraction"], eps)
                             and r["class"] == cls]
                    if match:
                        cells.append(match[0]["asr_display"])
                    else:
                        cells.append("0/100 (0.0%)")
                overall = [r for r in overall_asr
                           if r["model"] == model and r["attack"] == attack
                           and np.isclose(r["epsilon_fraction"], eps)]
                overall_cell = overall[0]["asr_display"] if overall else "—"
                lines.append(f"| {model} | {attack} | {' | '.join(cells)} | {overall_cell} |")
        lines.append("")

    # --- Section 3: Paired PGD vs TEMP-DRIFT-v2 ---
    lines.append("## 3. Paired Comparison: PGD vs TEMP-DRIFT-v2 (McNemar)")
    lines.append("")
    lines.append("Discordant-pair analysis on the same 100 samples per model.")
    lines.append("")
    for model in MODELS:
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| ε | PGD ASR | TEMP ASR | Δ ASR | PGD-only | TEMP-only | Both | Neither | Discordant | McNemar p | 95% CI for Δ |")
        lines.append("|---|---------|----------|-------|----------|-----------|------|---------|------------|-----------|--------------|")
        for r in paired_attack_results:
            if r["model"] != model:
                continue
            eps_str = f"{r['epsilon_fraction']*100:.2f}%"
            pgd_asr_str = f"{r['pgd_successes']}/{r['n']} ({r['pgd_asr']*100:.1f}%)"
            temp_asr_str = f"{r['temp_successes']}/{r['n']} ({r['temp_asr']*100:.1f}%)"
            delta = r["paired_asr_difference"]
            delta_str = f"{delta*100:+.1f}%"
            p_str = f"{r['mcnemar_exact_p']:.4f}" if r["mcnemar_exact_p"] >= 0.0001 else f"{r['mcnemar_exact_p']:.2e}"
            ci_str = f"[{r['ci_95_low']*100:+.1f}%, {r['ci_95_high']*100:+.1f}%]"
            lines.append(
                f"| {eps_str} | {pgd_asr_str} | {temp_asr_str} | {delta_str} "
                f"| {r['pgd_only']} | {r['temp_only']} | {r['both_succeed']} | {r['neither_succeed']} "
                f"| {r['discordant_pairs']} | {p_str} | {ci_str} |"
            )
        lines.append("")

    # --- Section 4: Paired SNN vs QSNN ---
    lines.append("## 4. Paired Comparison: SNN vs QSNN (McNemar)")
    lines.append("")
    lines.append("Discordant-pair analysis on the same 100 samples per attack.")
    lines.append("")
    for attack in ATTACKS:
        lines.append(f"### {attack}")
        lines.append("")
        lines.append("| ε | SNN ASR | QSNN ASR | Δ ASR (QSNN-SNN) | SNN-only fails | QSNN-only fails | Both succeed | Both robust | Discordant | McNemar p | 95% CI for Δ |")
        lines.append("|---|---------|----------|-------------------|----------------|-----------------|--------------|-------------|------------|-----------|--------------|")
        for r in paired_model_results:
            if r["attack"] != attack:
                continue
            eps_str = f"{r['epsilon_fraction']*100:.2f}%"
            snn_asr_str = f"{r['snn_successes']}/{r['n']} ({r['snn_asr']*100:.1f}%)"
            qsnn_asr_str = f"{r['qsnn_successes']}/{r['n']} ({r['qsnn_asr']*100:.1f}%)"
            delta = r["qsnn_asr_minus_snn_asr"]
            delta_str = f"{delta*100:+.1f}%"
            p_str = f"{r['mcnemar_exact_p']:.4f}" if r["mcnemar_exact_p"] >= 0.0001 else f"{r['mcnemar_exact_p']:.2e}"
            ci_str = f"[{r['ci_95_low']*100:+.1f}%, {r['ci_95_high']*100:+.1f}%]"
            lines.append(
                f"| {eps_str} | {snn_asr_str} | {qsnn_asr_str} | {delta_str} "
                f"| {r['snn_only_fails']} | {r['qsnn_only_fails']} | {r['both_succeed']} | {r['both_robust']} "
                f"| {r['discordant_pairs']} | {p_str} | {ci_str} |"
            )
        lines.append("")

    # --- Section 5: Successful attack traces ---
    lines.append("## 5. Successful Attack Traces and Overlap")
    lines.append("")
    lines.append(f"**Total successful records:** {trace_info['total_successful_records']}")
    lines.append("")

    for eps in [0.05, 0.1]:
        ov = trace_info["overlap_by_epsilon"].get(eps, {})
        lines.append(f"### ε = {eps*100:.0f}%")
        lines.append("")
        lines.append(f"- QSNN PGD only: {ov.get('qsnn_pgd_only', [])}")
        lines.append(f"- QSNN TEMP-DRIFT-v2 only: {ov.get('qsnn_temp_only', [])}")
        lines.append(f"- QSNN both attacks succeed: {ov.get('qsnn_both', [])}")
        lines.append(f"- SNN PGD only: {ov.get('snn_pgd_only', [])}")
        lines.append(f"- SNN TEMP-DRIFT-v2 only: {ov.get('snn_temp_only', [])}")
        lines.append(f"- SNN both attacks succeed: {ov.get('snn_both', [])}")
        lines.append(f"- Cross-model PGD overlap: {ov.get('cross_model_pgd', [])}")
        lines.append(f"- Cross-model TEMP overlap: {ov.get('cross_model_temp', [])}")
        lines.append(f"- Any success at ε={eps*100:.0f}%: {ov.get('any_success', [])}")
        lines.append("")

    # Per-sample detail for successful attacks at 5% and 10%
    lines.append("### Per-Sample Success Detail")
    lines.append("")
    for eps in [0.05, 0.1]:
        lines.append(f"#### ε = {eps*100:.0f}%")
        lines.append("")
        lines.append("| Sample | Label | Model | Attack | Clean Pred | Adv Pred |")
        lines.append("|--------|-------|-------|--------|------------|----------|")
        for model in MODELS:
            for attack in ATTACKS:
                key = f"model={model}_attack={attack}_eps={eps}"
                for entry in trace_info["success_map"].get(key, []):
                    lines.append(
                        f"| {entry['sample_id']} | {entry['label']} "
                        f"| {model} | {attack} "
                        f"| {entry['clean_prediction']} | {entry['adversarial_prediction']} |"
                    )
        lines.append("")

    # --- Caveats ---
    lines.append("## Caveats")
    lines.append("")
    lines.append("- All results are Seed-42 evidence only; not multi-seed robustness evidence.")
    lines.append("- With very few successes (0–5 per cell), McNemar tests have low power.")
    lines.append("- Small event counts mean p-values and CIs should be interpreted cautiously.")
    lines.append("- The five-seed attack campaign has NOT been started.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    records = load_records()
    print(f"  Loaded {len(records)} records")

    # 1. Per-class ASR
    print("Computing per-class ASR...")
    per_class = compute_per_class_asr(records)
    overall = compute_overall_asr(records)
    write_csv(per_class, OUTPUT_DIR / "per_class_asr.csv")
    write_csv(overall, OUTPUT_DIR / "overall_asr.csv")
    write_json({"per_class_asr": per_class, "overall_asr": overall},
               OUTPUT_DIR / "asr_tables.json")

    # 2. Paired PGD vs TEMP-DRIFT-v2
    print("Running paired PGD vs TEMP-DRIFT-v2 McNemar tests...")
    paired_attack = []
    for model in MODELS:
        for eps in [0.0] + EPSILONS_OF_INTEREST:
            result = paired_attack_analysis(records, model, eps)
            paired_attack.append(result)
    write_csv([{k: v for k, v in r.items()
                if k not in ("pgd_only_samples", "temp_only_samples", "both_samples")}
               for r in paired_attack],
              OUTPUT_DIR / "paired_attack_comparison.csv")
    write_json(paired_attack, OUTPUT_DIR / "paired_attack_comparison.json")

    # 3. Paired SNN vs QSNN
    print("Running paired SNN vs QSNN McNemar tests...")
    paired_model = []
    for attack in ATTACKS:
        for eps in [0.0] + EPSILONS_OF_INTEREST:
            result = paired_model_analysis(records, attack, eps)
            paired_model.append(result)
    write_csv([{k: v for k, v in r.items()
                if k not in ("snn_only_fail_samples", "qsnn_only_fail_samples", "both_succeed_samples")}
               for r in paired_model],
              OUTPUT_DIR / "paired_model_comparison.csv")
    write_json(paired_model, OUTPUT_DIR / "paired_model_comparison.json")

    # 4. Trace identification
    print("Identifying successful attack traces...")
    traces = identify_successful_traces(records)
    write_json(traces, OUTPUT_DIR / "successful_attack_traces.json")

    # 5. Markdown report
    print("Generating Markdown report...")
    report = generate_markdown_report(overall, per_class, paired_attack, paired_model, traces)
    (OUTPUT_DIR / "statistical_report_seed42.md").write_text(report, encoding="utf-8")

    print(f"\nAll outputs written to {OUTPUT_DIR}/")
    print("  - statistical_report_seed42.md")
    print("  - overall_asr.csv")
    print("  - per_class_asr.csv")
    print("  - asr_tables.json")
    print("  - paired_attack_comparison.csv / .json")
    print("  - paired_model_comparison.csv / .json")
    print("  - successful_attack_traces.json")


if __name__ == "__main__":
    main()
