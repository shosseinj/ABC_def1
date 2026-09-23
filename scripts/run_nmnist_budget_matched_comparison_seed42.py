"""Budget-matched PGD vs TEMP-DRIFT-v2 comparison on frozen Seed-42 N-MNIST.

Three comparison conditions:
  (A) Original fixed-budget: PGD=20 steps, TEMP=1600 candidates
  (B) Evaluation/query-budget matched: total query cost equalized
  (C) Wall-clock/compute matched: runtime equalized per model

All use the same frozen 100-sample manifest, canonical preprocessing,
and frozen SNN/QSNN-v3 models. No model retraining. No five-seed campaign.

Focus epsilon: 5% and 10% (nonzero ASR region).
"""
from __future__ import annotations

import csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.nmnist_snn import NMNISTConvSNN
from models.nmnist_hybrid_qsnn import NMNISTHybridQSNNV2, NMNISTSpatialLIFExtractor
from scripts.run_nmnist_hybrid_qsnn_seed42 import sha256
from scripts.run_nmnist_qsnn_v3_multiseed import load_fixed_frontend, QUANTUM_CONFIG, FRONTEND_CONFIG
from scripts.run_nmnist_attack_protocol_v2_canonical_seed42 import canonical_frames_torch

SEED = 42
SPLIT = ROOT / "results/nmnist_snn_multiseed_split.json"
SNN_CKPT = ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt"
QSNN_CKPT = ROOT / "checkpoints/nmnist_hybrid_qsnn_seed42/nmnist_qsnn_v3_multiseed/seed42_best.pt"
MANIFEST_PATH = ROOT / "results/nmnist_common_attack_protocol_seed42/common_clean_correct_manifest.json"
DEST = ROOT / "results/nmnist_budget_matched_comparison_seed42"

EPSILONS = [0.05, 0.10]
PGD_STEPS_ORIGINAL = 20
TEMP_INITIAL_ORIGINAL = 600
TEMP_GENERATIONS_ORIGINAL = 4
TEMP_PER_GEN_ORIGINAL = 250
TEMP_TOTAL_ORIGINAL = TEMP_INITIAL_ORIGINAL + TEMP_GENERATIONS_ORIGINAL * TEMP_PER_GEN_ORIGINAL
ALPHA_BACKWARD = 0.5


def load_models():
    snn = NMNISTConvSNN(0.5).cuda().eval()
    snn.load_state_dict(torch.load(SNN_CKPT, map_location="cuda", weights_only=True)["model_state"])
    extractor = NMNISTSpatialLIFExtractor(**FRONTEND_CONFIG)
    qsnn = NMNISTHybridQSNNV2(extractor=extractor, latent_dim=32, **QUANTUM_CONFIG).cuda().eval()
    q = torch.load(QSNN_CKPT, map_location="cuda", weights_only=True)
    load_fixed_frontend(qsnn)
    qsnn.load_state_dict(q["model_state"])
    return {"SNN": snn, "QSNN": qsnn}


def project(candidate, clean, eps):
    out = np.clip(candidate, clean - eps, clean + eps)
    out = np.maximum.accumulate(out)
    return np.clip(out, clean[0], clean[-1])


def bins(events, timestamps):
    t0 = float(events["t"][0])
    duration = max(float(events["t"][-1]) - t0 + 1, 1.0)
    return np.minimum(((np.asarray(timestamps) - t0) * 10 // duration).astype(int), 9)


def true_class_margin(logits_out, label):
    competitors = logits_out.clone()
    competitors[:, int(label)] = -torch.inf
    return logits_out[:, int(label)] - competitors.max(1).values


def frame_metrics_from_events(events, clean_t, adv_t):
    clean_frame = canonical_frames_torch(events, torch.tensor(clean_t, device="cuda", dtype=torch.float32))
    adv_frame = canonical_frames_torch(events, torch.tensor(adv_t, device="cuda", dtype=torch.float32))
    d = (adv_frame - clean_frame).float()
    flat = d.flatten()
    return {
        "frame_l0": int((flat != 0).sum().item()),
        "frame_l1": float(flat.abs().sum().item()),
        "frame_l2": float(torch.linalg.vector_norm(flat, 2).item()),
        "frame_linf": float(flat.abs().max().item()),
        "changed_frame_elements": int((flat != 0).sum().item()),
    }


def temporal_metrics(clean_t, adv_t, events):
    clean_bins = bins(events, clean_t)
    adv_bins = bins(events, adv_t)
    delta = np.abs(adv_t - clean_t)
    duration = float(clean_t[-1] - clean_t[0] + 1)
    return {
        "mean_normalized_abs_dt": float(np.mean(delta / duration)),
        "median_normalized_abs_dt": float(np.median(delta / duration)),
        "max_normalized_abs_dt": float(np.max(delta / duration)),
        "fraction_events_bin_changed": float(np.mean(clean_bins != adv_bins)),
        "events_bin_changed": int(np.sum(clean_bins != adv_bins)),
    }


def pgd_attack(model, events, label, eps, n_steps):
    clean = np.asarray(events["t"], dtype=np.float64)
    x = torch.tensor(clean, device="cuda", dtype=torch.float32)
    best, best_loss = clean.copy(), -float("inf")
    fwd_count, bwd_count = 0, 0

    for step in range(n_steps + 1):
        with torch.no_grad():
            out = model(canonical_frames_torch(events, x))
            fwd_count += 1
            loss = float(F.cross_entropy(out, torch.tensor([label], device="cuda")))
        if loss > best_loss:
            best_loss = loss
            best = x.detach().cpu().numpy().copy()
        if step == n_steps:
            break
        x = x.detach().requires_grad_(True)
        out = model(canonical_frames_torch(events, x))
        fwd_count += 1
        loss = F.cross_entropy(out, torch.tensor([label], device="cuda"))
        grad = torch.autograd.grad(loss, x)[0]
        bwd_count += 1
        x = torch.tensor(
            project((x.detach() + eps / 5 * grad.sign()).cpu().numpy(), clean, eps),
            device="cuda", dtype=torch.float32,
        )

    with torch.no_grad():
        prediction = int(model(canonical_frames_torch(events, torch.tensor(best, device="cuda", dtype=torch.float32))).argmax(1).item())
    fwd_count += 1
    return best, {
        "model_forward_evaluations": fwd_count,
        "backward_evaluations": bwd_count,
        "candidate_evaluations": 0,
        "total_query_cost_forward_equiv": fwd_count + ALPHA_BACKWARD * bwd_count,
        "prediction": prediction,
        "success": prediction != label,
    }


@torch.no_grad()
def evaluate_candidates(model, events, candidates, label, chunk=64):
    values, predictions = [], []
    for start in range(0, len(candidates), chunk):
        timestamps = torch.as_tensor(candidates[start:start + chunk], device="cuda", dtype=torch.float32)
        output = model(canonical_frames_torch(events, timestamps))
        values.extend(true_class_margin(output, label).cpu().tolist())
        predictions.extend(output.argmax(1).cpu().tolist())
    return np.asarray(values), np.asarray(predictions)


def temp_drift_attack(model, events, label, eps, initial, generations, per_gen, rng):
    clean = np.asarray(events["t"], dtype=np.float64)
    n = len(clean)
    total_candidates = initial + generations * per_gen

    if initial <= 0:
        return clean, {
            "model_forward_evaluations": 0, "backward_evaluations": 0,
            "candidate_evaluations": 0, "total_query_cost_forward_equiv": 0.0,
            "prediction": -1, "success": False,
        }

    pool = np.asarray([project(clean + rng.uniform(-eps, eps, n), clean, eps) for _ in range(initial)])
    values, predictions = evaluate_candidates(model, events, pool, label)

    scales = [0.50, 0.30, 0.18, 0.10]
    for gen_idx in range(generations):
        scale = scales[gen_idx % len(scales)]
        elite_indices = np.argsort(values, kind="stable")[:12]
        elite = pool[elite_indices]
        children = []
        for i in range(per_gen):
            parent = elite[i % len(elite)]
            proposal = parent + 0.5 * (elite[rng.integers(12)] - elite[rng.integers(12)])
            proposal += rng.normal(0, eps * scale, n)
            children.append(project(proposal, clean, eps))
        children = np.asarray(children)
        child_values, child_predictions = evaluate_candidates(model, events, children, label)
        pool = np.vstack((elite, children))
        values = np.concatenate((values[elite_indices], child_values))
        predictions = np.concatenate((predictions[elite_indices], child_predictions))

    best = int(np.argmin(values))
    return pool[best], {
        "model_forward_evaluations": total_candidates,
        "backward_evaluations": 0,
        "candidate_evaluations": total_candidates,
        "total_query_cost_forward_equiv": float(total_candidates),
        "prediction": int(predictions[best]),
        "success": int(predictions[best]) != label,
    }


def compute_budget_definitions():
    pgd_fwd = PGD_STEPS_ORIGINAL + 1
    pgd_bwd = PGD_STEPS_ORIGINAL
    pgd_cost = pgd_fwd + ALPHA_BACKWARD * pgd_bwd

    budgets = {}

    budgets["A_original"] = {
        "label": "A: Original fixed-budget",
        "pgd": {"n_steps": PGD_STEPS_ORIGINAL},
        "temp": {"initial": TEMP_INITIAL_ORIGINAL, "generations": TEMP_GENERATIONS_ORIGINAL, "per_gen": TEMP_PER_GEN_ORIGINAL},
        "pgd_query_cost": pgd_cost,
        "temp_query_cost": float(TEMP_TOTAL_ORIGINAL),
    }

    budgets["B1_forward_match"] = {
        "label": "B1: Forward-count matched (TEMP=PGD forwards only)",
        "pgd": {"n_steps": PGD_STEPS_ORIGINAL},
        "temp": {"initial": pgd_fwd, "generations": 0, "per_gen": 0},
        "pgd_query_cost": pgd_cost,
        "temp_query_cost": float(pgd_fwd),
    }

    temp_cost_match = int(round(pgd_cost))
    budgets["B2_query_cost_match"] = {
        "label": f"B2: Query-cost matched (alpha={ALPHA_BACKWARD})",
        "pgd": {"n_steps": PGD_STEPS_ORIGINAL},
        "temp": {"initial": temp_cost_match, "generations": 0, "per_gen": 0},
        "pgd_query_cost": pgd_cost,
        "temp_query_cost": float(temp_cost_match),
    }

    budgets["B3_evolved_match"] = {
        "label": "B3: Query-cost matched with evolution (15+16)",
        "pgd": {"n_steps": PGD_STEPS_ORIGINAL},
        "temp": {"initial": 15, "generations": 1, "per_gen": 16},
        "pgd_query_cost": pgd_cost,
        "temp_query_cost": float(15 + 1 * 16),
    }

    return budgets, pgd_cost


def calibrate_wallclock(model, events, label, eps):
    rng = np.random.default_rng(99999)

    pgd_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pgd_attack(model, events, label, eps, PGD_STEPS_ORIGINAL)
        pgd_times.append(time.perf_counter() - t0)
    pgd_mean = float(np.mean(pgd_times))

    temp_profile = []
    for n_init in [10, 20, 30, 50, 80, 100, 150, 200]:
        t0 = time.perf_counter()
        temp_drift_attack(model, events, label, eps, n_init, 0, 0, rng)
        elapsed = time.perf_counter() - t0
        temp_profile.append({"initial": n_init, "seconds": float(elapsed)})
        if elapsed > pgd_mean * 2:
            break

    best = min(temp_profile, key=lambda x: abs(x["seconds"] - pgd_mean))
    return {
        "pgd_mean_seconds": pgd_mean,
        "temp_profile": temp_profile,
        "wallclock_matched_initial": best["initial"],
        "wallclock_matched_seconds": best["seconds"],
    }


def run_single(model, events, label, eps, attack_fn, rng):
    clean_t = np.asarray(events["t"], dtype=np.float64)
    started = time.perf_counter()
    adv_t, stats = attack_fn(model, events, label, eps, rng)
    runtime = time.perf_counter() - started
    adv_t = project(adv_t, clean_t, eps)

    fm = frame_metrics_from_events(events, clean_t, adv_t)
    tm = temporal_metrics(clean_t, adv_t, events)

    with torch.no_grad():
        clean_pred = int(model(canonical_frames_torch(events, torch.tensor(clean_t, device="cuda", dtype=torch.float32))).argmax(1).item())

    return {
        "epsilon_fraction": eps, "label": int(label),
        "clean_prediction": clean_pred,
        "adversarial_prediction": stats["prediction"],
        "attack_success": stats["success"],
        "model_forward_evaluations": stats["model_forward_evaluations"],
        "backward_evaluations": stats["backward_evaluations"],
        "candidate_evaluations": stats["candidate_evaluations"],
        "total_query_cost_forward_equiv": stats["total_query_cost_forward_equiv"],
        "runtime_seconds": runtime,
        **fm, **tm,
    }


def paired_analysis(results_a, results_b):
    a_success = {r["sample_id"]: r["attack_success"] for r in results_a}
    b_success = {r["sample_id"]: r["attack_success"] for r in results_b}
    common = sorted(set(a_success) & set(b_success))
    n = len(common)

    a_only = sum(1 for s in common if a_success[s] and not b_success[s])
    b_only = sum(1 for s in common if not a_success[s] and b_success[s])
    both = sum(1 for s in common if a_success[s] and b_success[s])
    neither = sum(1 for s in common if not a_success[s] and not b_success[s])

    discordant = a_only + b_only
    p_val = sp_stats.binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue if discordant > 0 else 1.0

    a_asr = sum(1 for s in common if a_success[s]) / n if n else 0.0
    b_asr = sum(1 for s in common if b_success[s]) / n if n else 0.0
    diff = b_asr - a_asr
    se = np.sqrt(discordant) / n if n > 0 else 0.0

    return {
        "n": n, "a_asr": a_asr, "b_asr": b_asr, "paired_difference": diff,
        "a_only": a_only, "b_only": b_only,
        "both_succeed": both, "neither_succeed": neither,
        "discordant_pairs": discordant, "mcnemar_p": p_val,
        "ci_95_low": diff - 1.96 * se, "ci_95_high": diff + 1.96 * se,
    }


def summarize_condition(rows):
    """Aggregate per-sample rows into a summary dict."""
    if not rows:
        return {}
    n = len(rows)
    successes = sum(1 for r in rows if r["attack_success"])
    return {
        "n": n,
        "successes": successes,
        "asr": successes / n if n else 0.0,
        "mean_runtime": float(np.mean([r["runtime_seconds"] for r in rows])),
        "median_runtime": float(np.median([r["runtime_seconds"] for r in rows])),
        "mean_forwards": float(np.mean([r["model_forward_evaluations"] for r in rows])),
        "mean_backwards": float(np.mean([r["backward_evaluations"] for r in rows])),
        "mean_candidates": float(np.mean([r["candidate_evaluations"] for r in rows])),
        "mean_query_cost": float(np.mean([r["total_query_cost_forward_equiv"] for r in rows])),
        "mean_frame_l1": float(np.mean([r["frame_l1"] for r in rows])),
        "mean_frame_linf": float(np.mean([r["frame_linf"] for r in rows])),
        "max_frame_linf": float(max(r["frame_linf"] for r in rows)),
        "mean_norm_dt": float(np.mean([r["mean_normalized_abs_dt"] for r in rows])),
        "median_norm_dt": float(np.median([r["median_normalized_abs_dt"] for r in rows])),
        "mean_bin_change_frac": float(np.mean([r["fraction_events_bin_changed"] for r in rows])),
        "successful_ids": sorted([r["sample_id"] for r in rows if r["attack_success"]]),
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=True)

    print("Loading models...")
    models = load_models()

    budgets, pgd_cost = compute_budget_definitions()
    print(f"\nBudget definitions (PGD original: {PGD_STEPS_ORIGINAL} steps, alpha={ALPHA_BACKWARD}):")
    for name, b in budgets.items():
        print(f"  {name}: {b['label']}")
        print(f"    PGD: steps={b['pgd']['n_steps']}, cost={b['pgd_query_cost']:.1f}")
        print(f"    TEMP: initial={b['temp']['initial']}, gen={b['temp']['generations']}, per_gen={b['temp']['per_gen']}, cost={b['temp_query_cost']:.1f}")

    # Phase 1: wall-clock calibration on first sample
    print("\n--- Phase 1: Wall-clock calibration ---")
    wallclock_info = {}
    for model_name, model in models.items():
        item = manifest["samples"][0]
        sid, label = int(item["sample_id"]), int(item["label"])
        events, _ = dataset[sid]
        wc = calibrate_wallclock(model, events, label, 0.10)
        wallclock_info[model_name] = wc
        print(f"  {model_name}: PGD mean={wc['pgd_mean_seconds']:.3f}s, TEMP matched init={wc['wallclock_matched_initial']} ({wc['wallclock_matched_seconds']:.3f}s)")

    # Add wall-clock condition
    for model_name in models:
        wc = wallclock_info[model_name]
        budgets[f"C_wallclock_{model_name}"] = {
            "label": f"C: Wall-clock matched for {model_name} (TEMP init={wc['wallclock_matched_initial']})",
            "pgd": {"n_steps": PGD_STEPS_ORIGINAL},
            "temp": {"initial": wc["wallclock_matched_initial"], "generations": 0, "per_gen": 0},
            "pgd_query_cost": pgd_cost,
            "temp_query_cost": float(wc["wallclock_matched_initial"]),
        }

    # Phase 2: run all conditions
    print("\n--- Phase 2: Running all matched comparisons ---")
    all_results = []

    # Sample-level progress (independent from attack-result row count).
    # A sample is considered fully processed in this non-resume runner only
    # after both models have completed all configured epsilons/conditions.
    total_samples = len(manifest["samples"])
    completed_sample_ids = set()
    sample_expected_records = {}
    for item in manifest["samples"]:
        _sid = int(item["sample_id"])
        _expected = 0
        for _model_name in models:
            _relevant = {
                k: v for k, v in budgets.items()
                if not k.startswith("C_wallclock_") or _model_name in k
            }
            _expected += len(EPSILONS) * len(_relevant) * 2  # PGD + TEMP
        sample_expected_records[_sid] = _expected

    def print_sample_progress(current_sid=None):
        completed = len(completed_sample_ids)
        pct = (100.0 * completed / total_samples) if total_samples else 100.0
        current_txt = f" | current_sample={current_sid}" if current_sid is not None else ""
        print(
            f"SAMPLE PROGRESS: {completed}/{total_samples} "
            f"({pct:.2f}%) | remaining={total_samples - completed}{current_txt}",
            flush=True,
        )

    print_sample_progress()

    for model_name, model in models.items():
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        relevant_budgets = {k: v for k, v in budgets.items() if not k.startswith("C_wallclock_") or model_name in k}

        for sample_idx, item in enumerate(manifest["samples"]):
            sid = int(item["sample_id"])
            label = int(item["label"])
            events, actual = dataset[sid]
            if int(actual) != label:
                raise RuntimeError(f"Manifest mismatch at {sid}")

            if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
                print(f"  Sample {sample_idx+1}/100 (id={sid})")

            for eps in EPSILONS:
                duration = float(events["t"][-1] - events["t"][0] + 1)
                epsilon = eps * duration
                rng_base = SEED + sample_idx * 1009 + int(eps * 1000000)

                for budget_name, budget_def in relevant_budgets.items():
                    pgd_steps = budget_def["pgd"]["n_steps"]
                    temp_init = budget_def["temp"]["initial"]
                    temp_gen = budget_def["temp"]["generations"]
                    temp_per = budget_def["temp"]["per_gen"]

                    rng_pgd = np.random.default_rng(rng_base)
                    pgd_fn = lambda m, ev, lb, ep, rg, s=pgd_steps: pgd_attack(m, ev, lb, ep, s)
                    pgd_result = run_single(model, events, label, epsilon, pgd_fn, rng_pgd)
                    pgd_result.update({
                        "sample_id": sid, "model": model_name, "attack": "PGD",
                        "budget_condition": budget_name, "budget_label": budget_def["label"],
                    })
                    all_results.append(pgd_result)

                    rng_temp = np.random.default_rng(rng_base)
                    temp_fn = lambda m, ev, lb, ep, rg, i=temp_init, g=temp_gen, p=temp_per: temp_drift_attack(m, ev, lb, ep, i, g, p, rg)
                    temp_result = run_single(model, events, label, epsilon, temp_fn, rng_temp)
                    temp_result.update({
                        "sample_id": sid, "model": model_name, "attack": "TEMP-DRIFT-v2",
                        "budget_condition": budget_name, "budget_label": budget_def["label"],
                    })
                    all_results.append(temp_result)

            # Check whether this sample has now accumulated every expected
            # result row across all models/epsilons/conditions processed so far.
            sample_rows = sum(1 for r in all_results if int(r["sample_id"]) == sid)
            if (
                sample_rows == sample_expected_records[sid]
                and sid not in completed_sample_ids
            ):
                completed_sample_ids.add(sid)
                pct = 100.0 * len(completed_sample_ids) / total_samples
                print(
                    f"[SAMPLE {len(completed_sample_ids)}/{total_samples} | "
                    f"{pct:.2f}%] sample_id={sid} COMPLETE",
                    flush=True,
                )
                print_sample_progress(current_sid=sid)

    # Phase 3: save raw results
    print("\n--- Phase 3: Saving results ---")
    csv_path = DEST / "per_sample_results.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
    print(f"  Saved {len(all_results)} rows to {csv_path}")

    # Phase 4: aggregate and paired analysis
    print("\n--- Phase 4: Aggregation and paired analysis ---")
    summaries = []
    paired_results = []

    for model_name in models:
        for budget_name in budgets:
            for eps in EPSILONS:
                pgd_rows = [r for r in all_results if r["model"] == model_name and r["attack"] == "PGD"
                            and r["budget_condition"] == budget_name and r["epsilon_fraction"] == eps]
                temp_rows = [r for r in all_results if r["model"] == model_name and r["attack"] == "TEMP-DRIFT-v2"
                             and r["budget_condition"] == budget_name and r["epsilon_fraction"] == eps]

                pgd_summary = summarize_condition(pgd_rows)
                temp_summary = summarize_condition(temp_rows)

                summaries.append({
                    "model": model_name, "budget_condition": budget_name,
                    "budget_label": budgets[budget_name]["label"],
                    "epsilon_fraction": eps,
                    "pgd": pgd_summary, "temp": temp_summary,
                })

                if pgd_rows and temp_rows:
                    paired = paired_analysis(pgd_rows, temp_rows)
                    paired.update({
                        "model": model_name, "budget_condition": budget_name,
                        "budget_label": budgets[budget_name]["label"],
                        "epsilon_fraction": eps,
                        "pgd_asr": pgd_summary.get("asr", 0),
                        "temp_asr": temp_summary.get("asr", 0),
                        "pgd_successful_ids": pgd_summary.get("successful_ids", []),
                        "temp_successful_ids": temp_summary.get("successful_ids", []),
                    })
                    paired_results.append(paired)

    # Save summaries
    (DEST / "budget_definitions.json").write_text(json.dumps(budgets, indent=2, default=str), encoding="utf-8")
    (DEST / "condition_summaries.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    (DEST / "paired_comparisons.json").write_text(json.dumps(paired_results, indent=2, default=str), encoding="utf-8")
    (DEST / "wallclock_calibration.json").write_text(json.dumps(wallclock_info, indent=2, default=str), encoding="utf-8")

    # Phase 5: Markdown report
    print("\n--- Phase 5: Generating report ---")
    report = generate_report(budgets, summaries, paired_results, wallclock_info)
    (DEST / "budget_matched_report_seed42.md").write_text(report, encoding="utf-8")

    print(f"\nAll outputs saved to {DEST}/")
    print("  - per_sample_results.csv")
    print("  - budget_definitions.json")
    print("  - condition_summaries.json")
    print("  - paired_comparisons.json")
    print("  - wallclock_calibration.json")
    print("  - budget_matched_report_seed42.md")


def generate_report(budgets, summaries, paired, wallclock):
    L = []
    L.append("# Budget-Matched PGD vs TEMP-DRIFT-v2 Comparison (Seed-42)")
    L.append("")
    L.append("**Evidence basis:** Seed-42 only, 100 common clean-correct samples.")
    L.append("**Frozen models:** SNN, QSNN-v3. No retraining. No five-seed campaign.")
    L.append("")

    # Budget definitions
    L.append("## Budget Definitions")
    L.append("")
    L.append(f"**Query-cost model:** backward pass = {ALPHA_BACKWARD} forward equivalents")
    L.append(f"**PGD original:** {PGD_STEPS_ORIGINAL} steps = {PGD_STEPS_ORIGINAL+1} forwards + {PGD_STEPS_ORIGINAL} backward = {PGD_STEPS_ORIGINAL+1 + ALPHA_BACKWARD*PGD_STEPS_ORIGINAL:.0f} forward-equiv")
    L.append(f"**TEMP original:** {TEMP_TOTAL_ORIGINAL} candidates = {TEMP_TOTAL_ORIGINAL} forwards")
    L.append("")
    L.append("| Condition | PGD steps | PGD cost | TEMP initial | TEMP gen x per | TEMP cost |")
    L.append("|-----------|-----------|----------|--------------|----------------|-----------|")
    for name, b in budgets.items():
        pgd_s = b["pgd"]["n_steps"]
        pgd_c = b["pgd_query_cost"]
        t_i = b["temp"]["initial"]
        t_g = b["temp"]["generations"]
        t_p = b["temp"]["per_gen"]
        t_c = b["temp_query_cost"]
        L.append(f"| {b['label']} | {pgd_s} | {pgd_c:.1f} | {t_i} | {t_g}x{t_p} | {t_c:.0f} |")
    L.append("")

    # Wall-clock calibration
    L.append("## Wall-Clock Calibration")
    L.append("")
    for model_name, wc in wallclock.items():
        L.append(f"### {model_name}")
        L.append(f"- PGD mean runtime: {wc['pgd_mean_seconds']:.3f}s")
        L.append(f"- TEMP matched initial: {wc['wallclock_matched_initial']} ({wc['wallclock_matched_seconds']:.3f}s)")
        L.append("- TEMP runtime profile:")
        for p in wc["temp_profile"]:
            L.append(f"  - init={p['initial']}: {p['seconds']:.3f}s")
        L.append("")

    # ASR summary table
    L.append("## ASR Summary by Condition")
    L.append("")
    L.append("| Model | Condition | ε | PGD ASR | TEMP ASR | Δ (T-P) | McNemar p | PGD IDs | TEMP IDs |")
    L.append("|-------|-----------|---|---------|----------|---------|-----------|---------|----------|")
    for p in paired:
        model = p["model"]
        cond = p["budget_condition"]
        eps = p["epsilon_fraction"]
        pgd_a = p.get("pgd_asr", 0)
        temp_a = p.get("temp_asr", 0)
        d = p["paired_difference"]
        mp = p["mcnemar_p"]
        pgd_ids = p.get("pgd_successful_ids", [])
        temp_ids = p.get("temp_successful_ids", [])
        p_str = f"{mp:.4f}" if mp >= 0.0001 else f"{mp:.2e}"
        L.append(f"| {model} | {cond} | {eps*100:.0f}% | {pgd_a*100:.0f}% | {temp_a*100:.0f}% | {d*100:+.1f}% | {p_str} | {pgd_ids} | {temp_ids} |")
    L.append("")

    # Detailed paired results
    L.append("## Detailed Paired Comparisons")
    L.append("")
    for p in paired:
        if p["epsilon_fraction"] not in [0.05, 0.10]:
            continue
        if p["discordant_pairs"] == 0 and p["both_succeed"] == 0:
            continue
        L.append(f"### {p['model']} | {p['budget_condition']} | ε={p['epsilon_fraction']*100:.0f}%")
        L.append(f"- PGD ASR: {p['pgd_asr']*100:.1f}% | TEMP ASR: {p['temp_asr']*100:.1f}%")
        L.append(f"- PGD-only: {p['a_only']} | TEMP-only: {p['b_only']} | Both: {p['both_succeed']} | Neither: {p['neither_succeed']}")
        L.append(f"- Discordant: {p['discordant_pairs']} | McNemar p: {p['mcnemar_p']:.4f}")
        L.append(f"- 95% CI for Δ: [{p['ci_95_low']*100:+.1f}%, {p['ci_95_high']*100:+.1f}%]")
        L.append(f"- PGD successes: {p.get('pgd_successful_ids', [])}")
        L.append(f"- TEMP successes: {p.get('temp_successful_ids', [])}")
        L.append("")

    # Distortion comparison
    L.append("## Distortion Metrics by Condition (successful attacks only)")
    L.append("")
    L.append("| Model | Condition | ε | Attack | Mean L1 | Max Linf | Mean |dt|/dur | Mean bin change |")
    L.append("|-------|-----------|---|--------|---------|----------|-------------|-----------------|")
    for s in summaries:
        eps = s["epsilon_fraction"]
        if eps not in [0.05, 0.10]:
            continue
        model = s["model"]
        cond = s["budget_condition"]
        for attack_key, label in [("pgd", "PGD"), ("temp", "TEMP")]:
            sm = s[attack_key]
            if not sm:
                continue
            L.append(f"| {model} | {cond} | {eps*100:.0f}% | {label} | {sm.get('mean_frame_l1',0):.1f} | {sm.get('max_frame_linf',0):.1f} | {sm.get('mean_norm_dt',0):.6f} | {sm.get('mean_bin_change_frac',0):.4f} |")
    L.append("")

    # Caveats
    L.append("## Caveats")
    L.append("")
    L.append("- All results are Seed-42 evidence only; not multi-seed robustness evidence.")
    L.append("- With very few successes (0-5 per cell), McNemar tests have very low power.")
    L.append("- Do NOT claim one attack is stronger unless the relevant matched comparison supports it.")
    L.append("- The five-seed attack campaign has NOT been started.")
    L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
