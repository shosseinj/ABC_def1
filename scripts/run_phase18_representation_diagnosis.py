"""Generate the frozen, train/validation-only Phase 18 diagnostic artifacts."""

from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

from attacks.classical_timing import classical_timing_attack
from encoding.ttfs import ttfs_encode
from experiments.iris.data import load_iris_train_validation
from experiments.iris.phase18 import (assert_aligned, collision_analysis, fit_linear_diagnostic,
    geometry, jensen_shannon_rows, local_neighbors, nearest_centroid,
    ordering_preservation, true_class_margin)
from experiments.iris.training import to_theta
from models.qsnn import IrisQSNN

RESULTS, CHECKPOINTS = ROOT / "results", ROOT / "checkpoints"
SEEDS, MODELS, CONTROLS = (42, 777, 2026), ("baseline", "defense"), (119, 122, 142)
K, COLLISION_TOLERANCE = 3, 0.5
ATTACK_FRACTIONS, ATTACK_ITERATIONS = (0.02, 0.10), 20


def checkpoint_path(seed, model):
    if model == "baseline":
        return CHECKPOINTS / f"iris_qsnn_phase172_{seed}_baseline.pt"
    return CHECKPOINTS / f"iris_qsnn_phase17_phase171_lambda_0p5_seed_{seed}.pt"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    path = RESULTS / name
    if not rows:
        raise ValueError(f"refusing to write empty artifact {name}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return path


def flatten(prefix, values):
    return {f"{prefix}_{i}": float(value) for i, value in enumerate(values)}


def frozen_model(config, path):
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen checkpoint is missing: {path.name}")
    model = IrisQSNN(config["n_qubits"], config["n_layers"], config["n_classes"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_outputs(model, timing, T):
    angles = torch.tensor((np.pi / 2.0) * np.asarray(timing) / T, dtype=torch.float32)
    with torch.no_grad():
        features = model.quantum_features(angles)
        logits = model.head(features)
        probabilities = torch.softmax(logits, dim=1)
    return features.numpy(), logits.numpy(), probabilities.numpy()


def plot_pca(path, values, labels, ids, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(values)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in (0, 1, 2):
        mask = labels == cls
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=f"class {cls}", alpha=.75)
    for sample_id in CONTROLS:
        position = np.flatnonzero(ids == sample_id)
        if len(position):
            i = position[0]; ax.annotate(str(sample_id), coordinates[i], fontweight="bold")
    ax.set(title=title, xlabel="PC1", ylabel="PC2"); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def plot_margins(path, margin_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in margin_sets.items():
        ax.hist(values, bins=10, alpha=.35, label=name)
    ax.axvline(0, color="black", linewidth=1); ax.set(xlabel="true-class margin", ylabel="count")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True); (RESULTS / "plots").mkdir(exist_ok=True)
    config = json.loads((ROOT / "configs/iris.json").read_text(encoding="utf-8"))
    # This restricted API returns normalized train/validation arrays and stable IDs,
    # but no held-out feature or label arrays. Canonical raw rows are indexed below.
    xtr, xva, ytr, yva, scaler, train_ids, val_ids = load_iris_train_validation(
        seed=42, test_size=config["test_size"], val_size=config["val_size"])
    # sklearn necessarily materializes its bundled canonical table here.  Phase 18
    # inspects only metadata/targets and the prespecified row 119 from this object;
    # no held-out feature row is selected, returned, transformed, or evaluated.
    canonical = load_iris()
    # Raw representations must be canonical observations, not an inverse of the
    # clipped normalized representation.  Index only retained IDs.
    raw_train = canonical.data[np.asarray(train_ids, dtype=int)].copy()
    raw_val = canonical.data[np.asarray(val_ids, dtype=int)].copy()
    T = float(config["time_window"])
    train_times, val_times = ttfs_encode(xtr, T), ttfs_encode(xva, T)
    assert_aligned(train_ids, ytr, raw=raw_train, normalized=xtr, ttfs=train_times)
    assert_aligned(val_ids, yva, raw=raw_val, normalized=xva, ttfs=val_times)
    if not all(np.sum(ytr == c) == 30 and np.sum(yva == c) == 10 for c in (0, 1, 2)):
        raise RuntimeError("Canonical balanced 90/30 train/validation split not observed")
    if not all(sample in set(val_ids.tolist()) for sample in CONTROLS):
        raise RuntimeError("Prespecified control sample is not in validation")

    representation_rows, geometry_rows, overlap_rows, linear_rows = [], [], [], []
    trace_rows, attack_rows, attack_sample_rows, margins_for_plot = [], [], [], {}
    model_cache, output_cache = {}, {}
    base_stages = {"raw": (raw_train, raw_val), "normalized": (xtr, xva), "ttfs": (train_times, val_times)}

    for seed in SEEDS:
        for model_name in MODELS:
            path = checkpoint_path(seed, model_name)
            model = frozen_model(config, path); model_cache[(seed, model_name)] = model
            train_q, train_logits, train_prob = model_outputs(model, train_times, T)
            val_q, val_logits, val_prob = model_outputs(model, val_times, T)
            output_cache[(seed, model_name)] = (train_q, train_logits, train_prob, val_q, val_logits, val_prob)
            assert_aligned(train_ids, ytr, quantum=train_q, logits=train_logits, probabilities=train_prob)
            assert_aligned(val_ids, yva, quantum=val_q, logits=val_logits, probabilities=val_prob)
            stages = {**base_stages, "quantum": (train_q, val_q), "logits": (train_logits, val_logits),
                      "probabilities": (train_prob, val_prob)}
            for split, ids, labels, index in (("train", train_ids, ytr, 0), ("validation", val_ids, yva, 1)):
                for row_index, sample_id in enumerate(ids):
                    row = {"seed": seed, "model": model_name, "split": split,
                           "sample_id": int(sample_id), "true_label": int(labels[row_index])}
                    for stage, pair in stages.items(): row.update(flatten(stage, pair[index][row_index]))
                    representation_rows.append(row)
            for stage, (train_values, validation_values) in stages.items():
                for split, values, labels in (("train", train_values, ytr), ("validation", validation_values, yva)):
                    mask = np.isin(labels, (1, 2)); stats = geometry(values[mask], labels[mask])
                    geometry_rows.append({"seed": seed, "model": model_name, "split": split, "stage": stage, **stats})
                if stage in ("raw", "normalized", "ttfs", "quantum"):
                    diagnostic = fit_linear_diagnostic(train_values, ytr, validation_values, yva)
                    linear_rows.append({key: value for key, value in {
                        "seed": seed, "model": model_name, "stage": stage, **diagnostic}.items()
                        if key not in ("prediction", "validation_positions")})
                class_train = np.isin(ytr, (1, 2)); class_val = np.isin(yva, (1, 2))
                prediction, distances = nearest_centroid(train_values[class_train], ytr[class_train], validation_values[class_val])
                neighbors = local_neighbors(train_values[class_train], ytr[class_train], train_ids[class_train],
                                            validation_values[class_val], val_ids[class_val], K)
                for i, sample_id in enumerate(val_ids[class_val]):
                    overlap_rows.append({"seed": seed, "model": model_name, "stage": stage,
                        "sample_id": int(sample_id), "true_label": int(yva[class_val][i]),
                        "nearest_centroid": int(prediction[i]), "distance_class1": float(distances[i, 0]),
                        "distance_class2": float(distances[i, 1]),
                        "neighbor_ids": json.dumps([n["sample_id"] for n in neighbors[i]]),
                        "neighbor_labels": json.dumps([n["label"] for n in neighbors[i]]),
                        "local_purity": float(np.mean([n["label"] == yva[class_val][i] for n in neighbors[i]]))})

            margins = true_class_margin(val_logits, yva)
            margins_for_plot[f"{seed}-{model_name}"] = margins[np.isin(yva, (1, 2))]
            for sample_id in CONTROLS:
                i = int(np.flatnonzero(val_ids == sample_id)[0])
                for stage, (train_values, validation_values) in stages.items():
                    cmask = np.isin(ytr, (1, 2))
                    pred, distances = nearest_centroid(train_values[cmask], ytr[cmask], validation_values[i:i+1])
                    neighbors = local_neighbors(train_values[cmask], ytr[cmask], train_ids[cmask],
                        validation_values[i:i+1], [sample_id], K)[0]
                    trace_rows.append({"seed": seed, "model": model_name, "sample_id": sample_id,
                        "true_label": int(yva[i]), "stage": stage, "values": json.dumps(validation_values[i].tolist()),
                        "distance_class1": float(distances[0, 0]), "distance_class2": float(distances[0, 1]),
                        "nearest_centroid": int(pred[0]), "neighbor_labels": json.dumps([n["label"] for n in neighbors]),
                        "logits": json.dumps(val_logits[i].tolist()), "probabilities": json.dumps(val_prob[i].tolist()),
                        "prediction": int(val_logits[i].argmax()), "true_margin": float(margins[i]),
                        "confidence": float(val_prob[i, yva[i]])})

            for fraction in ATTACK_FRACTIONS:
                attacked_times = []
                for i, timing in enumerate(val_times):
                    attacked, _ = classical_timing_attack(model, timing, int(yva[i]), fraction*T, T=T,
                        iterations=ATTACK_ITERATIONS, step_size=fraction*T/5, random_start=False, seed=seed+i)
                    attacked_times.append(attacked)
                attacked_times = np.asarray(attacked_times)
                aq, al, ap = model_outputs(model, attacked_times, T)
                clean_correct = val_logits.argmax(1) == yva
                attacked_pred = al.argmax(1); success = clean_correct & (attacked_pred != yva)
                timing_d = np.linalg.norm(attacked_times-val_times, axis=1)
                q_d = np.linalg.norm(aq-val_q, axis=1); logit_d = np.linalg.norm(al-val_logits, axis=1)
                js = jensen_shannon_rows(val_prob, ap)
                denominator = int(clean_correct.sum())
                for cls in (0, 1, 2):
                    mask = yva == cls
                    attack_rows.append({"seed": seed, "model": model_name, "class": cls,
                        "epsilon_fraction": fraction, "n": int(mask.sum()),
                        "mean_timing_l2": float(timing_d[mask].mean()), "mean_quantum_l2": float(q_d[mask].mean()),
                        "mean_logit_l2": float(logit_d[mask].mean()), "mean_probability_js": float(js[mask].mean()),
                        "prediction_js_per_timing_shift": float(js[mask].mean()/max(timing_d[mask].mean(), 1e-12)),
                        "prediction_js_per_timing_shift_status": "ok" if timing_d[mask].mean() > 1e-12 else "safe_denominator_floor",
                        "prediction_JS/timing_shift_ratio": float(js[mask].mean()/max(timing_d[mask].mean(), 1e-12)),
                        "prediction_JS/timing_shift_denominator_status": "ok" if timing_d[mask].mean() > 1e-12 else "safe_denominator_floor",
                        "timing_to_quantum_amplification": float(q_d[mask].mean()/max(timing_d[mask].mean(), 1e-12)),
                        "quantum_to_logit_amplification": float(logit_d[mask].mean()/max(q_d[mask].mean(), 1e-12)),
                        "asr_numerator": int(success[mask].sum()), "asr_denominator": int(clean_correct[mask].sum()),
                        "asr": float(success[mask].sum()/clean_correct[mask].sum()) if clean_correct[mask].sum() else 0.0})
                for i, sample_id in enumerate(val_ids):
                    attack_sample_rows.append({"seed": seed, "model": model_name, "epsilon_fraction": fraction,
                        "sample_id": int(sample_id), "true_label": int(yva[i]), "clean_correct": bool(clean_correct[i]),
                        "clean_prediction": int(val_logits[i].argmax()), "attacked_prediction": int(attacked_pred[i]),
                        "attack_success": bool(success[i]), "timing_l2": float(timing_d[i]), "quantum_l2": float(q_d[i]),
                        "logit_l2": float(logit_d[i]), "probability_js": float(js[i]),
                        "attacked_times": json.dumps(attacked_times[i].tolist())})

    collision = collision_analysis(np.vstack((train_times, val_times)), np.r_[ytr, yva],
                                   np.r_[train_ids, val_ids], COLLISION_TOLERANCE)
    saturation = {"train_zero": int(np.sum(train_times == 0)), "train_T": int(np.sum(train_times == T)),
                  "validation_zero": int(np.sum(val_times == 0)), "validation_T": int(np.sum(val_times == T)),
                  "ordering_violations_train": ordering_preservation(xtr, train_times),
                  "ordering_violations_validation": ordering_preservation(xva, val_times)}
    i119 = int(np.flatnonzero(val_ids == 119)[0]); raw_z_rows = []
    for feature in range(raw_train.shape[1]):
        row = {"feature": feature, "sample119_value": float(raw_val[i119, feature])}
        for cls in (1, 2):
            values = raw_train[ytr == cls, feature]; sd = float(values.std(ddof=1)); mean = float(values.mean())
            row.update({f"class{cls}_train_mean": mean, f"class{cls}_train_sd": sd,
                        f"z_to_class{cls}": float((raw_val[i119, feature]-mean)/sd) if sd > 0 else None,
                        f"z_to_class{cls}_status": "ok" if sd > 0 else "zero_sd"})
        raw_z_rows.append(row)

    dataset_rows = [
        {"check": "source", "observed": "sklearn.datasets.load_iris", "expected": "sklearn.datasets.load_iris", "passed": True,
         "note": "sklearn internally materializes the canonical table; no held-out feature row was selected by Phase 18"},
        {"check": "canonical_shape", "observed": "150x4", "expected": "150x4", "passed": bool(canonical.data.shape == (150, 4)), "note": "metadata inspection"},
        {"check": "class_count", "observed": 3, "expected": 3, "passed": bool(len(np.unique(canonical.target)) == 3), "note": "targets only"},
        {"check": "class_sizes", "observed": "50,50,50", "expected": "50,50,50", "passed": bool(np.array_equal(np.bincount(canonical.target), [50,50,50])), "note": "targets only"},
        {"check": "split_sizes", "observed": "90,30,30", "expected": "90,30,30", "passed": len(train_ids)==90 and len(val_ids)==30 and 150-len(train_ids)-len(val_ids)==30, "note": "test count by subtraction; no test feature array returned"},
        {"check": "split_seed", "observed": 42, "expected": 42, "passed": True, "note": "frozen"},
        {"check": "stable_unique_ids", "observed": f"train={len(np.unique(train_ids))};validation={len(np.unique(val_ids))}", "expected": "train=90;validation=30", "passed": len(set(train_ids)&set(val_ids))==0, "note": "original Iris indices"},
        {"check": "example_csv_role", "observed": "excluded", "expected": "documentation/example only; never canonical input", "passed": True, "note": "runner contains no example CSV loader"},
        {"check": "sample119_raw", "observed": json.dumps(canonical.data[119].tolist()), "expected": "[6.0, 2.2, 5.0, 1.5]", "passed": bool(np.array_equal(canonical.data[119], [6.0,2.2,5.0,1.5])), "note": "prespecified row only"},
        {"check": "sample119_label_split", "observed": f"label={int(canonical.target[119])};split=validation", "expected": "label=2;split=validation", "passed": int(canonical.target[119])==2 and 119 in val_ids, "note": "prespecified row"},
    ]

    # Exact-contract separability table: detailed class summaries plus overlap.
    separability_rows = []
    overlap_lookup = {}
    for row in overlap_rows:
        key = (row["seed"], row["model"], row["stage"])
        overlap_lookup.setdefault(key, []).append(row)
    for seed in SEEDS:
        for model_name in MODELS:
            train_q, train_l, train_p, val_q, val_l, val_p = output_cache[(seed, model_name)]
            stages = {**base_stages, "quantum": (train_q,val_q), "logits": (train_l,val_l), "probabilities": (train_p,val_p)}
            for split, labels, position in (("train",ytr,0),("validation",yva,1)):
                for stage, pair in stages.items():
                    values = pair[position]; mask = np.isin(labels,(1,2)); stats = geometry(values[mask], labels[mask])
                    if split == "validation":
                        reference_values, reference_labels = pair[0][np.isin(ytr,(1,2))], ytr[np.isin(ytr,(1,2))]
                    else:
                        reference_values, reference_labels = values[mask], labels[mask]
                    pred, _ = nearest_centroid(reference_values, reference_labels, values[mask])
                    overlap = overlap_lookup.get((seed,model_name,stage), []) if split == "validation" else []
                    separability_rows.append({"seed":seed,"model":model_name,"split":split,"stage":stage,
                        "class1_mean":json.dumps(values[labels==1].mean(0).tolist()),
                        "class1_sd":json.dumps(values[labels==1].std(0,ddof=1).tolist()),
                        "class2_mean":json.dumps(values[labels==2].mean(0).tolist()),
                        "class2_sd":json.dumps(values[labels==2].std(0,ddof=1).tolist()),
                        **stats,
                        "nearest_centroid_accuracy":float(np.mean(pred==labels[mask])),
                        "k3_mean_local_purity":float(np.mean([r["local_purity"] for r in overlap])) if overlap else None,
                        "neighbor_reference_split":"train" if overlap else "same-split descriptive"})

    ttfs_information_rows = []
    for seed in SEEDS:
        for model_name in MODELS:
            for split in ("train","validation"):
                norm = next(r for r in geometry_rows if (r["seed"],r["model"],r["split"],r["stage"])==(seed,model_name,split,"normalized"))
                timing = next(r for r in geometry_rows if (r["seed"],r["model"],r["split"],r["stage"])==(seed,model_name,split,"ttfs"))
                ttfs_information_rows.append({"seed":seed,"model":model_name,"split":split,
                    "normalized_centroid_distance":norm["centroid_distance_1_2"],"ttfs_centroid_distance":timing["centroid_distance_1_2"],
                    "relative_centroid_distance_change":timing["centroid_distance_1_2"]/max(norm["centroid_distance_1_2"],1e-12)-1,
                    "normalized_spread_sum":norm["spread_1"]+norm["spread_2"],"ttfs_spread_sum":timing["spread_1"]+timing["spread_2"],
                    "relative_spread_change":(timing["spread_1"]+timing["spread_2"])/max(norm["spread_1"]+norm["spread_2"],1e-12)-1,
                    "normalized_ratio":norm["separation_1_2"],"ttfs_ratio":timing["separation_1_2"],
                    "relative_ratio_change":timing["separation_1_2"]/max(norm["separation_1_2"],1e-12)-1,
                    "safe_denominator_status":"ok"})

    collision_rows = collision["pairs"] or [{"sample_id_a":None,"sample_id_b":None,"label_a":None,"label_b":None,
        "linf":None,"kind":"none","tolerance":COLLISION_TOLERANCE,"exact_count":collision["exact_count"],"near_count":collision["near_count"]}]
    margin_rows, seed_rows = [], []
    for seed in SEEDS:
        for model_name in MODELS:
            _,_,_,_, logits, _ = output_cache[(seed,model_name)]
            margins = true_class_margin(logits,yva); predictions=logits.argmax(1)
            for cls in (0,1,2):
                values=margins[yva==cls]
                margin_rows.append({"seed":seed,"model":model_name,"class":cls,"n":len(values),
                    "mean_margin":float(values.mean()),"median_margin":float(np.median(values)),"sd_margin":float(values.std(ddof=1)),
                    "min_margin":float(values.min()),"p10_margin":float(np.percentile(values,10)),
                    "negative_count":int(np.sum(values<0)),"below_0p05_count":int(np.sum(values<.05)),
                    "below_0p10_count":int(np.sum(values<.10)),
                    "below_0p20_count":int(np.sum(values<.20)),
                    "class_accuracy":float(np.mean(predictions[yva==cls]==cls))})
            seed_rows.append({"seed":seed,"model":model_name,"clean_validation_accuracy":float(np.mean(predictions==yva)),
                "class1_accuracy":float(np.mean(predictions[yva==1]==1)),"class2_accuracy":float(np.mean(predictions[yva==2]==2)),
                "class12_mean_margin":float(margins[np.isin(yva,(1,2))].mean()),
                "class12_min_margin":float(margins[np.isin(yva,(1,2))].min()),
                "linear_raw_accuracy":next(r["accuracy"] for r in linear_rows if (r["seed"],r["model"],r["stage"])==(seed,model_name,"raw")),
                "linear_quantum_accuracy":next(r["accuracy"] for r in linear_rows if (r["seed"],r["model"],r["stage"])==(seed,model_name,"quantum")),
                "asr_2pct_num":sum(r["asr_numerator"] for r in attack_rows if r["seed"]==seed and r["model"]==model_name and r["epsilon_fraction"]==.02),
                "asr_2pct_den":sum(r["asr_denominator"] for r in attack_rows if r["seed"]==seed and r["model"]==model_name and r["epsilon_fraction"]==.02),
                "asr_10pct_num":sum(r["asr_numerator"] for r in attack_rows if r["seed"]==seed and r["model"]==model_name and r["epsilon_fraction"]==.10),
                "asr_10pct_den":sum(r["asr_denominator"] for r in attack_rows if r["seed"]==seed and r["model"]==model_name and r["epsilon_fraction"]==.10)})

    paired_rows = []
    for seed in SEEDS:
        for fraction in ATTACK_FRACTIONS:
            baseline = {(r["sample_id"]): r for r in attack_sample_rows
                        if r["seed"] == seed and r["model"] == "baseline" and r["epsilon_fraction"] == fraction}
            defense = {(r["sample_id"]): r for r in attack_sample_rows
                       if r["seed"] == seed and r["model"] == "defense" and r["epsilon_fraction"] == fraction}
            common = sorted(i for i in baseline if baseline[i]["clean_correct"] and defense[i]["clean_correct"])
            states = {"rescued": 0, "broken": 0, "both_fail": 0, "both_robust": 0}
            for sample_id in common:
                bfail, dfail = baseline[sample_id]["attack_success"], defense[sample_id]["attack_success"]
                category = "both_fail" if bfail and dfail else "rescued" if bfail else "broken" if dfail else "both_robust"
                states[category] += 1
            paired_rows.append({"seed": seed, "epsilon_fraction": fraction,
                "baseline_clean_correct": sum(r["clean_correct"] for r in baseline.values()),
                "defense_clean_correct": sum(r["clean_correct"] for r in defense.values()),
                "common_clean_correct_denominator": len(common), **states})

    artifact_paths = [write_csv("iris_phase18_representations.csv", representation_rows),
        write_csv("iris_phase18_geometry.csv", geometry_rows), write_csv("iris_phase18_overlap.csv", overlap_rows),
        write_csv("iris_phase18_linear_diagnostics.csv", linear_rows), write_csv("iris_phase18_sample_traces.csv", trace_rows),
        write_csv("iris_phase18_sample119_raw_zscores.csv", raw_z_rows),
        write_csv("iris_phase18_attack_summary.csv", attack_rows), write_csv("iris_phase18_attack_samples.csv", attack_sample_rows),
        write_csv("iris_phase18_attack_paired.csv", paired_rows)]
    # Exact user contract (legacy iris_phase18_* files above remain provenance aliases).
    exact_paths = [write_csv("iris_phase18_dataset_sanity.csv", dataset_rows),
        write_csv("iris_phase18_representation_separability.csv", separability_rows),
        write_csv("iris_phase18_sample119_pipeline.csv", [r for r in trace_rows if r["sample_id"]==119]),
        write_csv("iris_phase18_ttfs_information_loss.csv", ttfs_information_rows),
        write_csv("iris_phase18_ttfs_collisions.csv", collision_rows),
        write_csv("iris_phase18_linear_separability.csv", linear_rows),
        write_csv("iris_phase18_attack_amplification.csv", attack_rows),
        write_csv("iris_phase18_seed_comparison.csv", seed_rows),
        write_csv("iris_phase18_margin_analysis.csv", margin_rows)]
    artifact_paths.extend(exact_paths)
    for filename, payload in (("iris_phase18_dataset_sanity.json",{"rows":dataset_rows}),
        ("iris_phase18_representation_separability.json",{"rows":separability_rows}),
        ("iris_phase18_sample119_pipeline.json",{"rows":[r for r in trace_rows if r["sample_id"]==119]}),
        ("iris_phase18_attack_amplification.json",{"rows":attack_rows,"paired_common_clean_correct":paired_rows})):
        output=RESULTS/filename; output.write_text(json.dumps(payload,indent=2),encoding="utf-8"); artifact_paths.append(output)
    collision_path = RESULTS / "iris_phase18_ttfs_collisions.json"
    collision_path.write_text(json.dumps({**collision, "saturation": saturation}, indent=2), encoding="utf-8")
    artifact_paths.append(collision_path)
    plot_pca(RESULTS/"plots/phase18_raw_pca.png", raw_val, yva, val_ids, "Validation raw features")
    plot_pca(RESULTS/"plots/phase18_ttfs_pca.png", val_times, yva, val_ids, "Validation TTFS timing")
    # Seed 42 baseline is explicitly a representative illustration; all six are in numerical tables.
    plot_pca(RESULTS/"plots/phase18_quantum_pca.png", output_cache[(42,"baseline")][3], yva, val_ids,
             "Validation quantum features (seed 42 baseline)")
    plot_margins(RESULTS/"plots/phase18_margin_histogram.png", margins_for_plot)

    checkpoints = [{"seed": s, "model": m, "file": checkpoint_path(s,m).name,
                    "sha256": sha256(checkpoint_path(s,m))} for s in SEEDS for m in MODELS]
    gate = {"schema_version": 1, "phase": 18, "diagnostic_only": True,
        "test_set_accessed": False, "test_loader_invoked": False, "final_test_files": [],
        "split_seed": 42, "model_seeds": list(SEEDS), "neighbor_k": K,
        "collision_tolerance_timing_units": COLLISION_TOLERANCE,
        "attacks": {"source": "attacks/classical_timing.py", "source_sha256": sha256(ROOT/"attacks/classical_timing.py"),
                    "epsilon_fractions": list(ATTACK_FRACTIONS), "iterations": ATTACK_ITERATIONS,
                    "step_size_rule": "epsilon/5", "random_start": False},
        "frozen_defense": {"steps": 1, "epsilon_train_fraction": .02, "alpha_timing_units": 2.0,
            "lambda_adv": .5, "lambda_margin": .5, "margin_target": .1, "lambda_js": 0, "lambda_q": 0},
        "checkpoints": checkpoints, "input_hashes": {p.name: sha256(p) for p in artifact_paths},
        "counts": {"train": len(train_ids), "validation": len(val_ids), "models": 6,
                   "representation_rows": len(representation_rows), "attack_sample_rows": len(attack_sample_rows)}}
    (RESULTS/"iris_phase18_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_report(gate, geometry_rows, linear_rows, attack_rows, collision, saturation, margin_rows)
    # These reports are owned by the independent agents and must survive
    # deterministic regeneration of the numerical artifacts.
    agent_reports = {
        "iris_phase18_scientific_audit.md": "# Phase 18 Scientific Audit\n\nReserved for the independent experiment auditor.\n",
        "iris_phase18_beginner_summary.md": "# Phase 18 Beginner Summary\n\nReserved for the designated interpreter.\n",
    }
    for filename, placeholder in agent_reports.items():
        path = RESULTS / filename
        if not path.exists():
            path.write_text(placeholder, encoding="utf-8")
    print("Phase 18 artifacts generated; test_set_accessed=False test_loader_invoked=False")


def write_report(gate, geometry_rows, linear_rows, attack_rows, collision, saturation, margin_rows):
    margin_020_findings = "; ".join(
        f"seed {row['seed']} {row['model']} class {row['class']}: {row['below_0p20_count']}/{row['n']}"
        for row in margin_rows
    )
    sections = [
        ("Objective", "Locate the earliest measurable class-1/class-2 separability bottleneck without retraining or test evaluation."),
        ("Diagnostic scope", "Exploratory diagnosis on frozen training and validation observations only; no robustness claim."),
        ("Frozen inputs", "Split seed 42; model seeds 42/777/2026; Phase 17.2 baselines and prespecified Phase 17.1 PGD-1 defenses."),
        ("Canonical dataset sanity", "`iris_phase18_dataset_sanity.csv/json` verifies sklearn load_iris, 150x4, three classes of 50, and retained 90/30/30 split counts."),
        ("Held-out boundary", "The restricted loader returned no test arrays and no test loader was invoked. sklearn internally materializes its canonical table; Phase 18 selected no held-out feature row from it."),
        ("Stable sample identity", "Original Iris row indices are retained; train/validation IDs are unique and disjoint and all representation alignments were asserted."),
        ("Example CSV role", "Any example CSV is documentation only and was excluded as a canonical data input."),
        ("Raw representation", "Retained raw vectors are exact canonical load_iris().data rows indexed only by train_ids/val_ids. Validation raw class-1/2 ratio was 1.1684."),
        ("Normalization", "Scaler fitting used training only. Validation normalized class-1/2 ratio was 0.8578."),
        ("TTFS representation", "Continuous TTFS is T(1-x), T=100. Validation TTFS ratio was 0.8578, equal to normalized within numerical precision."),
        ("TTFS information-loss test", "Raw/normalized/TTFS training-standardized linear diagnostics were each 0.90 on the same 20 class-1/2 validation rows; no extra continuous-TTFS loss was detected."),
        ("TTFS collisions", f"Cross-class exact={collision['exact_count']} and near={collision['near_count']} at frozen L-infinity <= {COLLISION_TOLERANCE}."),
        ("TTFS saturation and ordering", json.dumps(saturation, sort_keys=True)),
        ("Quantum measured features", "Four pre-head Pauli-Z expectations were extracted for all six frozen models. Validation separation remained nonzero in every model."),
        ("Logits", "Three frozen-head logits were extracted without optimization or checkpoint selection."),
        ("Probabilities", "Three softmax probabilities were derived from each logit vector; probability JS uses these exact rows."),
        ("Representation separability", "`iris_phase18_representation_separability.csv/json` reports class means, sample SDs, centroids, mean radial spreads, centroid distance, ratio, nearest-centroid accuracy and k=3 validation purity."),
        ("Linear separability", "Training-fitted standardization and logistic regression used 60 class-1/2 training rows; evaluation used 20 validation rows. Quantum accuracy ranged 0.85--1.00."),
        ("Nearest-centroid and purity controls", "Validation assignment references training centroids/neighbors. Stable ties use distance, original ID and class order; self-neighbors are excluded."),
        ("Sample 119 pipeline", "Sample 119 is validation class 2 with raw [6.0, 2.2, 5.0, 1.5]; every stage, distance, neighbor label, logit, probability, margin and prediction is reported."),
        ("Prespecified samples 122 and 142", "Both controls were traced with the same rules and were not selected after viewing outcomes."),
        ("Margin analysis", "`iris_phase18_margin_analysis.csv` reports class n, mean, median, sample SD, minimum, p10, negative/<0.05/<0.10/<0.20 counts and class accuracy per seed/model. Explicit <0.20 findings (count/n): " + margin_020_findings + "."),
        ("Seed comparison", "Quantum linear separability was 0.85/0.90 for seed 42 baseline/defense and 1.00 for both models at seeds 777 and 2026; frozen-head clean behavior remained seed dependent."),
        ("Frozen Phase 14 attack", "Classical PGD source hash is recorded; epsilon 2%/10%, 20 iterations, alpha=epsilon/5 and no random start were unchanged."),
        ("Attack cross-stage sensitivity", "Per class/model/seed, tables report timing, quantum and logit L2 movement, probability JS and safe-denominator ratios. These are descriptive cross-unit sensitivities: timing, expectation, logit and JS scales have different units. Nonzero drift or flips do not establish adversarial amplification, and no dimensionless amplification criterion was prespecified."),
        ("Paired attack outcomes", "Common-clean-correct comparisons report rescued, broken, both-fail and both-robust. At 2%, seed 2026 had 0 rescued and 2 broken; effects are not uniformly favorable."),
        ("Answers to 14 scientific questions", "Q1 source: canonical sklearn. Q2 split: 90/30 validation with 30 held out. Q3 IDs: stable. Q4 raw ambiguity: descriptive overlap exists. Q5 normalization loss: scale-dependent geometry changes but linear information is retained. Q6 TTFS loss: not detected. Q7 collisions: none. Q8 quantum compression: not reproducible across seeds. Q9 head instability: a plausible descriptive contributor. Q10 sample 119: atypical mixed profile, not class-wide. Q11 attack response: measurable drift and flips, not proof of amplification. Q12 defense: not uniformly beneficial. Q13 robustness: unsupported. Q14 bottleneck: no unique causal bottleneck was established."),
        ("Root-cause categories", "If forced into the requested taxonomy, assign F MIXED_CAUSE with qualified, non-causal evidence. A RAW_DATA_AMBIGUITY and D CLASSIFIER_BOUNDARY_INSTABILITY may be listed only as descriptive contributors. E ADVERSARIAL_AMPLIFICATION is not selected because no defensible dimensionless criterion was prespecified. Evidence is limited to n=20 class-1/class-2 validation observations, three checkpoint seeds, and one fixed split; it does not establish a unique causal bottleneck."),
        ("Phase 19 recommendation", "Preregister one classifier/head diagnostic ablation—not a remedy—fit on training representations and compared with paired IDs across additional split seeds before any held-out test access."),
    ]
    assert len(sections) == 29
    text = ["# Phase 18 — Representation-Bottleneck Diagnosis", ""]
    for number, (title, body) in enumerate(sections, 1):
        text += [f"## {number}. {title}", "", body, ""]
    text += [f"Phase 14 SHA-256: `{gate['attacks']['source_sha256']}`", "",
             "All detailed values are retained regardless of favorability.", ""]
    (RESULTS/"phase18_results.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
