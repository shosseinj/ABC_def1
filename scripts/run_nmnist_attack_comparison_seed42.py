"""Validation-only seed-42 timing-attack comparison for the selected N-MNIST QSNN.

This script intentionally never instantiates the official N-MNIST test partition.
It is an exploratory single-seed comparison, not evidence of general robustness.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.nmnist_qsnn import NMNISTCudaQSNN


SEED = 42
CHECKPOINT = ROOT / "checkpoints" / "nmnist_qsnn_ablation" / "T4_S8_Q12_B6_best.pt"
CHECKPOINT_SHA256 = "c68065369bdad33af3c98b9b0a9045722f05cb95f00e2649479990b8e8318e75"
SPLIT = ROOT / "results" / "nmnist_snn_multiseed_split.json"
SPLIT_SHA256 = "a12176ce117ab9a85dd29d277901f8617ad2b5427dd3f976dbba436942f70198"
OUTPUT_JSON = ROOT / "results" / "nmnist_attack_comparison_seed42.json"
SUMMARY_CSV = ROOT / "results" / "nmnist_attack_comparison_seed42.csv"
SAMPLE_CSV = ROOT / "results" / "nmnist_attack_samples_seed42.csv"
REPORT = ROOT / "results" / "nmnist_attack_comparison_seed42_report.md"
EPSILON_FRACTIONS = (0.02, 0.05, 0.10)
ATTACKS = ("PGD", "TEMP-DRIFT")
TEMP_INITIAL = 600
TEMP_GENERATIONS = 4
TEMP_PER_GENERATION = 250
TEMP_CANDIDATES = TEMP_INITIAL + TEMP_GENERATIONS * TEMP_PER_GENERATION
PGD_STEPS = 20
CUDA_CHUNK = 64


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_timestamps(candidate, clean, epsilon, t0=None, tlast=None):
    """Project onto clean +/- epsilon, the frozen clean window, and monotone order."""
    clean = np.asarray(clean, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if clean.ndim != 1 or candidate.shape != clean.shape or not len(clean):
        raise ValueError("Timestamp arrays must be nonempty one-dimensional arrays of equal shape.")
    if np.any(np.diff(clean) < 0) or epsilon < 0:
        raise ValueError("Clean timestamps must be ordered and epsilon nonnegative.")
    t0 = float(clean[0] if t0 is None else t0)
    tlast = float(clean[-1] if tlast is None else tlast)
    lower = np.maximum(clean - float(epsilon), t0)
    upper = np.minimum(clean + float(epsilon), tlast)
    projected = np.clip(candidate, lower, upper)
    projected = np.maximum.accumulate(projected)
    return np.minimum(projected, upper)


def exact_feasibility(clean, attacked, epsilon, t0=None, tlast=None, atol=1e-7):
    clean = np.asarray(clean, dtype=np.float64)
    attacked = np.asarray(attacked, dtype=np.float64)
    if attacked.shape != clean.shape or not len(clean):
        return False
    t0 = float(clean[0] if t0 is None else t0)
    tlast = float(clean[-1] if tlast is None else tlast)
    return bool(
        np.all(np.isfinite(attacked))
        and np.all(np.diff(attacked) >= -atol)
        and np.all(attacked >= t0 - atol)
        and np.all(attacked <= tlast + atol)
        and np.all(np.abs(attacked - clean) <= epsilon + atol)
    )


def replace_timestamps(events, timestamps):
    """Copy a native event array while changing timing and nothing else."""
    timestamps = np.asarray(timestamps)
    if len(events) != len(timestamps):
        raise ValueError("Event/timestamp length mismatch.")
    attacked = events.copy()
    attacked["t"] = timestamps.astype(attacked.dtype["t"], copy=False)
    return attacked


def representation_numpy(events, timestamps=None, bins=4, grid=(2, 2), sensor=(34, 34)):
    """Exact hard representation used during training, with optional float timestamps."""
    if not len(events):
        return np.zeros((bins, grid[0] * grid[1] * 2), dtype=np.float32)
    t = np.asarray(events["t"] if timestamps is None else timestamps, dtype=np.float64)
    duration = max(float(events["t"][-1]) - float(events["t"][0]) + 1.0, 1.0)
    time_bin = np.minimum(((t - float(events["t"][0])) * bins // duration).astype(int), bins - 1)
    time_bin = np.maximum(time_bin, 0)
    x, y, p = (np.asarray(events[name], dtype=np.int64) for name in ("x", "y", "p"))
    channel = (((y * grid[0]) // sensor[1]) * grid[1] + (x * grid[1]) // sensor[0]) * 2 + p
    output = np.zeros((bins, grid[0] * grid[1] * 2), dtype=np.float32)
    np.add.at(output, (time_bin, channel), 1.0)
    maximum = float(output.max())
    return (np.log1p(output) / np.log1p(maximum)).astype(np.float32) if maximum else output


def _represent_torch(events, timestamps, straight_through=False, bins=4):
    """Hard bins in the forward pass; triangular soft-bin gradients when requested."""
    if timestamps.ndim == 1:
        timestamps = timestamps.unsqueeze(0)
    device, dtype = timestamps.device, timestamps.dtype
    n, batch = timestamps.shape[1], timestamps.shape[0]
    clean_t = torch.as_tensor(np.asarray(events["t"], dtype=np.float32), device=device, dtype=dtype)
    duration = max(float(events["t"][-1]) - float(events["t"][0]) + 1.0, 1.0)
    u = (timestamps - clean_t[0]) * bins / duration
    hard_bin = torch.floor(u).long().clamp(0, bins - 1)
    channel_np = (((np.asarray(events["y"], dtype=np.int64) * 2) // 34) * 2
                  + (np.asarray(events["x"], dtype=np.int64) * 2) // 34) * 2 + np.asarray(events["p"], dtype=np.int64)
    channel = torch.as_tensor(channel_np, device=device).unsqueeze(0).expand(batch, n)
    flat = torch.zeros(batch, bins * 8, device=device, dtype=dtype)
    flat.scatter_add_(1, hard_bin * 8 + channel, torch.ones_like(timestamps))
    hard = flat.reshape(batch, bins, 8)
    if straight_through:
        centers = torch.arange(bins, device=device, dtype=dtype) + 0.5
        weights = (1.0 - torch.abs(u.unsqueeze(-1) - centers)).clamp_min(0.0)
        soft = torch.zeros(batch, bins, 8, device=device, dtype=dtype)
        soft.scatter_add_(2, channel.unsqueeze(1).expand(batch, bins, n), weights.transpose(1, 2))
        counts = soft + (hard - soft).detach()
    else:
        counts = hard
    maximum = counts.amax(dim=(1, 2), keepdim=True)
    return torch.where(maximum > 0, torch.log1p(counts) / torch.log1p(maximum), counts)


def select_stratified_clean_correct(validation_ids, labels, predictions, per_class=10):
    """Take the first clean-correct IDs per class in frozen validation order."""
    chosen = []
    counts = np.zeros(10, dtype=int)
    for sample_id, label, prediction in zip(validation_ids, labels, predictions):
        label = int(label)
        if int(prediction) == label and counts[label] < per_class:
            chosen.append(int(sample_id))
            counts[label] += 1
    if not np.all(counts == per_class):
        raise RuntimeError(f"Insufficient clean-correct validation samples by class: {counts.tolist()}")
    return chosen


def paired_overlap(pgd_success, temp_success):
    pgd, temp = np.asarray(pgd_success, dtype=bool), np.asarray(temp_success, dtype=bool)
    if pgd.shape != temp.shape:
        raise ValueError("Paired success vectors differ in shape.")
    return {
        "rescued": int(np.sum(~pgd & temp)),
        "broken": int(np.sum(pgd & ~temp)),
        "both_fail": int(np.sum(~pgd & ~temp)),
        "both_robust": int(np.sum(~pgd & ~temp)),
        "both_success": int(np.sum(pgd & temp)),
    }


def comparison_verdict(summary_rows):
    """Single-seed descriptive rule driven primarily by paired-denominator ASR."""
    by_key = {(row["epsilon_fraction"], row["attack"]): row for row in summary_rows}
    differences = [by_key[(epsilon, "TEMP-DRIFT")]["asr"] - by_key[(epsilon, "PGD")]["asr"]
                   for epsilon in EPSILON_FRACTIONS]
    if all(delta >= 0.0 for delta in differences) and any(delta >= 0.05 for delta in differences):
        return "TEMP-DRIFT: STRONGER"
    if all(delta >= -0.05 for delta in differences) and any(delta > 0.0 for delta in differences):
        return "TEMP-DRIFT: COMPETITIVE"
    return "TEMP-DRIFT: NOT BENEFICIAL"


class EvaluationCounter:
    def __init__(self):
        self.forward_calls = 0
        self.sample_equivalent_evaluations = 0
        self.state_forward_calls = 0
        self.state_sample_equivalent_evaluations = 0

    def classifier(self, model, features):
        self.forward_calls += 1
        self.sample_equivalent_evaluations += int(len(features))
        return model(features)

    def state(self, model, features):
        self.state_forward_calls += 1
        self.state_sample_equivalent_evaluations += int(len(features))
        return final_state(model, features)

    def snapshot(self):
        return {
            "classifier_forward_calls": int(self.forward_calls),
            "classifier_sample_equivalent_evaluations": int(self.sample_equivalent_evaluations),
            "state_forward_calls": int(self.state_forward_calls),
            "state_sample_equivalent_evaluations": int(self.state_sample_equivalent_evaluations),
        }


def final_state(model, event_channels):
    """Return the frozen circuit's final pure premeasurement state."""
    state = torch.zeros(len(event_channels), 2 ** model.n_qubits, dtype=torch.complex64,
                        device=event_channels.device)
    state[:, 0] = 1.0
    for block in range(model.n_blocks):
        temporal = (block,) if block < model.temporal_bins else ()
        for temporal_bin in temporal:
            for channel in range(model.spatial_polarity_channels):
                state = model._ry(state, math.pi * event_channels[:, temporal_bin, channel],
                                  channel % model.n_qubits)
        for wire in range(model.n_qubits):
            state = model._ry(state, model.weights[block, wire, 0].expand(len(event_channels)), wire)
            state = model._rz(state, model.weights[block, wire, 1].expand(len(event_channels)), wire)
        for wire in range(model.n_qubits):
            state = state[:, getattr(model, f"cnot_{wire}")]
    return state


def _margin(logits, label):
    alternatives = logits.clone()
    alternatives[:, int(label)] = -torch.inf
    return logits[:, int(label)] - alternatives.max(1).values


def _chunked_evaluate(model, events, candidates, clean_state, label, counter, chunk_size):
    losses, margins, predictions, fidelities = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(candidates), chunk_size):
            timestamps = torch.as_tensor(candidates[start:start + chunk_size], device=clean_state.device,
                                         dtype=torch.float32)
            features = _represent_torch(events, timestamps)
            logits = counter.classifier(model, features)
            states = counter.state(model, features)
            losses.extend(F.cross_entropy(logits, torch.full((len(logits),), int(label), device=logits.device),
                                          reduction="none").cpu().tolist())
            margins.extend(_margin(logits, label).cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            overlap = torch.sum(clean_state.conj() * states, dim=1).abs().square().clamp(0, 1)
            fidelities.extend(overlap.cpu().tolist())
    return (np.asarray(losses), np.asarray(margins), np.asarray(predictions), np.asarray(fidelities))


def pgd_attack(model, events, label, epsilon, counter):
    clean = np.asarray(events["t"], dtype=np.float64)
    t0, tlast = float(clean[0]), float(clean[-1])
    candidate = torch.as_tensor(clean, device=next(model.parameters()).device, dtype=torch.float32)
    best, best_loss = clean.copy(), -math.inf
    candidate_evaluations = 0
    for iteration in range(PGD_STEPS + 1):
        hard = _represent_torch(events, candidate)
        with torch.no_grad():
            logits = counter.classifier(model, hard)
            hard_loss = float(F.cross_entropy(logits, torch.tensor([label], device=logits.device)))
        candidate_evaluations += 1
        if hard_loss > best_loss:
            best_loss, best = hard_loss, candidate.detach().cpu().numpy().copy()
        if iteration == PGD_STEPS:
            break
        variable = candidate.detach().requires_grad_(True)
        logits = counter.classifier(model, _represent_torch(events, variable, straight_through=True))
        loss = F.cross_entropy(logits, torch.tensor([label], device=logits.device))
        gradient, = torch.autograd.grad(loss, variable)
        proposal = variable.detach() + (epsilon / 5.0) * gradient.sign()
        candidate = torch.as_tensor(project_timestamps(proposal.cpu().numpy(), clean, epsilon, t0, tlast),
                                    device=proposal.device, dtype=torch.float32)
    return best, {"candidate_evaluations": candidate_evaluations, "surrogate_gradient_forwards": PGD_STEPS}


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values)) if len(values) > 1 else 0.0
    return ranks


def temp_attack(model, events, label, epsilon, clean_state, counter, chunk_size, seed):
    """Derivative-free boundary-guided TEMP search; no gradient API is called."""
    clean = np.asarray(events["t"], dtype=np.float64)
    t0, tlast = float(clean[0]), float(clean[-1])
    duration = tlast - t0 + 1.0
    rng = np.random.default_rng(seed)
    scaled = (clean - t0) * 4.0 / duration
    boundary_distance = np.abs(scaled - np.round(scaled))
    sensitivity = 1.0 / (boundary_distance + 0.05)
    sensitivity /= sensitivity.sum()

    delta = rng.uniform(-epsilon, epsilon, size=(TEMP_INITIAL, len(clean)))
    # Half the initialization explicitly prioritizes timestamps nearest hard-bin boundaries.
    priority = rng.choice(len(clean), size=(TEMP_INITIAL // 2, max(1, min(8, len(clean)))), p=sensitivity)
    delta[TEMP_INITIAL // 2:] = 0.0
    rows = np.arange(TEMP_INITIAL // 2)[:, None]
    delta[TEMP_INITIAL // 2 + rows, priority] = rng.choice((-epsilon, epsilon), size=priority.shape)
    pool = np.stack([project_timestamps(clean + row, clean, epsilon, t0, tlast) for row in delta])
    losses, margins, predictions, fidelities = _chunked_evaluate(
        model, events, pool, clean_state, label, counter, chunk_size
    )

    def score_population(current_margins, current_fidelities):
        drift_rank = _ranks(1.0 - current_fidelities)
        margin_rank = _ranks(-current_margins)
        boundary_bonus = (current_margins < 0.0).astype(float)
        return boundary_bonus + 0.35 * drift_rank + 0.65 * margin_rank

    scores = score_population(margins, fidelities)
    for generation, scale in enumerate((0.50, 0.30, 0.18, 0.10)):
        successful = predictions != label
        elite_order = list(np.argsort(scores)[::-1][:9])
        elite_order += list(np.argsort(np.where(successful, fidelities, np.inf))[:3])
        elite = np.asarray(list(dict.fromkeys(elite_order))[:12], dtype=int)
        parents = pool[elite]
        children = []
        for query in range(TEMP_PER_GENERATION):
            parent = parents[query % len(parents)].copy()
            first, second = parents[rng.integers(0, len(parents), size=2)]
            proposal = parent + 0.5 * (first - second)
            coordinates = rng.choice(len(clean), size=max(1, min(8, len(clean))), replace=False,
                                     p=sensitivity)
            proposal[coordinates] += rng.uniform(-epsilon * scale, epsilon * scale, len(coordinates))
            if query < 50:
                directions = rng.choice((-1.0, 1.0), len(coordinates))
                proposal[coordinates] = clean[coordinates] + directions * epsilon
            children.append(project_timestamps(proposal, clean, epsilon, t0, tlast))
        children = np.asarray(children)
        child_values = _chunked_evaluate(model, events, children, clean_state, label, counter, chunk_size)
        child_losses, child_margins, child_predictions, child_fidelities = child_values
        combined_pool = np.vstack((parents, children))
        combined_losses = np.concatenate((losses[elite], child_losses))
        combined_margins = np.concatenate((margins[elite], child_margins))
        combined_predictions = np.concatenate((predictions[elite], child_predictions))
        combined_fidelities = np.concatenate((fidelities[elite], child_fidelities))
        combined_scores = score_population(combined_margins, combined_fidelities)
        keep = np.argsort(combined_scores)[::-1][:12]
        pool, losses, margins, predictions, fidelities, scores = (
            combined_pool[keep], combined_losses[keep], combined_margins[keep],
            combined_predictions[keep], combined_fidelities[keep], combined_scores[keep]
        )
    best = int(np.argmax(scores))
    return pool[best], {
        "candidate_evaluations": TEMP_CANDIDATES, "initial_candidates": TEMP_INITIAL,
        "refinement_generations": TEMP_GENERATIONS, "candidates_per_generation": TEMP_PER_GENERATION,
        "objective_weights": {"fidelity_drift": 0.35, "margin_reduction": 0.65},
        "sensitivity_guidance": "inverse distance to one of the four frozen clean-window bin boundaries",
    }


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _csv_text(rows, columns):
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _summarize(samples):
    rows = []
    for epsilon in EPSILON_FRACTIONS:
        for attack in ATTACKS:
            selected = [row for row in samples if row["epsilon_fraction"] == epsilon and row["attack"] == attack]
            successful = [row for row in selected if row["attack_success"]]
            rows.append({
                "epsilon_fraction": epsilon, "attack": attack, "sample_count": len(selected),
                "subset_clean_accuracy": sum(row["clean_correct"] for row in selected) / len(selected),
                "attacked_accuracy": sum(not row["attack_success"] for row in selected) / len(selected),
                "asr": len(successful) / len(selected), "successes": len(successful),
                "mean_one_minus_fidelity": float(np.mean([1.0 - row["fidelity"] for row in selected])),
                "mean_trace_distance": float(np.mean([row["trace_distance"] for row in selected])),
                "successful_mean_one_minus_fidelity": (float(np.mean([1.0 - row["fidelity"] for row in successful]))
                                                        if successful else None),
                "successful_mean_trace_distance": (float(np.mean([row["trace_distance"] for row in successful]))
                                                     if successful else None),
                "mean_drift_all": float(np.mean([row["mean_absolute_timestamp_drift"] for row in selected])),
                "mean_drift_successful_only": (float(np.mean([row["mean_absolute_timestamp_drift"] for row in successful]))
                                                if successful else None),
                "mean_margin_change_clean_minus_attacked": float(np.mean([row["margin_change_clean_minus_attacked"] for row in selected])),
                "feasibility_rate": sum(row["feasible"] for row in selected) / len(selected),
                "candidate_evaluations": sum(row["candidate_evaluations"] for row in selected),
                "runtime_seconds": sum(row["runtime_seconds"] for row in selected),
                "classifier_forward_calls": sum(row["classifier_forward_calls"] for row in selected),
                "classifier_sample_equivalent_evaluations": sum(row["classifier_sample_equivalent_evaluations"] for row in selected),
                "state_forward_calls": sum(row["state_forward_calls"] for row in selected),
                "state_sample_equivalent_evaluations": sum(row["state_sample_equivalent_evaluations"] for row in selected),
            })
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is prohibited for this comparison.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("Selected checkpoint SHA-256 mismatch.")
    if sha256(SPLIT) != SPLIT_SHA256:
        raise RuntimeError("Frozen split manifest SHA-256 mismatch.")
    manifest = json.loads(SPLIT.read_text(encoding="utf-8"))
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    if len(validation_ids) != 5000 or len(train_ids) != 55000 or np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Frozen split integrity check failed.")
    # A JSON artifact is the completion marker; never leave an older one after a failed rerun.
    OUTPUT_JSON.unlink(missing_ok=True)

    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data" / "nmnist"), train=True)
    if len(dataset) != 60000:
        raise RuntimeError("Unexpected tonic N-MNIST official-training size.")
    checkpoint = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    model = NMNISTCudaQSNN(12, 6, 10, 4, 8).cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    labels, clean_predictions, validation_features = [], [], []
    for sample_id in validation_ids:
        events, label = dataset[int(sample_id)]
        labels.append(int(label))
        validation_features.append(representation_numpy(events))
    validation_features = np.stack(validation_features)
    with torch.no_grad():
        for start in range(0, len(validation_features), 256):
            features = torch.from_numpy(validation_features[start:start + 256]).cuda()
            clean_predictions.extend(model(features).argmax(1).cpu().tolist())
    selected_ids = select_stratified_clean_correct(validation_ids, labels, clean_predictions)

    samples = []
    for position, sample_id in enumerate(selected_ids):
        events, label = dataset[int(sample_id)]
        label = int(label)
        clean = np.asarray(events["t"], dtype=np.float64)
        if not len(clean) or np.any(np.diff(clean) < 0):
            raise RuntimeError(f"Invalid native event timing for sample {sample_id}.")
        t0, tlast = float(clean[0]), float(clean[-1])
        duration = tlast - t0 + 1.0
        clean_features = torch.from_numpy(representation_numpy(events)).unsqueeze(0).cuda()
        with torch.no_grad():
            clean_logits = model(clean_features)
            clean_prediction = int(clean_logits.argmax(1).item())
            clean_margin = float(_margin(clean_logits, label).item())
            clean_state = final_state(model, clean_features)
        if clean_prediction != label:
            raise RuntimeError("Selected clean-correct sample changed prediction.")
        for epsilon_fraction in EPSILON_FRACTIONS:
            epsilon = epsilon_fraction * duration
            for attack in ATTACKS:
                counter = EvaluationCounter()
                started = time.perf_counter()
                if attack == "PGD":
                    attacked, details = pgd_attack(model, events, label, epsilon, counter)
                else:
                    attacked, details = temp_attack(
                        model, events, label, epsilon, clean_state, counter, CUDA_CHUNK,
                        SEED + position * 1009 + int(epsilon_fraction * 100),
                    )
                runtime = time.perf_counter() - started
                attacked = project_timestamps(attacked, clean, epsilon, t0, tlast)
                if not exact_feasibility(clean, attacked, epsilon, t0, tlast):
                    raise RuntimeError("Attack returned an infeasible candidate.")
                attacked_features = _represent_torch(events, torch.as_tensor(attacked, device="cuda", dtype=torch.float32))
                with torch.no_grad():
                    attacked_logits = counter.classifier(model, attacked_features)
                    attacked_state = counter.state(model, attacked_features)
                    prediction = int(attacked_logits.argmax(1).item())
                    attacked_margin = float(_margin(attacked_logits, label).item())
                    fidelity = float(torch.sum(clean_state.conj() * attacked_state, dim=1).abs().square().clamp(0, 1).item())
                counts = counter.snapshot()
                row = {
                    "sample_id": sample_id, "validation_order": int(np.flatnonzero(validation_ids == sample_id)[0]),
                    "label": label, "event_count": len(events), "t0": t0, "tlast": tlast,
                    "duration_inclusive": duration, "epsilon_fraction": epsilon_fraction, "epsilon": epsilon,
                    "epsilon_span_semantics": "epsilon_fraction * inclusive clean duration (tlast-t0+1)",
                    "attack": attack, "clean_prediction": clean_prediction, "attacked_prediction": prediction,
                    "clean_correct": True, "attack_success": prediction != label,
                    "clean_margin": clean_margin, "attacked_margin": attacked_margin,
                    "margin_change_clean_minus_attacked": clean_margin - attacked_margin,
                    "fidelity": fidelity, "one_minus_fidelity": 1.0 - fidelity,
                    "trace_distance": math.sqrt(max(0.0, 1.0 - fidelity)),
                    "mean_absolute_timestamp_drift": float(np.mean(np.abs(attacked - clean))),
                    "max_absolute_timestamp_drift": float(np.max(np.abs(attacked - clean))),
                    "feasible": exact_feasibility(clean, attacked, epsilon, t0, tlast),
                    "x_preserved": True, "y_preserved": True, "p_preserved": True,
                    "label_preserved": True, "event_count_preserved": True,
                    "runtime_seconds": runtime, **details, **counts,
                }
                samples.append(row)
                print(f"sample={position + 1}/100 id={sample_id} epsilon={epsilon_fraction:.2f} attack={attack} ASR={int(row['attack_success'])}", flush=True)

    summary = _summarize(samples)
    overlaps = []
    for epsilon in EPSILON_FRACTIONS:
        keyed = {(row["sample_id"], row["attack"]): row for row in samples if row["epsilon_fraction"] == epsilon}
        ids = selected_ids
        overlap = paired_overlap([keyed[(sample_id, "PGD")]["attack_success"] for sample_id in ids],
                                 [keyed[(sample_id, "TEMP-DRIFT")]["attack_success"] for sample_id in ids])
        overlaps.append({"epsilon_fraction": epsilon, "paired_samples": len(ids), **overlap})
    verdict = comparison_verdict(summary)
    config = {
        "seed": SEED, "validation_only": True, "tonic_nmnist_train": True,
        "official_test_instantiated": False, "selected_samples": 100, "per_class": 10,
        "selection": "first 10 clean-correct per class in frozen validation manifest order",
        "epsilon_fractions": EPSILON_FRACTIONS,
        "epsilon_span_semantics": "inclusive original clean window duration=tlast-t0+1",
        "projection": "clean +/- epsilon intersected with original t0..tlast, then nondecreasing",
        "pgd": {"steps": PGD_STEPS, "step_size": "epsilon/5", "random_start": False,
                "objective": "true-label CE", "selection": "highest unchanged-hard-bin CE iterate",
                "surrogate": "triangular soft bins; straight-through hard forward/soft gradient"},
        "temp_drift": {"derivative_free": True, "candidate_budget": TEMP_CANDIDATES,
                       "initialization": TEMP_INITIAL, "refinement_generations": TEMP_GENERATIONS,
                       "candidates_per_generation": TEMP_PER_GENERATION, "elite_preservation": True,
                        "differential_mutation": True, "boundary_priority": True,
                        "boundary_crossing_bonus": 1.0,
                       "fidelity_weight": 0.35, "margin_weight": 0.65},
        "candidate_batch_size": CUDA_CHUNK,
        "evaluation_definitions": {
            "classifier_forward_calls": "number of batched frozen classifier invocations",
            "classifier_sample_equivalent_evaluations": "sum of candidate/sample rows across classifier invocations",
            "state_forward_calls": "number of batched pure-state helper invocations",
            "state_sample_equivalent_evaluations": "sum of candidate/sample rows across state helper invocations",
            "candidate_evaluations": "hard candidate evaluations in the attack budget; PGD surrogate-gradient forwards are separate",
        },
        "verdict_rule": "STRONGER iff TEMP ASR is never lower and is >=5 percentage points higher at least once; COMPETITIVE iff never >5 points lower and higher at least once; otherwise NOT BENEFICIAL",
    }
    artifact = {
        "schema_version": 1, "study": "N-MNIST attack comparison seed 42", "config": config,
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_path": str(SPLIT.relative_to(ROOT)).replace("\\", "/"), "split_sha256": SPLIT_SHA256,
        "selected_validation_ids": selected_ids, "summary": summary, "paired_overlaps": overlaps,
        "samples": samples, "verdict": verdict,
        "scientific_scope": "Single-seed exploratory validation evidence; no robustness or superiority claim.",
        "outputs": {"json": str(OUTPUT_JSON.relative_to(ROOT)), "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                    "sample_csv": str(SAMPLE_CSV.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))},
    }
    summary_columns = list(summary[0])
    sample_columns = list(samples[0])
    report_lines = [
        "# N-MNIST Seed-42 Validation Attack Comparison", "",
        "Validation-only comparison on the same 100 clean-correct samples. The official test partition was not instantiated.", "",
        f"- Checkpoint: `{artifact['checkpoint_path']}` (`{CHECKPOINT_SHA256}`)",
        f"- Split: `{artifact['split_path']}` (`{SPLIT_SHA256}`)",
        "- Temporal span: inclusive clean duration `tlast - t0 + 1`; each epsilon is a fraction of this span.",
        "- Margin change: clean true-class margin minus attacked true-class margin.",
        "- Trace metric: `sqrt(1-F)` for clean/attacked final pure premeasurement circuit states.", "",
        "| Epsilon | Attack | ASR | Attacked Acc | 1-Fidelity | Trace Distance | Feasibility | Runtime |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report_lines.append(f"| {100 * row['epsilon_fraction']:.0f}% | {row['attack']} | {row['asr']:.4f} | {row['attacked_accuracy']:.4f} | {row['mean_one_minus_fidelity']:.6f} | {row['mean_trace_distance']:.6f} | {row['feasibility_rate']:.4f} | {row['runtime_seconds']:.2f} s |")
    report_lines.extend(["", "| Epsilon | PGD-only | TEMP-only | Both Success | Both Robust |",
                         "|---|---:|---:|---:|---:|"])
    for row in overlaps:
        report_lines.append(f"| {100 * row['epsilon_fraction']:.0f}% | {row['broken']} | {row['rescued']} | {row['both_success']} | {row['both_robust']} |")
    report_lines.extend(["", "Successful-attack-only drift and exact evaluation counts are included in the JSON and comparison CSV.",
                         "", f"**{verdict}**", "", artifact["scientific_scope"]])
    # JSON is the completion marker and is replaced last, preventing a partial run from looking complete.
    _atomic_text(SUMMARY_CSV, _csv_text(summary, summary_columns))
    _atomic_text(SAMPLE_CSV, _csv_text(samples, sample_columns))
    _atomic_text(REPORT, "\n".join(report_lines) + "\n")
    _atomic_text(OUTPUT_JSON, json.dumps(artifact, indent=2) + "\n")

    print("\nN-MNIST QSNN TIMING ATTACK COMPARISON", flush=True)
    by_key = {(row["epsilon_fraction"], row["attack"]): row for row in summary}
    for epsilon in EPSILON_FRACTIONS:
        pgd, temp = by_key[(epsilon, "PGD")], by_key[(epsilon, "TEMP-DRIFT")]
        print(f"\nEpsilon {100 * epsilon:.0f}%:", flush=True)
        print(f"PGD ASR: {pgd['asr']:.4f}", flush=True)
        print(f"TEMP ASR: {temp['asr']:.4f}", flush=True)
        print(f"PGD 1-Fidelity: {pgd['mean_one_minus_fidelity']:.6f}", flush=True)
        print(f"TEMP 1-Fidelity: {temp['mean_one_minus_fidelity']:.6f}", flush=True)
    print("\nPaired unique successes:", flush=True)
    for row in overlaps:
        print(f"{100 * row['epsilon_fraction']:.0f}%: PGD-only={row['broken']}, TEMP-only={row['rescued']}", flush=True)
    print(f"\nFinal scientific verdict:\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
