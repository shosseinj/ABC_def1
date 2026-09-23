"""Exact old-vs-optimized equivalence test on a frozen N-MNIST subset."""
from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import attacks.pil_pgd as optimized
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json


@torch.no_grad()
def legacy_project(clean, probabilities, budget_type, beta):
    """Frozen pre-optimization projector, retained only for equivalence testing."""
    kind = optimized._kind(budget_type)
    b, t, c, h, w = clean.shape
    lines = c * h * w
    x = clean.reshape(b, t, lines)
    pi = probabilities.reshape(b, t, lines, -1)
    target, valid = optimized._targets(t, kind, beta, clean.device)
    out = torch.zeros_like(x)
    displacement = torch.zeros_like(x, dtype=torch.int64)
    for sample in range(b):
        sources = torch.nonzero(x[sample] != 0, as_tuple=False)
        reserved = {(int(s), int(line)) for s, line in sources.tolist()}
        occupied, placed, candidates = set(), set(), []
        for s_tensor, line_tensor in sources:
            s, line = int(s_tensor), int(line_tensor)
            for option in range(pi.shape[-1]):
                if not bool(valid[s, option]):
                    continue
                destination = int(target[s, option])
                if destination == s:
                    continue
                score = float(pi[sample, s, line, option])
                candidates.append((-score, abs(destination - s), s, line, destination))
        candidates.sort()
        spent = 0
        for _, distance, source, line, destination in candidates:
            key, destination_key = (source, line), (destination, line)
            if key in placed or destination_key in occupied or destination_key in reserved:
                continue
            cost = distance if kind == "B1" else 1 if kind == "B0" else 0
            if kind == "B_inf" and distance > beta:
                continue
            if kind in {"B1", "B0"} and spent + cost > beta:
                continue
            out[sample, destination, line] = x[sample, source, line]
            displacement[sample, source, line] = destination - source
            occupied.add(destination_key); reserved.discard(key); placed.add(key); spent += cost
        for source_tensor, line_tensor in sources:
            source, line = int(source_tensor), int(line_tensor)
            if (source, line) not in placed:
                out[sample, source, line] = x[sample, source, line]
                occupied.add((source, line))
    return out.reshape_as(clean), displacement.reshape_as(clean)


class LegacyAttack:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def __call__(self, clean, labels):
        kind = optimized._kind(self.config.budget_type)
        b, t, c, h, w = clean.shape
        target, valid = optimized._targets(t, kind, self.config.beta, clean.device)
        options = target.shape[1]
        logits = torch.zeros((b, t, c, h, w, options), device=clean.device,
                             dtype=clean.dtype, requires_grad=True)
        source_mask = (clean != 0).unsqueeze(-1)
        iterations = self.config.iterations or (20 if kind == "B_inf" else 40)
        source_times = torch.arange(t, device=clean.device).view(1, t, 1, 1, 1, 1)
        target_times = target.view(1, t, 1, 1, 1, options)
        distance = (target_times - source_times).abs().to(clean.dtype)
        for _ in range(iterations):
            masked = logits.masked_fill(~(source_mask & valid.view(1, t, 1, 1, 1, options)), -torch.inf)
            probabilities = torch.nan_to_num(torch.softmax(masked / self.config.temperature, dim=-1))
            soft = optimized.soft_retime(clean, probabilities, target, valid)
            hard, _ = legacy_project(clean, probabilities, kind, self.config.beta)
            pil = hard + soft - soft.detach()
            task = F.cross_entropy(self.model(pil), labels)
            occupancy = optimized.soft_retime((clean != 0).to(clean.dtype), probabilities, target, valid)
            capacity = F.relu(occupancy - 1).square().sum() / source_mask.sum().clamp_min(1)
            if kind == "B1":
                soft_cost = (probabilities * distance * source_mask).sum(dim=(1, 2, 3, 4, 5))
            elif kind == "B0":
                stay = target_times == source_times
                soft_cost = ((probabilities * (~stay).to(clean.dtype)) * source_mask).sum(dim=(1, 2, 3, 4, 5))
            else:
                soft_cost = torch.zeros(b, device=clean.device)
            penalty = F.relu(soft_cost / max(self.config.beta, 1) - 1).mean()
            objective = task - self.config.capacity_weight * capacity - self.config.budget_weight * penalty
            gradient, = torch.autograd.grad(objective, logits)
            with torch.no_grad():
                logits.add_(self.config.alpha * gradient.sign()).clamp_(-self.config.logit_clip,
                                                                         self.config.logit_clip)
            logits.requires_grad_(True)
        masked = logits.masked_fill(~(source_mask & valid.view(1, t, 1, 1, 1, options)), -torch.inf)
        probabilities = torch.nan_to_num(torch.softmax(masked / self.config.temperature, dim=-1))
        final, displacement = legacy_project(clean, probabilities, kind, self.config.beta)
        with torch.no_grad():
            final_logits = self.model(final).detach().clone()
        return final, displacement, final_logits


def realized(displacement, clean):
    values = displacement[clean != 0].abs()
    return int(values.max()), int(values.sum()), int((values != 0).sum())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256(path)


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    configured = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"])
    if Path(sys.executable).resolve() != configured.resolve():
        raise RuntimeError("wrong interpreter")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_data = json.loads((ROOT / "configs/nmnist_snn_clean_seed42.json").read_text())
    checkpoint = torch.load(ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt",
                            map_location=device, weights_only=True)
    model = NMNISTConvSNN(config_data["lif_decay"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    manifest = json.loads((ROOT / "Reports/checkpoints/n_mnist_seed42_clean_correct_manifest.json").read_text())
    by_id = {int(row["sample_id"]): row for row in manifest["samples"]}
    selected_ids = [6697, 415, 8930, 1098, 653]
    rows = [by_id[sample_id] for sample_id in selected_ids]
    frames, labels = [], []
    for row in rows:
        events, label = dataset[int(row["sample_id"])]
        frames.append((events_to_frames(events, 10) > 0).astype(np.uint8))
        labels.append(int(label))
    batch = torch.from_numpy(np.stack(frames).astype(np.float32)).to(device)
    labels_tensor = torch.tensor(labels, device=device)
    conditions = (("B_inf", 1), ("B1", 500), ("B0", 200))
    item_dir = ROOT / "Reports/checkpoints/phase1_equivalence_items"
    item_dir.mkdir(parents=True, exist_ok=True)
    progress_path = ROOT / "Reports/checkpoints/phase1_equivalence_progress.json"
    log_path = ROOT / "Reports/logs/phase1_equivalence_optimization.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    work = []
    for kind, beta in conditions:
        for index, row in enumerate(rows):
            for stage in ("old", "optimized", "compare_audit"):
                work.append((stage, kind, beta, index, int(row["sample_id"])))
    total_items = len(work)

    def stem(kind, beta, sample_id):
        return f"{kind}_{beta}_sample{sample_id}"

    def eval_valid(kind, beta, sample_id, implementation):
        base = stem(kind, beta, sample_id) + f"_{implementation}"
        artifact, metadata = item_dir / f"{base}.npz", item_dir / f"{base}.json"
        if not artifact.exists() or not metadata.exists():
            return False
        try:
            return json.loads(metadata.read_text()).get("artifact_sha256") == sha256(artifact)
        except Exception:
            return False

    def comparison_path(kind, beta, sample_id):
        return item_dir / f"{stem(kind, beta, sample_id)}_comparison.json"

    def item_complete(item):
        stage, kind, beta, _, sample_id = item
        if stage in {"old", "optimized"}:
            return eval_valid(kind, beta, sample_id, stage)
        path = comparison_path(kind, beta, sample_id)
        return path.exists() and "passed" in json.loads(path.read_text())

    def completed_elapsed():
        seconds = 0.0
        for path in item_dir.glob("*.json"):
            try:
                seconds += float(json.loads(path.read_text()).get("elapsed_seconds", 0.0))
            except Exception:
                pass
        return seconds

    def update_progress(status, current=None):
        completed = sum(item_complete(item) for item in work)
        elapsed = completed_elapsed()
        average = elapsed / completed if completed else 0.0
        eta = average * (total_items - completed)
        payload = {
            "status": status, "completed_items": completed, "total_items": total_items,
            "progress_percent": 100.0 * completed / total_items,
            "current_condition": None if current is None else {"budget_type": current[1], "beta": current[2], "stage": current[0]},
            "current_sample": None if current is None else current[4],
            "elapsed_seconds": elapsed, "average_seconds_per_item": average,
            "eta_seconds": eta, "eta_human": human_duration(eta),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "test_matrix": {"conditions": 3, "samples_per_condition": 5,
                            "old_evaluations": 15, "optimized_evaluations": 15,
                            "comparison_audits": 15},
        }
        atomic_write_json(progress_path, payload)
        return payload

    def log(message):
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    initial = update_progress("RUNNING")
    log(f"PLAN total={total_items} conditions=3 samples_per_condition=5 old=15 optimized=15 comparisons=15 "
        f"resume_completed={initial['completed_items']} ETA={initial['eta_human']}")
    for item in work:
        if item_complete(item):
            continue
        stage, kind, beta, index, sample_id = item
        before = update_progress("RUNNING", item)
        log(f"BEGIN [{before['completed_items'] + 1}/{total_items}] {stage} {kind}={beta} sample={sample_id} "
            f"elapsed={human_duration(before['elapsed_seconds'])} ETA={before['eta_human']}")
        started = time.perf_counter()
        if stage in {"old", "optimized"}:
            if stage == "old":
                adversarial, displacement, logits = LegacyAttack(
                    model, optimized.PILPGDConfig(kind, beta))(
                        batch[index:index + 1], labels_tensor[index:index + 1])
            else:
                attack = optimized.PILPGDAttack(model, optimized.PILPGDConfig(kind, beta))
                adversarial, displacement = attack(batch[index:index + 1], labels_tensor[index:index + 1])
                logits = attack.last_logits
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            base = stem(kind, beta, sample_id) + f"_{stage}"
            artifact = item_dir / f"{base}.npz"
            digest = atomic_npz(
                artifact, adversarial=adversarial.detach().cpu().numpy(),
                displacement=displacement.detach().cpu().numpy(), logits=logits.detach().cpu().numpy(),
                label=np.asarray([labels[index]], dtype=np.int8), sample_id=np.asarray([sample_id], dtype=np.int32))
            atomic_write_json(item_dir / f"{base}.json", {
                "status": "COMPLETE", "implementation": stage, "budget_type": kind, "beta": beta,
                "sample_id": sample_id, "artifact_sha256": digest,
                "elapsed_seconds": time.perf_counter() - started,
            })
        else:
            old = np.load(item_dir / f"{stem(kind, beta, sample_id)}_old.npz", allow_pickle=False)
            new = np.load(item_dir / f"{stem(kind, beta, sample_id)}_optimized.npz", allow_pickle=False)
            clean = batch[index].detach().cpu().numpy()
            old_adv, new_adv = old["adversarial"][0], new["adversarial"][0]
            old_disp, new_disp = old["displacement"][0], new["displacement"][0]
            source = np.argwhere(clean.reshape(10, -1) != 0)
            source_t, lines = source[:, 0], source[:, 1]
            old_delta = old_disp.reshape(10, -1)[source_t, lines]
            new_delta = new_disp.reshape(10, -1)[source_t, lines]
            old_target, new_target = source_t + old_delta, source_t + new_delta
            old_abs, new_abs = np.abs(old_delta), np.abs(new_delta)
            old_b = (int(old_abs.max(initial=0)), int(old_abs.sum()), int((old_abs != 0).sum()))
            new_b = (int(new_abs.max(initial=0)), int(new_abs.sum()), int((new_abs != 0).sum()))
            old_prediction, new_prediction = int(old["logits"].argmax(1)[0]), int(new["logits"].argmax(1)[0])
            clean_prediction = int(manifest["samples"][[int(r["sample_id"]) for r in manifest["samples"]].index(sample_id)]["binary_clean_prediction"])
            success_old, success_new = old_prediction != labels[index], new_prediction != labels[index]
            budget_index = {"B_inf": 0, "B1": 1, "B0": 2}[kind]
            old_collision_free = len(set(zip(lines.tolist(), old_target.tolist()))) == len(lines)
            new_collision_free = len(set(zip(lines.tolist(), new_target.tolist()))) == len(lines)
            old_audit = (old_b[budget_index] <= beta and old_collision_free
                         and int(old_adv.sum()) == int(clean.sum()))
            new_audit = (new_b[budget_index] <= beta and new_collision_free
                         and int(new_adv.sum()) == int(clean.sum()))
            record = {
                "sample_id": sample_id, "budget_type": kind, "beta": beta,
                "requested_beta_equal": True, "packet_count": int((clean != 0).sum()),
                "clean_prediction_old": clean_prediction, "clean_prediction_new": clean_prediction,
                "serialized_temporal_representation_exact": bool(np.array_equal(old_adv, new_adv)),
                "displacement_exact": bool(np.array_equal(old_disp, new_disp)),
                "packet_amplitudes_exact": bool(np.array_equal(np.sort(old_adv[old_adv != 0]), np.sort(new_adv[new_adv != 0]))),
                "packet_destinations_exact": bool(np.array_equal(old_target, new_target)),
                "old_attacked_prediction": old_prediction, "new_attacked_prediction": new_prediction,
                "old_success": success_old, "new_success": success_new,
                "old_realized_b_inf": old_b[0], "new_realized_b_inf": new_b[0],
                "old_realized_b1": old_b[1], "new_realized_b1": new_b[1],
                "old_realized_b0": old_b[2], "new_realized_b0": new_b[2],
                "old_audit_pass": old_audit, "new_audit_pass": new_audit,
            }
            record["passed"] = all((
                record["serialized_temporal_representation_exact"], record["displacement_exact"],
                record["packet_amplitudes_exact"], record["packet_destinations_exact"],
                old_prediction == new_prediction, success_old == success_new, old_b == new_b,
                old_audit == new_audit,
            ))
            record["elapsed_seconds"] = time.perf_counter() - started
            atomic_write_json(comparison_path(kind, beta, sample_id), record)
        state = update_progress("RUNNING")
        log(f"DONE [{state['completed_items']}/{total_items}] {state['progress_percent']:.1f}% "
            f"elapsed={human_duration(state['elapsed_seconds'])} avg={state['average_seconds_per_item']:.2f}s "
            f"ETA={state['eta_human']}")

    records = [json.loads(comparison_path(kind, beta, sample_id).read_text())
               for kind, beta in conditions for sample_id in selected_ids]
    passed = all(record["passed"] for record in records)
    result = {"status": "PASS" if passed else "FAIL", "exact_equality_required": True,
              "seed": 42, "sample_ids": selected_ids, "representation": "binary",
              "conditions": 3, "total_work_items": total_items,
              "elapsed_seconds": completed_elapsed(), "comparisons": records}
    atomic_write_json(ROOT / "Reports/results/phase1_runtime_equivalence.json", result)
    final = update_progress("PASS" if passed else "FAIL")
    log(f"FINAL {result['status']} completed={final['completed_items']}/{total_items} "
        f"elapsed={human_duration(final['elapsed_seconds'])}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
