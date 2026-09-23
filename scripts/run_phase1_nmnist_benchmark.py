"""Resumable Phase 1 N-MNIST PIL-PGD benchmark and pre-attack dry run."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.pil_pgd import PILPGDAttack, PILPGDConfig
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json, sha256_file

PYTHON = ROOT.parent.parent / ".venv" / "Scripts" / "python.exe"
BUDGETS = {"B_inf": (1, 2, 3), "B1": (500, 750, 1000, 1500), "B0": (200, 300, 400, 600)}
REPRESENTATIONS = ("binary", "integer")
ATTACK_BATCH_SIZE = 64
AUDIT_BATCH_SIZE = 64
RESULT_COLUMNS = (ROOT / "ResearchLoop/reference/result_columns.txt").read_text().split()


def write_live_progress(*, run_id: str, seed: int, representation: str, budget_type: str,
                        beta: int, completed: int, total: int, elapsed: float,
                        state: str = "RUNNING") -> None:
    rate = elapsed / completed if completed else 0.0
    eta = rate * (total - completed)
    gpu = {"gpu_utilization_percent": None, "vram_used_mib": None}
    try:
        query = subprocess.check_output([
            r"C:\Windows\System32\nvidia-smi.exe",
            "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip().split(",")
        gpu = {"gpu_utilization_percent": float(query[0].strip()),
               "vram_used_mib": float(query[1].strip())}
    except Exception:
        pass
    payload = {
        "status": state, "run_id": run_id, "seed": seed, "representation": representation,
        "budget_type": budget_type, "beta": beta, "completed_samples": completed,
        "total_samples": total, "progress_percent": 100.0 * completed / total,
        "elapsed_seconds": elapsed, "seconds_per_sample": rate, "eta_seconds": eta,
        "updated_at": datetime.now(timezone.utc).isoformat(), "runner": "OPTIMIZED",
        "selected_attack_batch_size": ATTACK_BATCH_SIZE,
        "selected_audit_batch_size": AUDIT_BATCH_SIZE, **gpu,
    }
    atomic_write_json(ROOT / "Reports/checkpoints/phase1_live_progress.json", payload)
    status_path = ROOT / "Reports/status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status["status"] = "RUNNING"
        status["reason"] = f"Phase 1 optimized runner active: {run_id} {completed}/{total}."
        status["updated_at"] = payload["updated_at"]
        status["phases"]["1"]["status"] = "RUNNING"
        status["phases"]["1"]["active_condition"] = payload
        atomic_write_json(status_path, status)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for attempt in range(60):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 59:
                    raise
                time.sleep(min(0.05 * (attempt + 1), 0.5))
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def load_or_build_frozen_cache(dataset, seed: int, manifest: dict) -> dict:
    """Load immutable manifest-ordered T=10 grids, or construct them atomically."""
    directory = ROOT / "Reports/checkpoints/nmnist_preprocessed"
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"seed{seed}_t10_uint8.npz"
    metadata_path = directory / f"seed{seed}_t10_uint8.json"
    expected_ids = np.asarray([int(row["sample_id"]) for row in manifest["samples"]], dtype=np.int32)
    if artifact.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("manifest_sha256") == manifest["manifest_sha256"]
                and metadata.get("artifact_sha256") == sha256_file(artifact)):
            payload = np.load(artifact, allow_pickle=False)
            if np.array_equal(payload["sample_ids"], expected_ids):
                integer = payload["integer"]
                labels = payload["labels"]
            else:
                raise RuntimeError("cached sample order differs from frozen manifest")
        else:
            raise RuntimeError("frozen preprocessing cache provenance mismatch")
    else:
        frames, labels = [], []
        for row in manifest["samples"]:
            events, label = dataset[int(row["sample_id"])]
            frames.append(events_to_frames(events, 10))
            labels.append(int(label))
        integer = np.stack(frames).astype(np.uint8, copy=False)
        labels = np.asarray(labels, dtype=np.int8)
        digest = atomic_npz(artifact, sample_ids=expected_ids, labels=labels, integer=integer)
        atomic_write_json(metadata_path, {
            "status": "PASS", "seed": seed, "time_bins": 10,
            "manifest_sha256": manifest["manifest_sha256"], "sample_count": len(expected_ids),
            "artifact_path": str(artifact.relative_to(ROOT)), "artifact_sha256": digest,
            "integer_dtype": "uint8", "binary_definition": "integer > 0",
        })
    integer_tensor = torch.from_numpy(np.asarray(integer))
    binary_tensor = integer_tensor.ne(0).to(torch.uint8)
    labels_tensor = torch.from_numpy(np.asarray(labels))
    if torch.cuda.is_available():
        integer_tensor = integer_tensor.pin_memory()
        binary_tensor = binary_tensor.pin_memory()
        labels_tensor = labels_tensor.pin_memory()
    return {"integer": integer_tensor, "binary": binary_tensor, "labels": labels_tensor,
            "sample_ids": expected_ids, "artifact": artifact, "metadata": metadata_path}


def load_model(seed: int, device: torch.device):
    config_path = ROOT / "configs/nmnist_snn_clean_seed42.json"
    config = json.loads(config_path.read_text())
    checkpoint_path = ROOT / f"checkpoints/nmnist_snn_clean_seed{seed}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NMNISTConvSNN(config["lif_decay"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, config_path, checkpoint_path


def direct_validate(clean: np.ndarray, adv: np.ndarray, displacement: np.ndarray,
                    budget_type: str, beta: int) -> tuple[int, int, int]:
    source = np.argwhere(clean.reshape(10, -1) != 0)
    source_t, line = source[:, 0], source[:, 1]
    delta = displacement.reshape(10, -1)[source_t, line].astype(np.int64)
    target = source_t + delta
    if np.any(target < 0) or np.any(target >= 10):
        raise RuntimeError("projection left temporal domain")
    if len(set(zip(line.tolist(), target.tolist()))) != len(target):
        raise RuntimeError("projection produced a cell-packet collision")
    reconstructed = np.zeros_like(clean.reshape(10, -1))
    reconstructed[target, line] = clean.reshape(10, -1)[source_t, line]
    if not np.array_equal(reconstructed.reshape(clean.shape), adv):
        raise RuntimeError("displacement serialization does not reconstruct strict tensor")
    if int(clean.sum()) != int(adv.sum()):
        raise RuntimeError("projection changed total amplitude mass")
    clean_values = np.sort(clean[clean != 0].astype(np.int64))
    adv_values = np.sort(adv[adv != 0].astype(np.int64))
    if not np.array_equal(clean_values, adv_values):
        raise RuntimeError("projection changed/split/merged packet amplitudes")
    absolute = np.abs(delta)
    b_inf, b1, b0 = int(absolute.max(initial=0)), int(absolute.sum()), int((absolute != 0).sum())
    observed = {"B_inf": b_inf, "B1": b1, "B0": b0}[budget_type]
    if observed > beta:
        raise RuntimeError(f"projection exceeded {budget_type}: {observed}>{beta}")
    return b_inf, b1, b0


def run_condition(*, cache, model, device, seed: int, representation: str,
                  budget_type: str, beta: int, manifest: dict, output_dir: Path,
                  dry_run: bool) -> tuple[Path, Path]:
    run_id = (f"nmnist_dry_seed{seed}_{representation}_{budget_type}_{beta}" if dry_run else
              f"nmnist_seed{seed}_{representation}_{budget_type}_{beta}")
    metadata_path = output_dir / f"{run_id}.json"
    audit_path = output_dir / f"{run_id}.audit.json"
    marker_path = output_dir / f"{run_id}.complete.json"
    if marker_path.exists() and audit_path.exists():
        prior = json.loads(audit_path.read_text())
        if prior.get("passed"):
            print(f"RESUME skip validated condition {run_id}", flush=True)
            return metadata_path, audit_path

    rows = manifest["samples"][:1] if dry_run else manifest["samples"]
    attack_config = PILPGDConfig(budget_type, beta, projector="cuda")
    sample_ids, labels, clean_predictions, adv_predictions = [], [], [], []
    offsets, source_t_all, line_all, target_t_all, value_all = [0], [], [], [], []
    reported_b_inf, reported_b1, reported_b0 = [], [], []
    attack_times, gpu_times, frame_metrics = [], [], []
    partial_path = output_dir / f"{run_id}.partial.npz"
    partial_pointer = output_dir / f"{run_id}.partial.json"
    start_position = 0
    resumed_prefix_attack_batch_size = None
    resume_partial = partial_path
    if not dry_run and partial_pointer.exists():
        pointer = json.loads(partial_pointer.read_text())
        resumed_prefix_attack_batch_size = pointer.get("attack_batch_size")
        resume_partial = ROOT / pointer["artifact_path"]
        if sha256_file(resume_partial) != pointer["artifact_sha256"]:
            raise RuntimeError("partial checkpoint generation hash mismatch")
    if not dry_run and resume_partial.exists():
        with np.load(resume_partial, allow_pickle=False) as partial:
            completed_ids = partial["sample_ids"].astype(np.int64).tolist()
            expected_prefix = [int(row["sample_id"]) for row in rows[:len(completed_ids)]]
            if completed_ids != expected_prefix:
                raise RuntimeError("partial checkpoint is not a frozen-manifest prefix")
            sample_ids = completed_ids
            labels = partial["labels"].astype(int).tolist()
            clean_predictions = partial["clean_predictions"].astype(int).tolist()
            adv_predictions = partial["adv_predictions"].astype(int).tolist()
            offsets = partial["offsets"].astype(int).tolist()
            source_t_all = partial["source_t"].astype(int).tolist()
            line_all = partial["line"].astype(int).tolist()
            target_t_all = partial["target_t"].astype(int).tolist()
            value_all = partial["value"].astype(int).tolist()
            reported_b_inf = partial["reported_b_inf"].astype(int).tolist()
            reported_b1 = partial["reported_b1"].astype(int).tolist()
            reported_b0 = partial["reported_b0"].astype(int).tolist()
            attack_times = partial["attack_times"].astype(float).tolist()
            gpu_times = partial["gpu_times"].astype(float).tolist()
            frame_metrics = partial["frame_metrics"].astype(float).tolist()
        start_position = len(completed_ids)
    print(f"START {run_id} samples={len(rows)} resume_at={start_position + 1}", flush=True)
    write_live_progress(run_id=run_id, seed=seed, representation=representation,
                        budget_type=budget_type, beta=beta, completed=start_position,
                        total=len(rows), elapsed=sum(attack_times))

    def write_partial() -> None:
        completed = len(sample_ids)
        generation = output_dir / (
            f"{run_id}.partial.{completed:04d}.{time.time_ns()}.npz")
        digest = atomic_npz(
            generation, sample_ids=np.asarray(sample_ids, dtype=np.int32), labels=np.asarray(labels, dtype=np.int8),
            clean_predictions=np.asarray(clean_predictions, dtype=np.int8), adv_predictions=np.asarray(adv_predictions, dtype=np.int8),
            offsets=np.asarray(offsets, dtype=np.int64), source_t=np.asarray(source_t_all, dtype=np.int8),
            line=np.asarray(line_all, dtype=np.int16), target_t=np.asarray(target_t_all, dtype=np.int8),
            value=np.asarray(value_all, dtype=np.uint8), reported_b_inf=np.asarray(reported_b_inf, dtype=np.int16),
            reported_b1=np.asarray(reported_b1, dtype=np.int32), reported_b0=np.asarray(reported_b0, dtype=np.int32),
            attack_times=np.asarray(attack_times, dtype=np.float64), gpu_times=np.asarray(gpu_times, dtype=np.float64),
            frame_metrics=np.asarray(frame_metrics, dtype=np.float64))
        atomic_write_json(partial_pointer, {
            "status": "VALID_PREFIX", "run_id": run_id, "completed_samples": completed,
            "next_sample_position": completed + 1,
            "artifact_path": str(generation.relative_to(ROOT)), "artifact_sha256": digest,
            "manifest_sha256": manifest["manifest_sha256"],
            "attack_batch_size": ATTACK_BATCH_SIZE, "projector": "cuda_vectorized",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    frames_cpu = cache[representation]
    labels_cpu = cache["labels"]
    for batch_start in range(start_position, len(rows), ATTACK_BATCH_SIZE):
        batch_end = min(batch_start + ATTACK_BATCH_SIZE, len(rows))
        clean_cpu_batch = frames_cpu[batch_start:batch_end]
        clean_tensor_batch = clean_cpu_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        label_tensor_batch = labels_cpu[batch_start:batch_end].to(
            device=device, dtype=torch.long, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        attack = PILPGDAttack(model, attack_config)
        adversarial_batch, displacement_batch = attack(clean_tensor_batch, label_tensor_batch)
        logits_batch = attack.last_logits
        evaluated_batch = attack.last_evaluated_input
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_elapsed = time.perf_counter() - started
        if not torch.equal(adversarial_batch, evaluated_batch):
            raise RuntimeError("returned strict tensors were not exact model-evaluated tensors")
        adversarial_np = adversarial_batch.cpu().numpy().astype(np.uint8)
        displacement_np = displacement_batch.cpu().numpy().astype(np.int8)
        batch_predictions = logits_batch.argmax(1).cpu().tolist()
        per_sample_elapsed = batch_elapsed / (batch_end - batch_start)
        for local_index, position in enumerate(range(batch_start, batch_end)):
            manifest_row = rows[position]
            sample_id = int(manifest_row["sample_id"])
            label = int(labels_cpu[position])
            clean = clean_cpu_batch[local_index].numpy()
            adversarial = adversarial_np[local_index]
            displacement = displacement_np[local_index]
            b_inf, b1, b0 = direct_validate(clean, adversarial, displacement, budget_type, beta)
            source = np.argwhere(clean.reshape(10, -1) != 0)
            source_t, line = source[:, 0], source[:, 1]
            delta = displacement.reshape(10, -1)[source_t, line].astype(np.int64)
            target_t = source_t + delta
            values = clean.reshape(10, -1)[source_t, line]
            clean_prediction = int(manifest_row[f"{representation}_clean_prediction"])
            adv_prediction = int(batch_predictions[local_index])
            sample_ids.append(sample_id); labels.append(label)
            clean_predictions.append(clean_prediction); adv_predictions.append(adv_prediction)
            source_t_all.extend(source_t); line_all.extend(line); target_t_all.extend(target_t); value_all.extend(values)
            offsets.append(len(source_t_all))
            reported_b_inf.append(b_inf); reported_b1.append(b1); reported_b0.append(b0)
            attack_times.append(per_sample_elapsed); gpu_times.append(per_sample_elapsed)
            difference = np.abs(adversarial.astype(np.float64) - clean.astype(np.float64)).ravel()
            frame_metrics.append((float(np.count_nonzero(difference)), float(difference.sum()),
                                  float(np.sqrt(np.square(difference).sum())), float(difference.max(initial=0))))
            if not dry_run and ((position + 1) % 25 == 0 or position + 1 == len(rows)):
                write_partial()
                write_live_progress(run_id=run_id, seed=seed, representation=representation,
                                    budget_type=budget_type, beta=beta, completed=position + 1,
                                    total=len(rows), elapsed=sum(attack_times))
        final_position = batch_end - 1
        final_label = int(labels_cpu[final_position])
        print(f"{run_id} sample={batch_end}/{len(rows)} id={int(rows[final_position]['sample_id'])} "
              f"B=({reported_b_inf[-1]},{reported_b1[-1]},{reported_b0[-1]}) "
              f"success={adv_predictions[-1] != final_label}", flush=True)

    artifact_path = output_dir / f"{run_id}.npz"
    artifact_sha = atomic_npz(
        artifact_path, sample_ids=np.asarray(sample_ids, dtype=np.int32), labels=np.asarray(labels, dtype=np.int8),
        clean_predictions=np.asarray(clean_predictions, dtype=np.int8), adv_predictions=np.asarray(adv_predictions, dtype=np.int8),
        offsets=np.asarray(offsets, dtype=np.int64), source_t=np.asarray(source_t_all, dtype=np.int8),
        line=np.asarray(line_all, dtype=np.int16), target_t=np.asarray(target_t_all, dtype=np.int8),
        value=np.asarray(value_all, dtype=np.uint8), reported_b_inf=np.asarray(reported_b_inf, dtype=np.int16),
        reported_b1=np.asarray(reported_b1, dtype=np.int32), reported_b0=np.asarray(reported_b0, dtype=np.int32),
    )
    config_path = ROOT / "configs/nmnist_snn_clean_seed42.json"
    checkpoint_path = ROOT / f"checkpoints/nmnist_snn_clean_seed{seed}_best.pt"
    manifest_path = ROOT / f"Reports/checkpoints/n_mnist_seed{seed}_clean_correct_manifest.json"
    metadata = {
        "run_id": run_id, "dry_run": dry_run, "benchmark_result": not dry_run, "dataset": "N-MNIST",
        "seed": seed, "representation": representation, "budget_type": budget_type, "beta": beta,
        "sample_count": len(rows), "artifact_path": str(artifact_path.relative_to(ROOT)),
        "artifact_sha256": artifact_sha, "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest["manifest_sha256"], "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path), "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path), "exact_command": " ".join(sys.argv),
        "packet_semantics": "one indivisible amplitude-bearing packet per nonzero temporal cell",
        "no_unit_expansion": True, "attack_time_seconds": attack_times, "gpu_time_seconds": gpu_times,
        "frame_metrics": frame_metrics, "queries_per_sample": (20 if budget_type == "B_inf" else 40) + 1,
        "preprocessing_cache_path": str(cache["artifact"].relative_to(ROOT)),
        "preprocessing_cache_sha256": sha256_file(cache["artifact"]),
        "partial_checkpoint_interval": 25,
        "partial_checkpoint_format": "immutable generation NPZ plus atomic JSON pointer",
        "attack_batch_size": ATTACK_BATCH_SIZE,
        "audit_batch_size": AUDIT_BATCH_SIZE,
        "attack_projector": "cuda_vectorized_batch64",
        "resumed_prefix_samples": start_position,
        "resumed_prefix_attack_batch_size": resumed_prefix_attack_batch_size,
        "gpu_vectorization_validation": "Reports/results/phase1_gpu_vectorization_equivalence.json",
    }
    atomic_write_json(metadata_path, metadata)
    audit_command = [str(PYTHON), "-m", "ResearchLoop.tools.audit_nmnist_phase1",
                     str(metadata_path.relative_to(ROOT)), "--output", str(audit_path.relative_to(ROOT))]
    subprocess.run(audit_command, cwd=ROOT, check=True)
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed"):
        raise RuntimeError(f"independent audit failed: {audit.get('errors', [])[:3]}")
    atomic_write_json(marker_path, {"status": "PASS", "run_id": run_id,
                                   "metadata_sha256": sha256_file(metadata_path),
                                   "artifact_sha256": artifact_sha, "audit_sha256": sha256_file(audit_path)})
    write_live_progress(run_id=run_id, seed=seed, representation=representation,
                        budget_type=budget_type, beta=beta, completed=len(rows), total=len(rows),
                        elapsed=sum(attack_times), state="AUDIT_PASS")
    return metadata_path, audit_path


def update_summary(metadata_path: Path, audit_path: Path) -> None:
    meta, audit = json.loads(metadata_path.read_text()), json.loads(audit_path.read_text())
    if meta["dry_run"]:
        return
    payload = np.load(ROOT / meta["artifact_path"], allow_pickle=False)
    n = len(payload["sample_ids"])
    successes = sum(row["attack_success"] for row in audit["samples"])
    fm = np.asarray(meta["frame_metrics"], dtype=float)
    displacement_count = sum(row["packet_count"] for row in audit["samples"])
    row = {
        "run_id": meta["run_id"], "dataset": "N-MNIST", "model": "NMNISTConvSNN",
        "representation": meta["representation"], "seed": meta["seed"], "budget_type": meta["budget_type"],
        "beta": meta["beta"], "n_attacked": n, "clean_correct_count": n, "attack_success_count": successes,
        "clean_accuracy": 100.0, "adversarial_accuracy": 100.0 * (n - successes) / n,
        "asr": 100.0 * successes / n, "b_inf_max": int(payload["reported_b_inf"].max(initial=0)),
        "b1_max": int(payload["reported_b1"].max(initial=0)), "b0_max": int(payload["reported_b0"].max(initial=0)),
        "mean_abs_dt": sum(row["realized_b1"] for row in audit["samples"]) / max(displacement_count, 1),
        "changed_event_ratio": sum(row["realized_b0"] for row in audit["samples"]) / max(displacement_count, 1),
        "frame_l0": fm[:, 0].mean(), "frame_l1": fm[:, 1].mean(), "frame_l2": fm[:, 2].mean(),
        "frame_linf": fm[:, 3].mean(), "queries_mean": meta["queries_per_sample"],
        "attack_time_s_mean": float(np.mean(meta["attack_time_seconds"])),
        "gpu_time_s_mean": float(np.mean(meta["gpu_time_seconds"])), "audit_pass": True,
        "checkpoint_sha256": meta["checkpoint_sha256"], "config_sha256": meta["config_sha256"],
        "log_path": f"Reports/logs/phase_1_seed{meta['seed']}.log", "comparison_status": "NON_COMPARABLE",
    }
    summary_path = ROOT / "Reports/results/nmnist/summary.csv"
    existing = []
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            existing = [item for item in csv.DictReader(handle) if item["run_id"] != row["run_id"]]
    existing.append(row)
    existing.sort(key=lambda item: (int(item["seed"]), item["representation"], item["budget_type"], int(item["beta"])))
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader(); writer.writerows(existing)
    os.replace(temporary, summary_path)


def write_progress_report() -> None:
    summary = ROOT / "Reports/results/nmnist/summary.csv"
    rows = list(csv.DictReader(summary.open(encoding="utf-8"))) if summary.exists() else []
    expected = len(REPRESENTATIONS) * sum(len(values) for values in BUDGETS.values()) * 3
    text = (
        "# Phase 1 N-MNIST benchmark\n\n"
        f"Status: **{'PASS' if len(rows) == expected else 'RUNNING'}** ({len(rows)}/{expected} independently audited cells).\n\n"
        "The obsolete unit-packet blocker is resolved. Every integer nonzero temporal cell is one "
        "indivisible amplitude-bearing packet. Dry-run artifacts are excluded from this summary.\n"
    )
    path = ROOT / "Reports/phase_1_NMNIST_report.md"
    temporary = path.with_suffix(".md.tmp"); temporary.write_text(text, encoding="utf-8"); os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 123, 777))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"wrong interpreter: expected {PYTHON}")
    if sum((args.dry_run, args.seed is not None, args.all)) != 1:
        raise ValueError("select exactly one of --dry-run, --seed, or --all")
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = (42, 123, 777) if args.all else (42 if args.dry_run else args.seed,)
    for seed in seeds:
        set_determinism(seed)
        model, _, _ = load_model(seed, device)
        manifest = json.loads((ROOT / f"Reports/checkpoints/n_mnist_seed{seed}_clean_correct_manifest.json").read_text())
        cache = load_or_build_frozen_cache(dataset, seed, manifest)
        if args.dry_run:
            output_dir = ROOT / "Reports/results/nmnist_dry_run"
            conditions = (("integer", "B_inf", 1), ("integer", "B1", 500),
                          ("integer", "B0", 200), ("binary", "B1", 500))
        else:
            output_dir = ROOT / "Reports/results/nmnist"
            conditions = tuple((representation, kind, beta) for representation in REPRESENTATIONS
                               for kind, betas in BUDGETS.items() for beta in betas)
        for representation, kind, beta in conditions:
            metadata, audit = run_condition(cache=cache, model=model, device=device, seed=seed,
                                             representation=representation, budget_type=kind, beta=beta,
                                             manifest=manifest, output_dir=output_dir, dry_run=args.dry_run)
            update_summary(metadata, audit)
            write_progress_report()
        if args.dry_run:
            audits = sorted(output_dir.glob("*.audit.json"))
            atomic_write_json(output_dir / "dry_run_PASS.json", {
                "status": "PASS", "benchmark_result": False, "seed": 42,
                "conditions": [str(path.relative_to(ROOT)) for path in audits],
                "audits": {str(path.relative_to(ROOT)): sha256_file(path) for path in audits},
            })
            print("DRY RUN PASS (outputs excluded from benchmark)", flush=True)


if __name__ == "__main__":
    main()
