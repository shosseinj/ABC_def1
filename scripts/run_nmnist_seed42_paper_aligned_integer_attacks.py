"""Run one resumable seed-42 paper-aligned N-MNIST PIL-PGD condition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.official_pil_adapter_batched import (
    UPSTREAM_COMMIT, UPSTREAM_SOURCE_SHA256, build_attack, load_official_attack_module,
)
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
CONFIG = ROOT / "configs/nmnist_seed42_paper_aligned_integer_attack.json"
CLEAN_RESULT = ROOT / "Reports/results/nmnist_seed42_paper_aligned_integer/seed42_clean_result.json"
OUT = ROOT / "Reports/results/nmnist_seed42_paper_aligned_integer"
STATE = ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned_integer"
BUDGETS = {"B_inf": (1, 2, 3), "B1": (500, 750, 1000, 1500), "B0": (200, 300, 400, 600)}
BATCH_SIZE = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path)


def set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)


def create_manifest(clean_result: dict) -> tuple[dict, Path]:
    path = OUT / "seed42_attack_manifest.json"
    prediction_path = ROOT / clean_result["predictions_path"]
    with np.load(prediction_path, allow_pickle=False) as payload:
        ids = payload["sample_ids"].astype(np.int64)
        labels = payload["labels"].astype(np.int64)
        predictions = payload["predictions"].astype(np.int64)
    eligible = np.flatnonzero(predictions == labels)
    selected_positions = eligible[:1000]
    if len(selected_positions) != 1000:
        raise RuntimeError("fewer than 1000 clean-correct test samples")
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    rows = []
    for position in selected_positions:
        sample_id = int(ids[position])
        events, raw_label = dataset[sample_id]
        event_digest = hashlib.sha256()
        for field in ("x", "y", "t", "p"):
            event_digest.update(np.ascontiguousarray(events[field]).tobytes())
        if int(raw_label) != int(labels[position]):
            raise RuntimeError(f"raw label mismatch at sample {sample_id}")
        rows.append({"position": len(rows), "sample_id": sample_id,
                     "true_label": int(raw_label), "clean_prediction": int(predictions[position]),
                     "raw_event_sha256": event_digest.hexdigest()})
    manifest = {
        "dataset": "N-MNIST", "partition": "official_test", "seed": 42,
        "sample_count": 1000, "eligible_clean_correct": int(len(eligible)),
        "selection": "first 1000 clean-correct samples encountered in unshuffled official test order",
        "selection_source": "official test.py attack loop at upstream commit " + UPSTREAM_COMMIT,
        "clean_predictions_path": clean_result["predictions_path"],
        "clean_predictions_sha256": sha256(prediction_path), "samples": rows,
        "samples_sha256": canonical_hash(rows),
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError("existing frozen attack manifest differs")
    else:
        atomic_json(path, manifest)
    return manifest, path


def validate(clean, adversarial, displacement, kind: str, budget: int):
    flat = clean.reshape(10, -1)
    source = np.argwhere(flat != 0)
    source_t, line = source[:, 0], source[:, 1]
    delta = displacement.reshape(10, -1)[source_t, line].astype(np.int64)
    target_t = source_t + delta
    if np.any(target_t < 0) or np.any(target_t >= 10):
        raise RuntimeError("temporal-domain violation")
    if len(set(zip(line.tolist(), target_t.tolist()))) != len(target_t):
        raise RuntimeError("capacity/collision violation")
    reconstructed = np.zeros_like(flat)
    values = flat[source_t, line]
    reconstructed[target_t, line] = values
    if not np.array_equal(reconstructed.reshape(clean.shape), adversarial):
        raise RuntimeError("packet reconstruction mismatch")
    absolute = np.abs(delta)
    realized = (int(absolute.max(initial=0)), int(absolute.sum()), int(np.count_nonzero(delta)))
    active = {"B_inf": realized[0], "B1": realized[1], "B0": realized[2]}[kind]
    if active > budget:
        raise RuntimeError(f"active budget exceeded: requested={budget}, actual={active}")
    if int(clean.sum(dtype=np.int64)) != int(adversarial.sum(dtype=np.int64)):
        raise RuntimeError("event mass changed")
    if np.count_nonzero(clean) != np.count_nonzero(adversarial):
        raise RuntimeError("packet count changed")
    return source_t, line, target_t, values, realized


def state_arrays():
    return {
        "sample_ids": ([], np.int32), "labels": ([], np.int8),
        "clean_predictions": ([], np.int8), "adv_predictions": ([], np.int8),
        "offsets": ([0], np.int64), "source_t": ([], np.int8),
        "line": ([], np.int16), "target_t": ([], np.int8), "value": ([], np.uint16),
        "realized_b_inf": ([], np.int16), "realized_b1": ([], np.int32),
        "realized_b0": ([], np.int32), "success": ([], np.bool_),
    }


def write_state(path: Path, state) -> str:
    return atomic_npz(path, **{key: np.asarray(values, dtype=dtype)
                              for key, (values, dtype) in state.items()})


def run(kind: str, budget: int) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    if budget not in BUDGETS[kind]:
        raise ValueError("budget is not a requested Table-1 cell")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    set_determinism()
    clean_result = json.loads(CLEAN_RESULT.read_text(encoding="utf-8"))
    if not clean_result.get("clean_gate_pass") or clean_result["test"]["accuracy"] < .95:
        raise RuntimeError("mandatory clean gate has not passed")
    manifest, manifest_path = create_manifest(clean_result)
    checkpoint_path = ROOT / clean_result["checkpoint_path"]
    cache_path = ROOT / clean_result["test_cache"]["path"]
    labels_path = STATE / "test_labels_int8.npy"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model = NMNISTConvSNN(0.5, 10).cuda().eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    official = load_official_attack_module()
    attack = build_attack(official, model, torch.device("cuda"), kind, budget)
    frames = np.load(cache_path, mmap_mode="r")
    labels_all = np.load(labels_path, mmap_mode="r")
    selected_ids = np.asarray([row["sample_id"] for row in manifest["samples"]], dtype=np.int64)
    selected_labels = labels_all[selected_ids].astype(np.int64)
    run_id = f"seed42_paper_aligned_integer_{kind}_{budget}"
    artifact_path = OUT / "attacks" / f"{run_id}.npz"
    metadata_path = OUT / "attacks" / f"{run_id}.json"
    audit_path = OUT / "attacks" / f"{run_id}.audit.json"
    marker_path = STATE / f"{run_id}.complete.json"
    partial_path = STATE / f"{run_id}.partial.npz"
    if marker_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "PASS":
            print(f"PHASE 2 {kind}={budget}: already complete ASR={audit['asr_percent']:.2f}%", flush=True)
            return
    state = state_arrays()
    if partial_path.exists():
        with np.load(partial_path, allow_pickle=False) as prior:
            for key, (_, dtype) in state.items():
                state[key] = (prior[key].astype(dtype).tolist(), dtype)
    start = len(state["sample_ids"][0])
    if state["sample_ids"][0] != selected_ids[:start].tolist():
        raise RuntimeError("partial state is not a manifest prefix")
    started = time.perf_counter()
    successes = int(sum(state["success"][0]))
    print(f"PHASE 2 START {kind}={budget} seed=42 batch_size=64 resume={start}/1000", flush=True)
    print(f"official_source commit={UPSTREAM_COMMIT} sha256={UPSTREAM_SOURCE_SHA256}", flush=True)
    for batch_start in range(start, 1000, BATCH_SIZE):
        end = min(batch_start + BATCH_SIZE, 1000)
        batch_ids = selected_ids[batch_start:end]
        clean_np = np.array(frames[batch_ids], dtype=np.float32, copy=True)
        clean = torch.from_numpy(clean_np).cuda()
        labels = torch.from_numpy(selected_labels[batch_start:end].copy()).cuda()
        with torch.no_grad():
            clean_predictions = model(clean).argmax(1)
        if not torch.all(clean_predictions == labels):
            raise RuntimeError("manifest includes a batch-64 clean error")
        batch_started = time.perf_counter()
        adversarial_tm, displacement_tm = attack(
            clean.permute(1, 0, 2, 3, 4).contiguous(), labels,
            return_disp=True, use_PIL=True, use_cap=True, use_penalty=True, target_label=-1,
        )
        adversarial = adversarial_tm.permute(1, 0, 2, 3, 4).contiguous()
        displacement = displacement_tm.permute(1, 0, 2, 3, 4).contiguous()
        with torch.no_grad():
            adv_predictions = model(adversarial).argmax(1)
        adv_float = adversarial.cpu().numpy()
        if not np.all(np.isfinite(adv_float)) or np.any(adv_float < 0) or np.any(adv_float > np.iinfo(np.uint16).max) or not np.array_equal(adv_float, np.rint(adv_float)):
            raise RuntimeError("attack changed an integer packet amplitude")
        adv_np = adv_float.astype(np.uint16)
        disp_np = displacement.cpu().numpy().astype(np.int8)
        batch_realized = []
        for local, position in enumerate(range(batch_start, end)):
            source_t, line, target_t, values, realized = validate(
                clean_np[local].astype(np.uint16), adv_np[local], disp_np[local], kind, budget)
            label = int(selected_labels[position]); adv_prediction = int(adv_predictions[local])
            success = adv_prediction != label; successes += int(success)
            state["sample_ids"][0].append(int(selected_ids[position])); state["labels"][0].append(label)
            state["clean_predictions"][0].append(int(clean_predictions[local])); state["adv_predictions"][0].append(adv_prediction)
            state["source_t"][0].extend(source_t.tolist()); state["line"][0].extend(line.tolist())
            state["target_t"][0].extend(target_t.tolist()); state["value"][0].extend(values.tolist())
            state["offsets"][0].append(len(state["source_t"][0]))
            state["realized_b_inf"][0].append(realized[0]); state["realized_b1"][0].append(realized[1])
            state["realized_b0"][0].append(realized[2]); state["success"][0].append(success)
            batch_realized.append(realized)
        write_state(partial_path, state)
        values = np.asarray(batch_realized)
        elapsed = time.perf_counter() - started
        print(f"PHASE 2 {kind}={budget} completed={end}/1000 running_ASR={100*successes/end:.2f}% "
              f"realized_Binf={values[:,0].min()}..{values[:,0].max()} "
              f"realized_B1={values[:,1].min()}..{values[:,1].max()} "
              f"realized_B0={values[:,2].min()}..{values[:,2].max()} "
              f"batch_seconds={time.perf_counter()-batch_started:.1f} elapsed={elapsed:.1f}s", flush=True)
    artifact_sha = write_state(artifact_path, state)
    adapter_path = ROOT / "experiments/nmnist/official_pil_adapter_batched.py"
    metadata = {
        "run_id": run_id, "status": "AWAITING_INDEPENDENT_AUDIT", "dataset": "N-MNIST",
        "representation": "Integer", "temporal_bins": 10, "temporal_split": "equal-event-count",
        "seed": 42, "budget_type": kind, "requested_budget": budget,
        "clean_correct_denominator": 1000, "successful_attack_numerator_runner": successes,
        "asr_percent_runner": successes / 10.0, "attack_batch_size": 64,
        "attack_source_commit": UPSTREAM_COMMIT, "attack_source_sha256": UPSTREAM_SOURCE_SHA256,
        "adapter_path": str(adapter_path.relative_to(ROOT)).replace("\\", "/"),
        "adapter_sha256": sha256(adapter_path), "configuration_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "configuration_sha256": sha256(CONFIG), "checkpoint_path": clean_result["checkpoint_path"],
        "checkpoint_sha256": sha256(checkpoint_path), "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest_path), "clean_cache_path": clean_result["test_cache"]["path"],
        "clean_cache_sha256": sha256(cache_path), "artifact_path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": artifact_sha, "exact_command": " ".join([str(PYTHON), *sys.argv]),
        "runtime_seconds": time.perf_counter() - started,
        "implementation_note": "Exact upstream class; layout adapter plus independent per-example projection makes the upstream per-example budget semantics valid at required batch size 64. Upstream treats beta as an upper bound and does not force saturation.",
    }
    atomic_json(metadata_path, metadata)
    command = [str(PYTHON), "-u", "scripts/audit_nmnist_seed42_paper_aligned_integer_attacks.py",
               str(metadata_path.relative_to(ROOT)), "--output", str(audit_path.relative_to(ROOT))]
    subprocess.run(command, cwd=ROOT, check=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("independent audit failed")
    atomic_json(marker_path, {"status": "PASS", "run_id": run_id,
                              "metadata_sha256": sha256(metadata_path), "artifact_sha256": artifact_sha,
                              "audit_sha256": sha256(audit_path)})
    partial_path.unlink(missing_ok=True)
    print(f"PHASE 2 COMPLETE {kind}={budget} numerator={audit['successful_attacks']} "
          f"denominator={audit['clean_correct_denominator']} ASR={audit['asr_percent']:.2f}% AUDIT=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-type", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    run(args.budget_type, args.budget)


if __name__ == "__main__":
    main()
