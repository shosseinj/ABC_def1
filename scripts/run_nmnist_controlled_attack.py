"""Run one common-manifest attack for a locally controlled four-model comparison."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.controlled_models import build_model
from experiments.nmnist.official_pil_adapter_batched import build_attack, load_official_attack_module, UPSTREAM_COMMIT, UPSTREAM_SOURCE_SHA256
from scripts.run_nmnist_seed42_paper_aligned_integer_attacks import atomic_json, sha256, state_arrays, validate, write_state

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
OUT = ROOT / "Reports/results/nmnist_controlled_four_models_lr1e4/attacks"
STATE = ROOT / "Reports/checkpoints/nmnist_controlled_four_models_lr1e4"
BUDGETS = {"binary": {"B_inf": (1, 2, 3), "B1": (500, 750, 1000), "B0": (200, 300, 400)},
           "integer": {"B_inf": (1, 2, 3), "B1": (500, 750, 1000, 1500), "B0": (200, 300, 400, 600)}}


def run(representation: str, model_name: str, kind: str, budget: int) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    if budget not in BUDGETS[representation][kind]:
        raise ValueError("unsupported paper-table budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run_id = f"controlled_{representation}_{model_name}_{kind}_{budget}_seed42"
    result_dir = ROOT / "Reports/results/nmnist_controlled_four_models_lr1e4"
    custom_result = ROOT / f"Reports/results/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}/seed42_clean_result.json"
    model_result = result_dir / f"{representation}_{model_name}_seed42.json"
    result = json.loads(model_result.read_text(encoding="utf-8"))
    manifest_path = result_dir / f"{representation}_joint_clean_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or len(manifest["sample_ids"]) != 1000:
        raise RuntimeError("joint manifest invalid")
    if manifest["models"][model_name]["checkpoint_sha256"] != result["checkpoint_sha256"]:
        raise RuntimeError("model/manifest checkpoint mismatch")
    checkpoint_path = ROOT / result["checkpoint_path"]
    if sha256(checkpoint_path) != result["checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    cache_path = ROOT / result["test_cache"]["path"]
    if sha256(cache_path) != result["test_cache"]["sha256"]:
        raise RuntimeError("test cache hash mismatch")
    artifact = OUT / f"{run_id}.npz"
    metadata_path = OUT / f"{run_id}.json"
    audit_path = OUT / f"{run_id}.audit.json"
    marker = STATE / f"{run_id}.complete.json"
    partial = STATE / f"{run_id}.partial.npz"
    if marker.exists():
        done = json.loads(marker.read_text(encoding="utf-8"))
        if done["status"] == "PASS" and all(sha256(path) == done[key] for path, key in
            ((artifact, "artifact_sha256"), (metadata_path, "metadata_sha256"), (audit_path, "audit_sha256"))):
            print(f"SKIP {run_id}: completion hashes and independent audit PASS", flush=True)
            return
        raise RuntimeError("existing completion marker is not hash-valid")
    model = build_model(model_name).cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    attack = build_attack(load_official_attack_module(), model, torch.device("cuda"), kind, budget)
    frames = np.load(cache_path, mmap_mode="r")
    ids = np.asarray(manifest["sample_ids"], dtype=np.int64)
    labels = np.asarray(manifest["labels"], dtype=np.int64)
    state = state_arrays()
    if partial.exists():
        with np.load(partial, allow_pickle=False) as prior:
            for key, (_, dtype) in state.items():
                state[key] = (prior[key].astype(dtype).tolist(), dtype)
    start = len(state["sample_ids"][0])
    if state["sample_ids"][0] != ids[:start].tolist():
        raise RuntimeError("partial state is not joint-manifest prefix")
    successes = int(sum(state["success"][0]))
    started = time.perf_counter()
    print(f"START {run_id} resume={start}/1000 batch=64", flush=True)
    for begin in range(start, 1000, 64):
        end = min(begin + 64, 1000)
        clean_np = np.array(frames[ids[begin:end]], dtype=np.float32, copy=True)
        clean = torch.from_numpy(clean_np).cuda()
        target = torch.from_numpy(labels[begin:end].copy()).cuda()
        with torch.no_grad():
            clean_pred = model(clean).argmax(1)
        if not torch.all(clean_pred == target):
            raise RuntimeError("joint manifest contains a clean error for this model")
        adv_tm, disp_tm = attack(clean.permute(1, 0, 2, 3, 4).contiguous(), target,
                                 return_disp=True, use_PIL=True, use_cap=True, use_penalty=True, target_label=-1)
        adv = adv_tm.permute(1, 0, 2, 3, 4).contiguous()
        disp = disp_tm.permute(1, 0, 2, 3, 4).contiguous()
        with torch.no_grad():
            adv_pred = model(adv).argmax(1)
        adv_float = adv.cpu().numpy()
        if not np.all(np.isfinite(adv_float)) or np.any(adv_float < 0) or np.any(adv_float > 65535) or not np.array_equal(adv_float, np.rint(adv_float)):
            raise RuntimeError("attack changed integer packet amplitude")
        adv_np = adv_float.astype(np.uint16)
        disp_np = disp.cpu().numpy().astype(np.int8)
        for local, position in enumerate(range(begin, end)):
            source_t, line, target_t, values, realized = validate(clean_np[local].astype(np.uint16), adv_np[local], disp_np[local], kind, budget)
            success = int(adv_pred[local]) != int(labels[position]); successes += int(success)
            state["sample_ids"][0].append(int(ids[position])); state["labels"][0].append(int(labels[position]))
            state["clean_predictions"][0].append(int(clean_pred[local])); state["adv_predictions"][0].append(int(adv_pred[local]))
            state["source_t"][0].extend(source_t.tolist()); state["line"][0].extend(line.tolist())
            state["target_t"][0].extend(target_t.tolist()); state["value"][0].extend(values.tolist())
            state["offsets"][0].append(len(state["source_t"][0]))
            state["realized_b_inf"][0].append(realized[0]); state["realized_b1"][0].append(realized[1]); state["realized_b0"][0].append(realized[2]); state["success"][0].append(success)
        write_state(partial, state)
        print(f"{run_id}: {end}/1000 ASR_so_far={100*successes/end:.2f}% elapsed={time.perf_counter()-started:.1f}s", flush=True)
    artifact_hash = write_state(artifact, state)
    adapter = ROOT / "experiments/nmnist/official_pil_adapter_batched.py"
    config = ROOT / f"configs/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}_attack.json"
    metadata = {"run_id": run_id, "status": "AWAITING_INDEPENDENT_AUDIT", "dataset": "N-MNIST", "representation": representation,
                "model": model_name, "seed": 42, "budget_type": kind, "requested_budget": budget, "temporal_bins": 10,
                "clean_correct_denominator": 1000, "successful_attack_numerator_runner": successes,
                "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "manifest_sha256": sha256(manifest_path),
                "clean_result_path": str(model_result.relative_to(ROOT)).replace("\\", "/"), "clean_result_sha256": sha256(model_result),
                "model_source_path": result["model_source_path"], "model_source_sha256": result["model_source_sha256"],
                "checkpoint_path": result["checkpoint_path"], "checkpoint_sha256": sha256(checkpoint_path),
                "clean_cache_path": result["test_cache"]["path"], "clean_cache_sha256": sha256(cache_path),
                "artifact_path": str(artifact.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": artifact_hash,
                "adapter_path": str(adapter.relative_to(ROOT)).replace("\\", "/"), "adapter_sha256": sha256(adapter),
                "configuration_path": str(config.relative_to(ROOT)).replace("\\", "/"), "configuration_sha256": sha256(config),
                "attack_source_commit": UPSTREAM_COMMIT, "attack_source_sha256": UPSTREAM_SOURCE_SHA256,
                "exact_command": " ".join([str(PYTHON), *sys.argv]), "runtime_seconds": time.perf_counter() - started}
    atomic_json(metadata_path, metadata)
    subprocess.run([str(PYTHON), "-u", "scripts/audit_nmnist_controlled_attack.py", str(metadata_path.relative_to(ROOT)), "--output", str(audit_path.relative_to(ROOT))], cwd=ROOT, check=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("independent audit failed")
    atomic_json(marker, {"status": "PASS", "run_id": run_id, "metadata_sha256": sha256(metadata_path), "artifact_sha256": artifact_hash, "audit_sha256": sha256(audit_path)})
    partial.unlink(missing_ok=True)
    print(f"COMPLETE {run_id} ASR={audit['asr_percent']:.2f}% AUDIT=PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=tuple(BUDGETS), required=True)
    parser.add_argument("--model", choices=("custom", "convnet", "resnet18", "vggsnn"), required=True)
    parser.add_argument("--budget-type", choices=("B_inf", "B1", "B0"), required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    run(args.representation, args.model, args.budget_type, args.budget)
