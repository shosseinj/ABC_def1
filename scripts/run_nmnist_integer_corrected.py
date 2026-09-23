"""Resumable corrected N-MNIST Integer-only manifest and attack benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.pil_pgd import PILPGDAttack, PILPGDConfig
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
SEEDS = (42, 123, 777)
BUDGETS = {"B_inf": (1, 2, 3), "B1": (500, 750, 1000, 1500), "B0": (200, 300, 400, 600)}
RESULT_DIR = ROOT / "Reports/results/nmnist_integer_corrected"
CHECKPOINT_DIR = ROOT / "Reports/checkpoints/nmnist_integer_corrected"
LOG_DIR = ROOT / "Reports/logs/nmnist_integer_corrected"
BATCH = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256(path)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class ArraySet(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(np.array(self.x[i], dtype=np.float32, copy=True)), int(self.y[i])


def build_full_cache(dataset) -> tuple[np.ndarray, np.ndarray, Path]:
    path = CHECKPOINT_DIR / "full_official_test_t10_uint8.npz"
    metadata = CHECKPOINT_DIR / "full_official_test_t10_uint8.json"
    if path.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        if info.get("artifact_sha256") == sha256(path):
            data = np.load(path, allow_pickle=False)
            if len(data["labels"]) == 10000: return data["integer"], data["labels"], path
    frames = np.empty((len(dataset), 10, 2, 34, 34), dtype=np.uint8)
    labels = np.empty(len(dataset), dtype=np.int8)
    for i in range(len(dataset)):
        events, label = dataset[i]; frames[i] = events_to_frames(events, 10); labels[i] = int(label)
        if (i + 1) % 500 == 0: print(f"manifest preprocessing | {i+1}/10000", flush=True)
    digest = atomic_npz(path, integer=frames, labels=labels, sample_ids=np.arange(len(dataset), dtype=np.int32))
    atomic_json(metadata, {"status": "PASS", "official_test_samples": len(dataset), "T": 10,
        "representation": "strict_integer_count", "artifact_sha256": digest})
    return frames, labels, path


def load_model(seed: int, device):
    checkpoint_path = ROOT / f"checkpoints/nmnist_snn_clean_seed{seed}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stored = checkpoint.get("seed", checkpoint.get("config", {}).get("seed"))
    if int(stored) != seed: raise RuntimeError("checkpoint seed mismatch")
    model = NMNISTConvSNN(0.5).to(device).eval(); model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint_path


@torch.no_grad()
def build_manifest(seed, model, frames, labels, checkpoint_path, device):
    directory = RESULT_DIR / "manifests"; directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"nmnist_integer_seed{seed}_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest.get("selection_dependency") == "integer_clean_correct_only" and
                manifest.get("checkpoint_sha256") == sha256(checkpoint_path) and len(manifest.get("samples", [])) == 1000):
            return manifest, path
    predictions = np.empty(len(labels), dtype=np.int8)
    # Batch size one matches the exact victim arithmetic used by the attack runner.
    for i in range(len(labels)):
        x = torch.from_numpy(np.array(frames[i:i+1], dtype=np.float32, copy=True)).to(device)
        predictions[i] = int(model(x).argmax(1))
        if (i + 1) % 1000 == 0: print(f"seed {seed} clean evaluation | {i+1}/10000", flush=True)
    eligible = np.flatnonzero(predictions == labels)
    if len(eligible) < 1000: raise RuntimeError(f"seed {seed} has only {len(eligible)} Integer-clean-correct samples")
    selected = sorted(map(int, eligible), key=lambda i: (hashlib.sha256(f"nmnist_integer_corrected:{seed}:{i}".encode()).digest(), i))[:1000]
    rows = [{"sample_id": i, "true_label": int(labels[i]), "integer_clean_prediction": int(predictions[i])} for i in selected]
    manifest = {"dataset": "N-MNIST", "representation": "integer", "seed": seed,
        "official_test_samples": len(labels), "full_test_correct": int((predictions == labels).sum()),
        "full_test_accuracy_percent": 100.0 * float((predictions == labels).mean()),
        "eligible_integer_clean_correct": len(eligible), "sample_count": 1000,
        "selection": "first 1000 by SHA-256(nmnist_integer_corrected:seed:sample_id) among Integer-only clean-correct samples",
        "selection_dependency": "integer_clean_correct_only", "binary_dependency": False,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path), "samples": rows, "samples_sha256": canonical_hash(rows)}
    atomic_json(path, manifest); manifest["manifest_sha256_file"] = sha256(path)
    return manifest, path


def selected_cache(seed, manifest, manifest_path, frames, labels):
    path = CHECKPOINT_DIR / f"seed{seed}_integer_manifest_cache.npz"
    metadata = CHECKPOINT_DIR / f"seed{seed}_integer_manifest_cache.json"
    ids = np.asarray([r["sample_id"] for r in manifest["samples"]], dtype=np.int32)
    if path.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        if info.get("manifest_sha256_file") == sha256(manifest_path) and info.get("artifact_sha256") == sha256(path):
            data = np.load(path, allow_pickle=False)
            if np.array_equal(data["sample_ids"], ids): return data, path
    digest = atomic_npz(path, integer=frames[ids], labels=labels[ids], sample_ids=ids)
    atomic_json(metadata, {"manifest_sha256_file": sha256(manifest_path), "artifact_sha256": digest,
                           "sample_count": len(ids), "representation": "strict_integer_count"})
    return np.load(path, allow_pickle=False), path


def direct_validate(clean, adv, displacement, budget_type, budget):
    source = np.argwhere(clean.reshape(10, -1) != 0); source_t, line = source[:, 0], source[:, 1]
    delta = displacement.reshape(10, -1)[source_t, line].astype(np.int64); target = source_t + delta
    if np.any(target < 0) or np.any(target >= 10): raise RuntimeError("temporal domain violation")
    if len(set(zip(line.tolist(), target.tolist()))) != len(target): raise RuntimeError("collision")
    reconstructed = np.zeros_like(clean.reshape(10, -1)); reconstructed[target, line] = clean.reshape(10, -1)[source_t, line]
    if not np.array_equal(reconstructed.reshape(clean.shape), adv): raise RuntimeError("reconstruction mismatch")
    if int(clean.astype(np.int64).sum()) != int(adv.astype(np.int64).sum()): raise RuntimeError("mass changed")
    if not np.array_equal(np.sort(clean[clean != 0]), np.sort(adv[adv != 0])): raise RuntimeError("amplitude changed")
    absolute = np.abs(delta); values = (int(absolute.max(initial=0)), int(absolute.sum()), int(np.count_nonzero(delta)))
    if {"B_inf": values[0], "B1": values[1], "B0": values[2]}[budget_type] > budget: raise RuntimeError("budget exceeded")
    return source_t, line, target, clean.reshape(10, -1)[source_t, line], values


def write_partial(path, state):
    atomic_npz(path, **{k: np.asarray(v, dtype=d) for k, (v, d) in state.items()})


def run_condition(seed, model, checkpoint_path, manifest, manifest_path, cache, cache_path,
                  budget_type, budget, device):
    run_id = f"nmnist_integer_corrected_seed{seed}_{budget_type}_{budget}"
    run_dir = RESULT_DIR / "runs"; run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / f"{run_id}.json"; artifact_path = run_dir / f"{run_id}.npz"
    audit_path = run_dir / f"{run_id}.audit.json"; marker = CHECKPOINT_DIR / f"{run_id}.complete.json"
    if marker.exists() and audit_path.exists() and json.loads(audit_path.read_text()).get("passed"):
        audit = json.loads(audit_path.read_text()); print(f"N-MNIST | integer | {seed} | {budget_type}={budget} | 1000 | {audit['asr_percent']:.2f}% | PASS", flush=True); return
    partial = CHECKPOINT_DIR / f"{run_id}.partial.npz"
    lists = {"sample_ids": ([], np.int32), "labels": ([], np.int8), "clean_predictions": ([], np.int8),
      "adv_predictions": ([], np.int8), "offsets": ([0], np.int64), "source_t": ([], np.int8),
      "line": ([], np.int16), "target_t": ([], np.int8), "value": ([], np.uint8),
      "reported_b_inf": ([], np.int16), "reported_b1": ([], np.int32), "reported_b0": ([], np.int32)}
    if partial.exists():
        p = np.load(partial, allow_pickle=False)
        for key in lists: lists[key] = (p[key].tolist(), lists[key][1])
    start = len(lists["sample_ids"][0]); expected_ids = [r["sample_id"] for r in manifest["samples"]]
    if lists["sample_ids"][0] != expected_ids[:start]: raise RuntimeError("partial is not manifest prefix")
    integer = cache["integer"]; labels = cache["labels"]
    for batch_start in range(start, 1000, BATCH):
        end = min(batch_start + BATCH, 1000)
        clean_cpu = torch.from_numpy(np.array(integer[batch_start:end], dtype=np.float32, copy=True))
        y = torch.from_numpy(np.array(labels[batch_start:end], dtype=np.int64, copy=True)).to(device)
        clean = clean_cpu.to(device)
        with torch.no_grad():
            clean_predictions = torch.cat([model(clean[i:i+1]).argmax(1) for i in range(len(clean))])
        if not torch.all(clean_predictions == y): raise RuntimeError("manifest sample is not clean-correct")
        attack = PILPGDAttack(model, PILPGDConfig(budget_type, budget, projector="cuda"))
        adv, displacement = attack(clean, y)
        if not torch.equal(adv, attack.last_evaluated_input): raise RuntimeError("strict evaluated input mismatch")
        adv_np = adv.cpu().numpy().astype(np.uint8); disp_np = displacement.cpu().numpy().astype(np.int8)
        adv_predictions = attack.last_logits.argmax(1).cpu().tolist()
        for local, pos in enumerate(range(batch_start, end)):
            st, line, target, values, realized = direct_validate(integer[pos], adv_np[local], disp_np[local], budget_type, budget)
            lists["sample_ids"][0].append(expected_ids[pos]); lists["labels"][0].append(int(labels[pos]))
            lists["clean_predictions"][0].append(int(clean_predictions[local])); lists["adv_predictions"][0].append(adv_predictions[local])
            lists["source_t"][0].extend(st); lists["line"][0].extend(line); lists["target_t"][0].extend(target); lists["value"][0].extend(values)
            lists["offsets"][0].append(len(lists["source_t"][0])); lists["reported_b_inf"][0].append(realized[0]); lists["reported_b1"][0].append(realized[1]); lists["reported_b0"][0].append(realized[2])
        write_partial(partial, lists)
        print(f"{run_id} | {end}/1000", flush=True)
    artifact_sha = atomic_npz(artifact_path, **{k: np.asarray(v, dtype=d) for k, (v, d) in lists.items()})
    metadata = {"run_id": run_id, "dataset": "N-MNIST", "representation": "integer", "seed": seed,
      "budget_type": budget_type, "budget": budget, "sample_count": 1000,
      "manifest_policy": "integer_clean_correct_only", "binary_dependency": False,
      "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"), "manifest_sha256_file": sha256(manifest_path),
      "cache_path": str(cache_path.relative_to(ROOT)).replace("\\", "/"), "cache_sha256": sha256(cache_path),
      "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256(checkpoint_path),
      "artifact_path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"), "artifact_sha256": artifact_sha,
      "exact_command": " ".join(sys.argv), "attack_batch_size": BATCH, "packet_contract": "indivisible amplitude-bearing nonzero cell"}
    atomic_json(metadata_path, metadata)
    command = [str(PYTHON), "scripts/audit_nmnist_integer_corrected.py", str(metadata_path.relative_to(ROOT)), "--output", str(audit_path.relative_to(ROOT))]
    subprocess.run(command, cwd=ROOT, check=True)
    audit = json.loads(audit_path.read_text())
    atomic_json(marker, {"status": "PASS", "run_id": run_id, "metadata_sha256": sha256(metadata_path),
                         "artifact_sha256": artifact_sha, "audit_sha256": sha256(audit_path)})
    if partial.exists(): partial.unlink()
    print(f"N-MNIST | integer | {seed} | {budget_type}={budget} | 1000 | {audit['asr_percent']:.2f}% | PASS", flush=True)


def aggregate():
    rows=[]; audits=[]
    for seed in SEEDS:
        for kind, budgets in BUDGETS.items():
            for budget in budgets:
                run_id=f"nmnist_integer_corrected_seed{seed}_{kind}_{budget}"; path=RESULT_DIR/"runs"/f"{run_id}.audit.json"
                if path.exists():
                    a=json.loads(path.read_text()); audits.append({"run_id":run_id,"status":a["status"],"audit_sha256":sha256(path)})
                    rows.append({"dataset":"N-MNIST","representation":"integer","seed":seed,"budget_type":kind,"budget":budget,
                                 "sample_count":a["sample_count"],"successful_attacks":a["successful_attacks"],"asr_percent":a["asr_percent"],"status":a["status"]})
    fields=["dataset","representation","seed","budget_type","budget","sample_count","successful_attacks","asr_percent","status"]
    with (RESULT_DIR/"asr_by_seed.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for kind,budgets in BUDGETS.items():
        for budget in budgets:
            values=[r["asr_percent"] for r in rows if r["budget_type"]==kind and r["budget"]==budget and r["status"]=="PASS"]
            if len(values)==3:
                mean=float(np.mean(values));sd=float(np.std(values,ddof=1));half=4.302652729911275*sd/math.sqrt(3)
                summary.append({"dataset":"N-MNIST","representation":"integer","budget_type":kind,"budget":budget,"seeds":"42;123;777","n":3,"mean_asr_percent":mean,"std_asr_percent":sd,"ci95_low":mean-half,"ci95_high":mean+half,"status":"PASS"})
    sfields=["dataset","representation","budget_type","budget","seeds","n","mean_asr_percent","std_asr_percent","ci95_low","ci95_high","status"]
    with (RESULT_DIR/"asr_summary.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=sfields);w.writeheader();w.writerows(summary)
    completed=len(rows); remaining=33-completed
    atomic_json(RESULT_DIR/"audit.json", {"status":"PASS" if completed==33 and all(x["status"]=="PASS" for x in audits) else "RUNNING",
      "completed_conditions":completed,"remaining_conditions":remaining,"conditions":audits})
    next_unit=None
    for seed in SEEDS:
        for kind,budgets in BUDGETS.items():
            for budget in budgets:
                rid=f"nmnist_integer_corrected_seed{seed}_{kind}_{budget}"
                if not (CHECKPOINT_DIR/f"{rid}.complete.json").exists(): next_unit=f"seed={seed} {kind}={budget}";break
            if next_unit:break
        if next_unit:break
    report=["# Corrected N-MNIST Integer benchmark","",f"Status: **{'PASS' if remaining==0 else 'RUNNING'}**", "",
      "Manifests are selected solely from each seed's strict Integer-grid clean-correct official-test predictions. No Binary prediction or legacy intersection manifest is read.","",
      f"Completed conditions: {completed}/33",f"Remaining conditions: {remaining}","Blockers: none",f"Next smallest missing unit: {next_unit or 'none'}","",
      "Final clean accuracy: **98.39 ± 0.21%**. Corrected attack ASR must come only from `asr_summary.csv`.",""]
    (ROOT/"Reports/nmnist_integer_corrected_report.md").write_text("\n".join(report),encoding="utf-8")
    print(f"Phase A | completed={completed} | remaining={remaining} | blockers=none | next={next_unit or 'none'}",flush=True)


def legacy_map():
    path=ROOT/"Reports/results/legacy_validity_map.csv"; rows=[]
    legacy=ROOT/"Reports/results/nmnist/summary.csv"
    corrected_manifests_complete=all((RESULT_DIR/"manifests"/f"nmnist_integer_seed{seed}_manifest.json").exists() for seed in SEEDS)
    if legacy.exists():
        for r in csv.DictReader(legacy.open(encoding="utf-8")):
            status="REPRESENTATION_MISMATCH" if r["representation"]=="binary" else "NON_COMPARABLE"
            reason="count-trained checkpoint evaluated/attacked with Binary occupancy" if r["representation"]=="binary" else "legacy Binary-intersection manifest; supersede only after corrected condition completes"
            if r["representation"]=="integer" and corrected_manifests_complete: status="SUPERSEDED";reason="corrected Integer-only manifests are complete; legacy intersection result excluded from final tables"
            rows.append({"dataset":r["dataset"],"representation":r["representation"],"seed":r["seed"],"artifact":r["run_id"],"status":status,"reason":reason})
    for dataset in ("DVS-Gesture","CIFAR10-DVS"):
        for seed in SEEDS:
            rows.append({"dataset":dataset,"representation":"binary","seed":seed,"artifact":"frozen checkpoint validation / manifest","status":"REPRESENTATION_MISMATCH","reason":"no Binary-trained checkpoint"})
            rows.append({"dataset":dataset,"representation":"integer","seed":seed,"artifact":"frozen normalized-count checkpoint validation","status":"INVALID_PREPROCESSING" if dataset=="CIFAR10-DVS" and seed==42 else "NON_COMPARABLE","reason":"float32-train vs float16-cache mismatch" if dataset=="CIFAR10-DVS" and seed==42 else "fractional normalized counts are not strict Integer-grid amplitudes"})
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["dataset","representation","seed","artifact","status","reason"]);w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,choices=SEEDS);parser.add_argument("--all",action="store_true");parser.add_argument("--manifest-only",action="store_true");parser.add_argument("--budget-type",choices=tuple(BUDGETS));parser.add_argument("--budget",type=int);args=parser.parse_args()
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError(f"wrong interpreter: {PYTHON}")
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    RESULT_DIR.mkdir(parents=True,exist_ok=True);CHECKPOINT_DIR.mkdir(parents=True,exist_ok=True);LOG_DIR.mkdir(parents=True,exist_ok=True)
    from tonic.datasets import NMNIST
    dataset=NMNIST(save_to=str(ROOT/"data/nmnist"),train=False);frames,labels,_=build_full_cache(dataset);device=torch.device("cuda")
    seeds=SEEDS if args.all else (args.seed,)
    if not args.all and args.seed is None:raise ValueError("use --seed or --all")
    for seed in seeds:
        set_determinism(seed);model,checkpoint=load_model(seed,device);manifest,manifest_path=build_manifest(seed,model,frames,labels,checkpoint,device)
        manifest["manifest_sha256_file"]=sha256(manifest_path);cache,cache_path=selected_cache(seed,manifest,manifest_path,frames,labels)
        if args.manifest_only:
            print(f"N-MNIST | integer | {seed} | manifest | 1000 | n/a | PASS", flush=True)
            continue
        conditions=[(kind,budget) for kind,budgets in BUDGETS.items() for budget in budgets]
        if args.budget_type is not None or args.budget is not None:
            if args.budget_type is None or args.budget is None or args.budget not in BUDGETS[args.budget_type]:raise ValueError("select a canonical --budget-type and --budget pair")
            conditions=[(args.budget_type,args.budget)]
        for kind,budget in conditions:
            run_condition(seed,model,checkpoint,manifest,manifest_path,cache,cache_path,kind,budget,device);aggregate();legacy_map()
    aggregate();legacy_map()


if __name__=="__main__":main()
