"""Validate only attack batch 64 against frozen scalar-runner evidence."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.pil_pgd import (PILPGDAttack, PILPGDConfig,
                             strict_project_grid_cuda)
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json, sha256_file

PYTHON = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"])
BATCH_SIZE = 64
SAMPLE_IDS = [6697, 415, 8930, 1098, 653]
CONDITIONS = (("B_inf", 1), ("B1", 500), ("B0", 200))
RESULT = ROOT / "Reports/results/phase1_gpu_vectorization_equivalence.json"
ARTIFACT = ROOT / "Reports/results/phase1_gpu_vectorization_records.npz"
AUDIT = ROOT / "Reports/results/phase1_gpu_vectorization_audit.json"
STAGING = ROOT / "Reports/checkpoints/phase1_gpu_vectorization_staging.json"
TRACE = ROOT / "Reports/logs/phase1_gpu_vectorization_trace.json"
PROGRESS = ROOT / "Reports/checkpoints/phase1_gpu_vectorization_progress.json"
GPU_TOOL = Path(r"C:\Windows\System32\nvidia-smi.exe")


def atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def update(stage: str, completed: int, total: int, detail: str) -> None:
    atomic_write_json(PROGRESS, {
        "status": "RUNNING", "tested_attack_batch_size": BATCH_SIZE,
        "tested_audit_batch_size": BATCH_SIZE, "stage": stage,
        "completed": completed, "total": total,
        "progress_percent": 100.0 * completed / total, "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[{completed}/{total}] {stage}: {detail}", flush=True)


class UtilizationSampler:
    def __init__(self):
        self.stop = threading.Event(); self.rows = []
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.process.cpu_percent(None)
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output([
                    str(GPU_TOOL), "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                gpu, vram = (float(value.strip()) for value in raw.strip().split(","))
                self.rows.append((gpu, vram, self.process.cpu_percent(None)))
            except Exception:
                pass
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join(timeout=5)

    def summary(self):
        values = np.asarray(self.rows, dtype=float)
        if not len(values):
            return {"gpu_utilization_mean_percent": None,
                    "gpu_utilization_max_percent": None,
                    "vram_max_mib": None, "cpu_utilization_mean_percent": None}
        return {"gpu_utilization_mean_percent": float(values[:, 0].mean()),
                "gpu_utilization_max_percent": float(values[:, 0].max()),
                "vram_max_mib": float(values[:, 1].max()),
                "cpu_utilization_mean_percent": float(values[:, 2].mean())}


def realized(displacement: np.ndarray, clean: np.ndarray):
    absolute = np.abs(displacement[clean != 0])
    return (int(absolute.max(initial=0)), int(absolute.sum()),
            int((absolute != 0).sum()))


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("wrong interpreter")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    config = json.loads((ROOT / "configs/nmnist_snn_clean_seed42.json").read_text())
    checkpoint_path = ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NMNISTConvSNN(config["lif_decay"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)

    manifest = json.loads((ROOT / "Reports/checkpoints/n_mnist_seed42_clean_correct_manifest.json").read_text())
    cache_path = ROOT / "Reports/checkpoints/nmnist_preprocessed/seed42_t10_uint8.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        cache_ids = cached["sample_ids"].astype(int)
        integer = cached["integer"]
        labels_cache = cached["labels"].astype(int)
        positions = {sample_id: index for index, sample_id in enumerate(cache_ids.tolist())}
        base_clean = np.stack([(integer[positions[sample_id]] != 0).astype(np.uint8)
                               for sample_id in SAMPLE_IDS])
        base_labels = np.asarray([labels_cache[positions[sample_id]]
                                  for sample_id in SAMPLE_IDS], dtype=np.int8)
    repeat_index = np.arange(BATCH_SIZE) % len(SAMPLE_IDS)
    clean = base_clean[repeat_index]
    labels = base_labels[repeat_index]
    sample_ids = np.asarray(SAMPLE_IDS, dtype=np.int32)[repeat_index]
    clean_tensor = torch.from_numpy(clean.astype(np.float32)).to(device)
    labels_tensor = torch.from_numpy(labels.astype(np.int64)).to(device)

    # Compile the fixed batch-64 projector before timing; no alternate batch is run.
    warm_probabilities = torch.full((*clean_tensor.shape, 3), 1 / 3,
                                    device=device, dtype=torch.float32)
    strict_project_grid_cuda(clean_tensor, warm_probabilities, "B_inf", 1)
    torch.cuda.synchronize(device)

    all_clean, all_adv, all_disp, all_ref_adv, all_ref_disp = [], [], [], [], []
    all_ids, all_labels, all_kinds, all_betas = [], [], [], []
    all_predictions, all_ref_predictions = [], []
    comparisons, performance = [], []
    total = len(CONDITIONS) + 2
    completed = 0
    for kind, beta in CONDITIONS:
        update("batch64_attack", completed, total, f"{kind}={beta}")
        torch.cuda.reset_peak_memory_stats(device)
        attack = PILPGDAttack(model, PILPGDConfig(kind, beta, projector="cuda"))
        with UtilizationSampler() as sampler:
            torch.cuda.synchronize(device); started = time.perf_counter()
            adversarial, displacement = attack(clean_tensor, labels_tensor)
            torch.cuda.synchronize(device); wall = time.perf_counter() - started
        adv_np = adversarial.detach().cpu().numpy().astype(np.uint8)
        disp_np = displacement.detach().cpu().numpy().astype(np.int8)
        predictions = attack.last_logits.argmax(1).cpu().numpy().astype(np.int8)
        ref_adv, ref_disp, ref_pred, scalar_times = [], [], [], []
        for sample_id in sample_ids.tolist():
            stem = f"{kind}_{beta}_sample{sample_id}_optimized"
            with np.load(ROOT / f"Reports/checkpoints/phase1_equivalence_items/{stem}.npz",
                         allow_pickle=False) as reference:
                ref_adv.append(reference["adversarial"][0].astype(np.uint8))
                ref_disp.append(reference["displacement"][0].astype(np.int8))
                ref_pred.append(int(reference["logits"].argmax(1)[0]))
            scalar_times.append(float(json.loads((ROOT / f"Reports/checkpoints/phase1_equivalence_items/{stem}.json").read_text())["elapsed_seconds"]))
        ref_adv = np.stack(ref_adv); ref_disp = np.stack(ref_disp)
        ref_pred = np.asarray(ref_pred, dtype=np.int8)
        condition_pass = True
        for index in range(BATCH_SIZE):
            old_b = realized(ref_disp[index], clean[index])
            new_b = realized(disp_np[index], clean[index])
            row = {
                "sample_index": index, "sample_id": int(sample_ids[index]),
                "budget_type": kind, "beta": beta,
                "prediction_exact": int(predictions[index]) == int(ref_pred[index]),
                "success_flag_exact": ((int(predictions[index]) != int(labels[index])) ==
                                       (int(ref_pred[index]) != int(labels[index]))),
                "realized_b_inf_exact": old_b[0] == new_b[0],
                "realized_b1_exact": old_b[1] == new_b[1],
                "realized_b0_exact": old_b[2] == new_b[2],
                "projected_representation_exact": bool(np.array_equal(adv_np[index], ref_adv[index])),
                "packet_amplitudes_exact": bool(np.array_equal(
                    np.sort(adv_np[index][adv_np[index] != 0]),
                    np.sort(ref_adv[index][ref_adv[index] != 0]))),
                "packet_destinations_exact": bool(np.array_equal(disp_np[index], ref_disp[index])),
            }
            row["passed"] = all(value for key, value in row.items() if key.endswith("_exact"))
            condition_pass &= row["passed"]; comparisons.append(row)
        scalar_seconds_per_sample = float(np.mean(scalar_times))
        metrics = {
            "budget_type": kind, "beta": beta, "batch_size": BATCH_SIZE,
            "wall_seconds": wall, "seconds_per_sample": wall / BATCH_SIZE,
            "samples_per_second": BATCH_SIZE / wall,
            "scalar_seconds_per_sample": scalar_seconds_per_sample,
            "speedup_vs_scalar": scalar_seconds_per_sample / (wall / BATCH_SIZE),
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1048576,
            "exact_equivalence": condition_pass, **sampler.summary(),
        }
        performance.append(metrics)
        all_clean.extend(clean); all_adv.extend(adv_np); all_disp.extend(disp_np)
        all_ref_adv.extend(ref_adv); all_ref_disp.extend(ref_disp)
        all_ids.extend(sample_ids); all_labels.extend(labels)
        all_kinds.extend([kind] * BATCH_SIZE); all_betas.extend([beta] * BATCH_SIZE)
        all_predictions.extend(predictions); all_ref_predictions.extend(ref_pred)
        completed += 1

    update("cuda_profiler", completed, total, "batch64 strict projector")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as profiler:
        strict_project_grid_cuda(clean_tensor, warm_probabilities, "B_inf", 1)
        torch.cuda.synchronize(device)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(TRACE))
    profiler_table = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30)
    key_averages = profiler.key_averages()
    required_ops = ("aten::sort", "aten::gather", "aten::argsort")
    cuda_time_by_op = {
        name: float(sum(event.self_device_time_total for event in key_averages
                        if event.key == name)) for name in required_ops
    }
    profiler_evidence = {
        "trace_path": str(TRACE.relative_to(ROOT)), "trace_sha256": sha256_file(TRACE),
        "profiled_tensor_device": str(clean_tensor.device),
        "cuda_time_us_by_vectorized_op": cuda_time_by_op,
        "key_averages_table": profiler_table,
        "vectorized_cuda_ops_required": list(required_ops),
        "vectorized_cuda_ops_observed": [name for name in required_ops
                                         if cuda_time_by_op[name] > 0],
    }
    completed += 1

    artifact_sha = atomic_npz(
        ARTIFACT, sample_ids=np.asarray(all_ids, dtype=np.int32),
        labels=np.asarray(all_labels, dtype=np.int8),
        budget_type=np.asarray(all_kinds), beta=np.asarray(all_betas, dtype=np.int32),
        clean=np.asarray(all_clean, dtype=np.uint8), adversarial=np.asarray(all_adv, dtype=np.uint8),
        displacement=np.asarray(all_disp, dtype=np.int8),
        reference_adversarial=np.asarray(all_ref_adv, dtype=np.uint8),
        reference_displacement=np.asarray(all_ref_disp, dtype=np.int8),
        prediction=np.asarray(all_predictions, dtype=np.int8),
        reference_prediction=np.asarray(all_ref_predictions, dtype=np.int8))
    atomic_write_json(STAGING, {
        "status": "ATTACKS_AND_PROFILER_COMPLETE", "performance": performance,
        "comparisons": comparisons, "profiler_evidence": profiler_evidence,
        "artifact_path": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": artifact_sha,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    update("independent_audit", completed, total, "192 records")
    subprocess.run([str(PYTHON), "-m", "ResearchLoop.tools.audit_gpu_vectorization",
                    str(ARTIFACT.relative_to(ROOT)), "--output", str(AUDIT.relative_to(ROOT))],
                   cwd=ROOT, check=True)
    audit = json.loads(AUDIT.read_text())
    completed += 1
    exact = all(row["passed"] for row in comparisons)
    speedup = all(row["speedup_vs_scalar"] > 1 for row in performance)
    profiler_pass = (profiler_evidence["profiled_tensor_device"].startswith("cuda") and
                     len(profiler_evidence["vectorized_cuda_ops_observed"]) == 3)
    passed = exact and speedup and audit["passed"] and profiler_pass
    result = {
        "status": "PASS" if passed else "FAIL", "attack_batch_size_tested": 64,
        "audit_batch_size": 64, "other_attack_batch_sizes_tested": [],
        "seed": 42, "representation": "binary", "sample_count_per_condition": 64,
        "unique_sample_ids": SAMPLE_IDS, "conditions": [list(item) for item in CONDITIONS],
        "exact_equivalence_pass": exact, "regression_tests_pass": None,
        "independent_audit_pass": audit["passed"], "measured_speedup_pass": speedup,
        "profiler_pass": profiler_pass, "performance": performance,
        "comparisons": comparisons, "profiler_evidence": profiler_evidence,
        "artifact_path": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": artifact_sha,
        "audit_path": str(AUDIT.relative_to(ROOT)), "audit_sha256": sha256_file(AUDIT),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "implementation_note": "Projection/candidate generation is batch-64 CUDA-vectorized. Victim forwards remain independent batch-1 calls inside the 64-sample container because true convolution batching changed gradient signs and failed exact packet-destination equality.",
    }
    atomic_write_json(RESULT, result)
    atomic_write_json(PROGRESS, {"status": result["status"], "completed": total,
                                 "total": total, "progress_percent": 100.0,
                                 "tested_attack_batch_size": 64,
                                 "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": result["status"], "exact": exact, "speedup": speedup,
                      "audit": audit["passed"], "profiler": profiler_pass,
                      "performance": performance}, indent=2), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
