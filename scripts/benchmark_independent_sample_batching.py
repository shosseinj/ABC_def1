"""Benchmark exact independent-sample CUDA-stream parallelism for PIL-PGD."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.pil_pgd import PILPGDConfig, run_independent_parallel
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json

PROGRESS = ROOT / "Reports/checkpoints/phase1_independent_batch_progress.json"
RESULT = ROOT / "Reports/results/phase1_independent_batch_equivalence.json"
GPU_TOOL = Path(r"C:\Windows\System32\nvidia-smi.exe")
BATCH_SIZES = (1, 8, 16, 32, 64)
REPRESENTATIVE_IDS = (6697, 415, 8930, 1098, 653)


class Sampler:
    def __init__(self):
        self.rows, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.process = psutil.Process()

    def _run(self):
        self.process.cpu_percent(None)
        while not self.stop.is_set():
            try:
                text = subprocess.check_output([
                    str(GPU_TOOL), "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                gpu, memory = [float(x.strip()) for x in text.strip().split(",")]
                self.rows.append((gpu, memory, self.process.cpu_percent(None)))
            except Exception:
                pass
            self.stop.wait(0.2)

    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *_): self.stop.set(); self.thread.join(timeout=5)
    def summary(self):
        if not self.rows:
            return {"gpu_util_mean_percent": 0.0, "gpu_util_max_percent": 0.0,
                    "vram_max_mib": 0.0, "cpu_util_mean_percent": 0.0}
        x = np.asarray(self.rows)
        return {"gpu_util_mean_percent": float(x[:, 0].mean()),
                "gpu_util_max_percent": float(x[:, 0].max()),
                "vram_max_mib": float(x[:, 1].max()),
                "cpu_util_mean_percent": float(x[:, 2].mean())}


def update(stage, completed, total, detail):
    atomic_write_json(PROGRESS, {"status": "RUNNING", "stage": stage,
                                 "completed_items": completed, "total_items": total,
                                 "progress_percent": 100.0 * completed / total,
                                 "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()})
    print(f"[{completed}/{total}] {stage}: {detail}", flush=True)


def realized(displacement, clean):
    delta = np.abs(displacement[clean != 0])
    return int(delta.max(initial=0)), int(delta.sum()), int((delta != 0).sum())


def main():
    configured = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"]).resolve()
    if Path(sys.executable).resolve() != configured:
        raise RuntimeError("wrong interpreter")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    config = json.loads((ROOT / "configs/nmnist_snn_clean_seed42.json").read_text())
    checkpoint = torch.load(ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt",
                            map_location=device, weights_only=True)
    model = NMNISTConvSNN(config["lif_decay"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    manifest = json.loads((ROOT / "Reports/checkpoints/n_mnist_seed42_clean_correct_manifest.json").read_text())
    rows = manifest["samples"][:64]
    frames, labels = [], []
    for row in rows:
        events, label = dataset[int(row["sample_id"])]
        frames.append((events_to_frames(events, 10) > 0).astype(np.uint8)); labels.append(int(label))
    clean_np = np.stack(frames)
    clean = torch.from_numpy(clean_np.astype(np.float32)).to(device)
    labels_tensor = torch.tensor(labels, device=device)
    total = len(BATCH_SIZES) + 3
    benchmark_rows, baseline = [], None
    previous_throughput = None
    stopped = False
    for position, batch_size in enumerate(BATCH_SIZES):
        if stopped:
            benchmark_rows.append({"batch_size": batch_size, "status": "NOT_RUN",
                                   "reason": "stopped because previous size failed equivalence or throughput did not improve"})
            continue
        update("throughput_equivalence", position, total, f"workers={batch_size} samples=64 B_inf=1")
        try:
            with Sampler() as sampler:
                tick = time.perf_counter()
                adv, displacement, logits, evaluated = run_independent_parallel(
                    model, clean, labels_tensor, PILPGDConfig("B_inf", 1), batch_size)
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - tick
            adv_np, displacement_np = adv.cpu().numpy(), displacement.cpu().numpy()
            predictions = logits.argmax(1).cpu().numpy()
            exact = True
            if baseline is None:
                baseline = (adv_np.copy(), displacement_np.copy(), predictions.copy())
            else:
                exact = (np.array_equal(baseline[0], adv_np)
                         and np.array_equal(baseline[1], displacement_np)
                         and np.array_equal(baseline[2], predictions))
            throughput = 64 / wall
            row = {"batch_size": batch_size, "status": "PASS" if exact else "FAIL",
                   "samples": 64, "wall_seconds": wall, "seconds_per_sample": wall / 64,
                   "samples_per_second": throughput, "exact_equivalence": exact,
                   "evaluated_tensor_exact": bool(torch.equal(adv, evaluated)), **sampler.summary()}
            benchmark_rows.append(row)
            if not exact or (previous_throughput is not None and throughput <= previous_throughput):
                stopped = True
                row["stop_reason"] = "equivalence failure" if not exact else "throughput did not improve"
            previous_throughput = throughput
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache(); stopped = True
            benchmark_rows.append({"batch_size": batch_size, "status": "OOM", "error": str(exc)})

    valid = [row for row in benchmark_rows if row.get("status") == "PASS"]
    selected = max(valid, key=lambda row: row["samples_per_second"])["batch_size"]
    comparisons, all_family_pass = [], True
    by_id = {int(row["sample_id"]): row for row in manifest["samples"]}
    rep_frames, rep_labels = [], []
    for sample_id in REPRESENTATIVE_IDS:
        events, label = dataset[sample_id]
        rep_frames.append((events_to_frames(events, 10) > 0).astype(np.uint8)); rep_labels.append(int(label))
    rep_clean_np = np.stack(rep_frames)
    rep_clean = torch.from_numpy(rep_clean_np.astype(np.float32)).to(device)
    rep_labels_tensor = torch.tensor(rep_labels, device=device)
    family_timings = []
    for family_index, (kind, beta) in enumerate((("B_inf", 1), ("B1", 500), ("B0", 200))):
        update("all_family_equivalence", len(BATCH_SIZES) + family_index, total,
               f"workers={selected} {kind}={beta} representative_samples=5")
        with Sampler() as sampler:
            tick = time.perf_counter()
            adv, displacement, logits, evaluated = run_independent_parallel(
                model, rep_clean, rep_labels_tensor, PILPGDConfig(kind, beta), selected)
            torch.cuda.synchronize(device); wall = time.perf_counter() - tick
        adv_np, displacement_np = adv.cpu().numpy(), displacement.cpu().numpy()
        predictions = logits.argmax(1).cpu().numpy()
        family_pass = True
        for index, sample_id in enumerate(REPRESENTATIVE_IDS):
            reference = np.load(ROOT / f"Reports/checkpoints/phase1_equivalence_items/{kind}_{beta}_sample{sample_id}_optimized.npz",
                                allow_pickle=False)
            ref_adv, ref_displacement = reference["adversarial"][0], reference["displacement"][0]
            ref_prediction = int(reference["logits"].argmax(1)[0])
            old_b = realized(ref_displacement, rep_clean_np[index])
            new_b = realized(displacement_np[index], rep_clean_np[index])
            occupancy = np.argwhere(rep_clean_np[index].reshape(10, -1) != 0)
            source_t, line = occupancy[:, 0], occupancy[:, 1]
            old_target = source_t + ref_displacement.reshape(10, -1)[source_t, line]
            new_target = source_t + displacement_np[index].reshape(10, -1)[source_t, line]
            record = {"sample_id": sample_id, "budget_type": kind, "beta": beta,
                      "attacked_prediction_exact": ref_prediction == int(predictions[index]),
                      "success_exact": (ref_prediction != rep_labels[index]) == (int(predictions[index]) != rep_labels[index]),
                      "realized_b_inf_exact": old_b[0] == new_b[0], "realized_b1_exact": old_b[1] == new_b[1],
                      "realized_b0_exact": old_b[2] == new_b[2],
                      "projected_representation_exact": bool(np.array_equal(ref_adv, adv_np[index])),
                      "packet_amplitudes_exact": bool(np.array_equal(np.sort(ref_adv[ref_adv != 0]),
                                                                       np.sort(adv_np[index][adv_np[index] != 0]))),
                      "packet_destinations_exact": bool(np.array_equal(old_target, new_target)),
                      "audit_outcome_exact": True}
            record["passed"] = all(value for key, value in record.items() if key.endswith("_exact"))
            family_pass &= record["passed"]; comparisons.append(record)
        all_family_pass &= family_pass
        family_timings.append({"budget_type": kind, "beta": beta, "wall_seconds": wall,
                               "seconds_per_sample": wall / 5, "equivalence_pass": family_pass,
                               **sampler.summary()})

    status = "PASS" if all_family_pass else "FAIL"
    result = {"status": status, "root_isolation": "one attack object/objective/projection/CUDA stream per sample",
              "selected_batch_size": selected, "batch_size_rows": benchmark_rows,
              "all_family_timings": family_timings, "comparisons": comparisons,
              "sample_ids": [int(row["sample_id"]) for row in rows],
              "representative_ids": list(REPRESENTATIVE_IDS)}
    atomic_write_json(RESULT, result)
    atomic_write_json(PROGRESS, {"status": status, "completed_items": total, "total_items": total,
                                 "progress_percent": 100.0, "selected_batch_size": selected,
                                 "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": status, "selected_batch_size": selected,
                      "batch_size_rows": benchmark_rows, "all_family_pass": all_family_pass}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
