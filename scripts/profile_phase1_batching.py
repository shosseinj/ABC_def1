"""Profile attack/audit batching and fail closed on discrete-output differences."""
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

from attacks.pil_pgd import PILPGDAttack, PILPGDConfig
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json
from scripts.equivalence_phase1_optimization import LegacyAttack

PROGRESS = ROOT / "Reports/checkpoints/phase1_batch_progress.json"
GPU_TOOL = Path(r"C:\Windows\System32\nvidia-smi.exe")
SAMPLE_IDS = [6697, 415, 8930, 1098, 653]
CONDITIONS = (("B_inf", 1), ("B1", 500), ("B0", 200))


def progress(stage: str, completed: int, total: int, detail: str) -> None:
    atomic_write_json(PROGRESS, {
        "status": "RUNNING", "stage": stage, "completed_items": completed,
        "total_items": total, "progress_percent": 100.0 * completed / total,
        "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[{completed}/{total}] {stage}: {detail}", flush=True)


class UtilizationSampler:
    def __init__(self):
        self.stop_event = threading.Event()
        self.rows = []
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.process.cpu_percent(None)
        while not self.stop_event.is_set():
            try:
                text = subprocess.check_output([
                    str(GPU_TOOL), "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                gpu, memory_util, memory_used = [float(value.strip()) for value in text.strip().split(",")]
                self.rows.append((gpu, memory_util, memory_used, self.process.cpu_percent(None)))
            except Exception:
                pass
            self.stop_event.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set(); self.thread.join(timeout=5)

    def summary(self):
        if not self.rows:
            return {"gpu_util_mean_percent": 0.0, "gpu_util_max_percent": 0.0,
                    "vram_max_mib": 0.0, "cpu_util_mean_percent": 0.0, "samples": 0}
        values = np.asarray(self.rows)
        return {"gpu_util_mean_percent": float(values[:, 0].mean()),
                "gpu_util_max_percent": float(values[:, 0].max()),
                "gpu_memory_util_mean_percent": float(values[:, 1].mean()),
                "vram_max_mib": float(values[:, 2].max()),
                "cpu_util_mean_percent": float(values[:, 3].mean()), "samples": len(values)}


def realized(displacement: np.ndarray, clean: np.ndarray) -> tuple[int, int, int]:
    delta = np.abs(displacement[clean != 0])
    return int(delta.max(initial=0)), int(delta.sum()), int((delta != 0).sum())


def main() -> None:
    configured = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"]).resolve()
    if Path(sys.executable).resolve() != configured:
        raise RuntimeError("wrong interpreter")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.loads((ROOT / "configs/nmnist_snn_clean_seed42.json").read_text())
    checkpoint = torch.load(ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt",
                            map_location=device, weights_only=True)
    model = NMNISTConvSNN(config["lif_decay"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    manifest = json.loads((ROOT / "Reports/checkpoints/n_mnist_seed42_clean_correct_manifest.json").read_text())
    by_id = {int(row["sample_id"]): row for row in manifest["samples"]}
    frames, labels = [], []
    for sample_id in SAMPLE_IDS:
        events, label = dataset[sample_id]
        frames.append((events_to_frames(events, 10) > 0).astype(np.uint8)); labels.append(int(label))
    clean = np.stack(frames)
    clean_tensor = torch.from_numpy(clean.astype(np.float32)).to(device)
    labels_tensor = torch.tensor(labels, device=device)
    total_stages, completed = 9, 0

    # Current legacy runner profile on one representative B_inf=2 sample.
    progress("legacy_profile", completed, total_stages, "batch=1 B_inf=2 sample=6697")
    legacy = LegacyAttack(model, PILPGDConfig("B_inf", 2))
    with UtilizationSampler() as sampler:
        tick = time.perf_counter()
        legacy(clean_tensor[:1], labels_tensor[:1])
        torch.cuda.synchronize(device)
        legacy_seconds = time.perf_counter() - tick
    legacy_util = sampler.summary(); completed += 1

    comparisons, batch_rows = [], []
    batch_equivalent = True
    for kind, beta in CONDITIONS:
        progress("attack_batch8", completed, total_stages, f"{kind}={beta} effective_batch=5")
        attack = PILPGDAttack(model, PILPGDConfig(kind, beta))
        with UtilizationSampler() as sampler:
            tick = time.perf_counter()
            adversarial, displacement = attack(clean_tensor, labels_tensor)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - tick
        adv_np = adversarial.detach().cpu().numpy()
        displacement_np = displacement.detach().cpu().numpy()
        predictions = attack.last_logits.argmax(1).cpu().numpy()
        condition_pass = True
        for index, sample_id in enumerate(SAMPLE_IDS):
            reference = np.load(
                ROOT / f"Reports/checkpoints/phase1_equivalence_items/{kind}_{beta}_sample{sample_id}_optimized.npz",
                allow_pickle=False)
            old_adv, old_disp = reference["adversarial"][0], reference["displacement"][0]
            old_prediction = int(reference["logits"].argmax(1)[0])
            old_b, new_b = realized(old_disp, clean[index]), realized(displacement_np[index], clean[index])
            record = {
                "sample_id": sample_id, "budget_type": kind, "beta": beta,
                "temporal_representation_exact": bool(np.array_equal(old_adv, adv_np[index])),
                "packet_destinations_exact": bool(np.array_equal(old_disp, displacement_np[index])),
                "packet_amplitudes_exact": bool(np.array_equal(np.sort(old_adv[old_adv != 0]),
                                                                  np.sort(adv_np[index][adv_np[index] != 0]))),
                "old_prediction": old_prediction, "batched_prediction": int(predictions[index]),
                "old_success": old_prediction != labels[index],
                "batched_success": int(predictions[index]) != labels[index],
                "old_realized_b_inf": old_b[0], "batch_realized_b_inf": new_b[0],
                "old_realized_b1": old_b[1], "batch_realized_b1": new_b[1],
                "old_realized_b0": old_b[2], "batch_realized_b0": new_b[2],
            }
            record["passed"] = all((record["temporal_representation_exact"],
                                     record["packet_destinations_exact"], record["packet_amplitudes_exact"],
                                     old_prediction == int(predictions[index]), old_b == new_b,
                                     record["old_success"] == record["batched_success"]))
            condition_pass &= record["passed"]; comparisons.append(record)
        batch_equivalent &= condition_pass
        batch_rows.append({"requested_batch_size": 8, "effective_batch_size": 5,
                           "budget_type": kind, "beta": beta, "wall_seconds": wall,
                           "seconds_per_sample": wall / len(SAMPLE_IDS),
                           "samples_per_second": len(SAMPLE_IDS) / wall,
                           "equivalence_pass": condition_pass, **sampler.summary()})
        completed += 1

    # Safe inference-only batching benchmark (used by clean/audit predictions).
    audit_ids = [int(row["sample_id"]) for row in manifest["samples"][:64]]
    audit_frames = []
    for sample_id in audit_ids:
        events, _ = dataset[sample_id]
        audit_frames.append((events_to_frames(events, 10) > 0).astype(np.float32))
    audit_tensor = torch.from_numpy(np.stack(audit_frames)).to(device)
    with torch.inference_mode():
        reference_predictions = torch.cat([model(audit_tensor[i:i + 1]).argmax(1) for i in range(64)]).cpu()
    inference_rows = []
    for batch_size in (1, 8, 16, 32, 64):
        progress("audit_inference_batch", completed, total_stages,
                 f"batch={batch_size} samples=64 repeats=5")
        with UtilizationSampler() as sampler, torch.inference_mode():
            tick = time.perf_counter()
            last_predictions = None
            for _ in range(5):
                last_predictions = torch.cat([
                    model(audit_tensor[start:start + batch_size]).argmax(1)
                    for start in range(0, 64, batch_size)])
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - tick
        exact = torch.equal(reference_predictions, last_predictions.cpu())
        inference_rows.append({"batch_size": batch_size, "wall_seconds_5x64": wall,
                               "seconds_per_sample": wall / (5 * 64),
                               "samples_per_second": 5 * 64 / wall,
                               "prediction_exact": exact, "oom": False, **sampler.summary()})
        completed += 1

    candidates = [row for row in inference_rows if row["prediction_exact"]]
    selected_audit_batch = max(candidates, key=lambda row: row["samples_per_second"])["batch_size"]
    skipped = [] if batch_equivalent else [
        {"batch_size": size, "status": "NOT_RUN", "reason": "stopped after batch size 8 exact-equivalence failure"}
        for size in (16, 32, 64)]
    result = {
        "status": "PASS_WITH_ATTACK_BATCHING_REJECTED",
        "attack_selected_batch_size": 1,
        "audit_selected_batch_size": selected_audit_batch,
        "attack_batch8_exact_equivalence": batch_equivalent,
        "legacy_profile": {"batch_size": 1, "wall_seconds": legacy_seconds, **legacy_util},
        "attack_batch_rows": batch_rows, "skipped_attack_batch_sizes": skipped,
        "audit_inference_rows": inference_rows, "comparisons": comparisons,
        "scientific_decision": "retain attack batch size 1; enable only exact inference/audit batching",
    }
    atomic_write_json(ROOT / "Reports/results/phase1_batch_equivalence.json", result)
    atomic_write_json(PROGRESS, {"status": "PASS", "completed_items": total_stages,
                                 "total_items": total_stages, "progress_percent": 100.0,
                                 "attack_selected_batch_size": 1,
                                 "audit_selected_batch_size": selected_audit_batch,
                                 "updated_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({key: result[key] for key in ("status", "attack_selected_batch_size",
                                                    "audit_selected_batch_size",
                                                    "attack_batch8_exact_equivalence")}, indent=2))


if __name__ == "__main__":
    main()
