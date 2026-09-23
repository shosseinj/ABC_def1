"""Component wall-time profile for the unoptimized Phase 1 sample path."""
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

import attacks.pil_pgd as pil_module
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN
from ResearchLoop.core.contract import atomic_write_json

PYTHON = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"])


class TimedModel(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.seconds = 0.0

    def forward(self, value):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = self.model(value)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds += time.perf_counter() - started
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="old")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("wrong interpreter")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tonic.datasets import NMNIST
    started = time.perf_counter()
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    manifest = json.loads((ROOT / "Reports/checkpoints/n_mnist_seed42_clean_correct_manifest.json").read_text())
    sample_id = int(manifest["samples"][0]["sample_id"])
    events, label = dataset[sample_id]
    clean = (events_to_frames(events, 10) > 0).astype(np.uint8)
    preprocessing_seconds = time.perf_counter() - started

    config = json.loads((ROOT / "configs/nmnist_snn_clean_seed42.json").read_text())
    checkpoint = torch.load(ROOT / "checkpoints/nmnist_snn_clean_seed42_best.pt",
                            map_location=device, weights_only=True)
    base_model = NMNISTConvSNN(config["lif_decay"]).to(device).eval()
    base_model.load_state_dict(checkpoint["model_state"], strict=True)
    model = TimedModel(base_model, device)

    projection_seconds = 0.0
    original_project = pil_module.strict_project_grid

    def timed_project(*project_args, **project_kwargs):
        nonlocal projection_seconds
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        result = original_project(*project_args, **project_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        projection_seconds += time.perf_counter() - tick
        return result

    pil_module.strict_project_grid = timed_project
    attack = pil_module.PILPGDAttack(model, pil_module.PILPGDConfig("B_inf", 1))
    clean_tensor = torch.from_numpy(clean[None].astype(np.float32)).to(device)
    label_tensor = torch.tensor([int(label)], device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tick = time.perf_counter()
    adversarial, displacement = attack(clean_tensor, label_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    attack_total = time.perf_counter() - tick
    pil_module.strict_project_grid = original_project
    optimization_seconds = attack_total - model.seconds - projection_seconds

    tick = time.perf_counter()
    profile_dir = ROOT / "Reports/checkpoints/runtime_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    serialization_path = profile_dir / f"{args.label}_serialization.npz"
    temporary = serialization_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, clean=clean, adversarial=adversarial.detach().cpu().numpy(),
                            displacement=displacement.detach().cpu().numpy())
    os.replace(temporary, serialization_path)
    serialization_seconds = time.perf_counter() - tick

    tick = time.perf_counter()
    log_path = profile_dir / f"{args.label}_logging.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"sample=1 id={sample_id} profile={args.label}\n")
        handle.flush()
        os.fsync(handle.fileno())
    logging_seconds = time.perf_counter() - tick

    # Time the real standalone independent auditor on the frozen dry-run record.
    audit_output = profile_dir / f"{args.label}_audit.json"
    command = [str(PYTHON), "-m", "ResearchLoop.tools.audit_nmnist_phase1",
               "Reports/results/nmnist_dry_run/nmnist_dry_seed42_binary_B1_500.json",
               "--output", str(audit_output.relative_to(ROOT))]
    tick = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    audit_seconds = time.perf_counter() - tick
    result = {
        "label": args.label, "seed": 42, "sample_id": sample_id,
        "condition_profiled": {"representation": "binary", "budget_type": "B_inf", "beta": 1},
        "seconds_per_sample": {
            "dataset_loading_preprocessing": preprocessing_seconds,
            "model_forward": model.seconds,
            "attack_optimization_excluding_forward_and_projection": optimization_seconds,
            "budget_projection": projection_seconds,
            "serialization": serialization_seconds,
            "logging_with_flush": logging_seconds,
            "independent_audit": audit_seconds,
            "attack_total": attack_total,
            "end_to_end_including_audit": preprocessing_seconds + attack_total + serialization_seconds + logging_seconds + audit_seconds,
        },
    }
    atomic_write_json(ROOT / args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
