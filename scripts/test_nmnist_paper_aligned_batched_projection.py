"""Exact equivalence gate for optimized batched official projectors."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.official_pil_adapter import TimeMajorVictim
from experiments.nmnist.official_pil_adapter_batched import (
    _batched_official_binf_projection, _batched_official_global_projection,
    load_official_attack_module,
)

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
OUTPUT = ROOT / "Reports/results/nmnist_seed42_paper_aligned/batched_projection_equivalence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compare(module, value, probabilities, kind: str, budget: int):
    device = value.device
    dummy = torch.nn.Identity().to(device)
    if kind == "B1":
        official = module.PGDTimeShiftAfterEncoder_L1(
            device, TimeMajorVictim(dummy), l1_steps_budget=budget)
        method = official._strict_projection_allT_L1_active_global
        optimized = _batched_official_global_projection(
            value, probabilities, budget, False, True)
    else:
        official = module.PGDTimeShiftAfterEncoder_L0(
            device, TimeMajorVictim(dummy), l0_moves_budget=budget)
        method = official._strict_projection_allT_L0_active_global
        optimized = _batched_official_global_projection(
            value, probabilities, budget, True, True)
    references = [method(value[:, i:i + 1], probabilities[:, i:i + 1], budget,
                         return_disp=True) for i in range(value.shape[1])]
    reference_output = torch.cat([item[0] for item in references], dim=1)
    reference_displacement = torch.cat([item[1] for item in references], dim=1)
    output_equal = torch.equal(optimized[0], reference_output)
    displacement_equal = torch.equal(optimized[1], reference_displacement)
    if not output_equal or not displacement_equal:
        raise RuntimeError(f"{kind}={budget} optimized projection mismatch")
    return {"kind": kind, "budget": budget, "batch": value.shape[1],
            "output_equal": output_equal, "displacement_equal": displacement_equal}


def compare_binf(module, value, probabilities, radius: int):
    reference = module.PGDTimeShiftAfterEncoder_Lowgpu(
        torch.device("cuda"), TimeMajorVictim(torch.nn.Identity().cuda()), D=radius)
    expected = [reference._final_projection_packets_greedy_active(
        value[:, i:i + 1], probabilities[:, i:i + 1], return_disp=True)
        for i in range(value.shape[1])]
    actual = _batched_official_binf_projection(value, probabilities, radius, True)
    if not torch.equal(actual[0], torch.cat([item[0] for item in expected], 1)):
        raise RuntimeError(f"B_inf={radius} optimized output mismatch")
    if not torch.equal(actual[1], torch.cat([item[1] for item in expected], 1)):
        raise RuntimeError(f"B_inf={radius} optimized displacement mismatch")
    return {"kind": "B_inf", "budget": radius, "batch": value.shape[1],
            "output_equal": True, "displacement_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binf-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--attack-equivalence", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    module = load_official_attack_module()
    device = torch.device("cuda")
    if args.attack_equivalence:
        from experiments.nmnist.official_pil_adapter import _cat_projection
        from experiments.nmnist.official_pil_adapter_batched import build_attack
        from models.nmnist_snn import NMNISTConvSNN
        checkpoint = torch.load(ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned/seed42_best_epoch15.pt",
                                map_location="cuda", weights_only=True)
        victim = NMNISTConvSNN(0.5, 10).cuda().eval()
        victim.load_state_dict(checkpoint["model_state"], strict=True)
        manifest = json.loads((ROOT / "Reports/results/nmnist_seed42_paper_aligned/seed42_attack_manifest.json")
                              .read_text(encoding="utf-8"))
        ids = [row["sample_id"] for row in manifest["samples"][:4]]
        labels = torch.tensor([row["true_label"] for row in manifest["samples"][:4]], device=device)
        cache = np.load(ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned/test_t10_number_split_binary_uint8.npy",
                        mmap_mode="r")
        value = torch.from_numpy(np.array(cache[ids], dtype=np.float32, copy=True)).cuda()
        value = value.permute(1, 0, 2, 3, 4).contiguous()
        base = module.PGDTimeShiftAfterEncoder_Lowgpu
        class PriorHybrid(base):
            def _final_projection_packets_greedy_active(self, clean, pi, return_disp=False):
                return _cat_projection(base._final_projection_packets_greedy_active,
                                       self, clean, pi, return_disp=return_disp)
        old = PriorHybrid(device=device, model_without_encoder=TimeMajorVictim(victim),
                          D=3, steps=20, alpha_phi=1.0, lambda_cap=20.0,
                          temperature=1.0, random_start=False, cap_limit=1.0)
        new = build_attack(module, victim, device, "B_inf", 3)
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        started = time.perf_counter()
        expected = old(value, labels, return_disp=True, use_PIL=True,
                       use_cap=True, use_penalty=True, target_label=-1)
        torch.cuda.synchronize()
        old_seconds = time.perf_counter() - started
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        started = time.perf_counter()
        actual = new(value, labels, return_disp=True, use_PIL=True,
                     use_cap=True, use_penalty=True, target_label=-1)
        torch.cuda.synchronize()
        new_seconds = time.perf_counter() - started
        if not torch.equal(expected[0], actual[0]) or not torch.equal(expected[1], actual[1]):
            raise RuntimeError("end-to-end attack output or displacement mismatch")
        print(f"END_TO_END_EQUIVALENCE_PASS samples=4 old_seconds={old_seconds:.3f} "
              f"new_seconds={new_seconds:.3f}", flush=True)
        return
    if args.benchmark_only:
        cache = ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned/test_t10_number_split_binary_uint8.npy"
        real = torch.from_numpy(np.array(np.load(cache, mmap_mode="r")[:64],
                                         dtype=np.float32, copy=True)).cuda()
        real = real.permute(1, 0, 2, 3, 4).contiguous()
        for radius in (1, 2, 3):
            generator = torch.Generator(device=device).manual_seed(74000 + radius)
            pi = torch.softmax(torch.randn((*real.shape, 2 * radius + 1),
                                           generator=generator, device=device), dim=-1)
            _batched_official_binf_projection(real, pi, radius, True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                _batched_official_binf_projection(real, pi, radius, True)
            torch.cuda.synchronize()
            print(f"B_inf={radius} optimized_batch64_seconds={(time.perf_counter()-started)/5:.4f}",
                  flush=True)
        return
    rows = []
    # Random sparse cases exercise unequal packet counts and candidate scores.
    for trial in range(10):
        generator = torch.Generator(device=device).manual_seed(42000 + trial)
        value = (torch.rand((10, 8, 2, 4, 4), generator=generator, device=device) > .86).float()
        probabilities = torch.softmax(torch.randn(
            (10, 8, 2, 4, 4, 10), generator=generator, device=device), dim=-1)
        if not args.binf_only:
            for kind, budget in (("B1", 13), ("B1", 31), ("B0", 5), ("B0", 17)):
                row = compare(module, value, probabilities, kind, budget)
                row["trial"] = trial; row["case"] = "random"
                rows.append(row)
        for radius in (1, 2, 3):
            local_probabilities = torch.softmax(torch.randn(
                (10, 8, 2, 4, 4, 2 * radius + 1), generator=generator,
                device=device), dim=-1)
            row = compare_binf(module, value, local_probabilities, radius)
            row["trial"] = trial; row["case"] = "random"
            rows.append(row)
    # Uniform probabilities stress exact tie ordering used by upstream argsort.
    value = torch.zeros((10, 8, 2, 4, 4), device=device)
    for sample in range(8):
        value[(sample + 1) % 10, sample, sample % 2, sample % 4, (sample * 3) % 4] = 1
        value[(sample + 5) % 10, sample, sample % 2, sample % 4, (sample * 3) % 4] = 1
    probabilities = torch.full((10, 8, 2, 4, 4, 10), .1, device=device)
    if not args.binf_only:
        for kind, budget in (("B1", 3), ("B0", 1)):
            row = compare(module, value, probabilities, kind, budget)
            row["trial"] = 0; row["case"] = "uniform_ties"
            rows.append(row)
    if args.binf_only:
        for radius in (1, 2, 3):
            tied = torch.full((10, 8, 2, 4, 4, 2 * radius + 1),
                              1.0 / (2 * radius + 1), device=device)
            row = compare_binf(module, value, tied, radius)
            row["trial"] = 0; row["case"] = "uniform_ties"
            rows.append(row)
        cache = ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned/test_t10_number_split_binary_uint8.npy"
        real = torch.from_numpy(np.array(np.load(cache, mmap_mode="r")[:4],
                                         dtype=np.float32, copy=True)).cuda()
        real = real.permute(1, 0, 2, 3, 4).contiguous()
        for radius in (1, 2, 3):
            generator = torch.Generator(device=device).manual_seed(73000 + radius)
            pi = torch.softmax(torch.randn((*real.shape, 2 * radius + 1),
                                           generator=generator, device=device), dim=-1)
            started = time.perf_counter()
            row = compare_binf(module, real, pi, radius)
            row["trial"] = 0; row["case"] = "real_nmnist"
            row["comparison_seconds"] = time.perf_counter() - started
            rows.append(row)
    adapter = ROOT / "experiments/nmnist/official_pil_adapter_batched.py"
    result = {"status": "PASS", "comparisons": len(rows), "rows": rows,
              "adapter_path": str(adapter.relative_to(ROOT)).replace("\\", "/"),
              "adapter_sha256": sha256(adapter), "torch_version": torch.__version__}
    if not args.binf_only:
        atomic_json(OUTPUT, result)
    print(f"BATCHED PROJECTION EQUIVALENCE PASS: {len(rows)}/{len(rows)}", flush=True)
    if args.binf_only:
        print(json.dumps([row for row in rows if row["case"] == "real_nmnist"]), flush=True)


if __name__ == "__main__":
    main()
