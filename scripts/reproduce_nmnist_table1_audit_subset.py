"""Fresh, non-destructive reproduction of selected N-MNIST attack records.

This script intentionally writes only beneath
Reports/results/nmnist_table1_audit and never changes benchmark artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.pil_pgd import PILPGDAttack, PILPGDConfig
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
SEED = 42
SAMPLE_IDS = (5059, 6471, 7404, 862, 5403)
CONDITIONS = (("B1", 500), ("B0", 200))
SOURCE_DIR = ROOT / "Reports/results/nmnist_binary_true_attacks"
OUTPUT_DIR = ROOT / "Reports/results/nmnist_table1_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_saved(payload: np.lib.npyio.NpzFile, index: int, clean: np.ndarray) -> np.ndarray:
    start, end = map(int, payload["offsets"][index:index + 2])
    source_t = payload["source_t"][start:end].astype(np.int64)
    line = payload["line"][start:end].astype(np.int64)
    target_t = payload["target_t"][start:end].astype(np.int64)
    value = payload["value"][start:end]
    source = np.argwhere(clean.reshape(10, -1) != 0)
    if not (np.array_equal(source[:, 0], source_t) and np.array_equal(source[:, 1], line)):
        raise RuntimeError("saved packet identities do not match clean input")
    result = np.zeros_like(clean.reshape(10, -1))
    result[target_t, line] = value
    return result.reshape(clean.shape)


def independent_metrics(clean: np.ndarray, adversarial: np.ndarray, displacement: np.ndarray) -> dict:
    flat_clean = clean.reshape(10, -1)
    flat_adv = adversarial.reshape(10, -1)
    source = np.argwhere(flat_clean != 0)
    source_t, line = source[:, 0], source[:, 1]
    delta = displacement.reshape(10, -1)[source_t, line].astype(np.int64)
    target_t = source_t + delta
    reconstructed = np.zeros_like(flat_clean)
    reconstructed[target_t, line] = flat_clean[source_t, line]
    absolute = np.abs(delta)
    return {
        "packet_count_clean": int(np.count_nonzero(clean)),
        "packet_count_adversarial": int(np.count_nonzero(adversarial)),
        "mass_clean": int(clean.astype(np.int64).sum()),
        "mass_adversarial": int(adversarial.astype(np.int64).sum()),
        "domain_valid": bool(np.all((target_t >= 0) & (target_t < 10))),
        "collision_free": len(set(zip(line.tolist(), target_t.tolist()))) == len(target_t),
        "reconstruction_equal": bool(np.array_equal(reconstructed, flat_adv)),
        "amplitudes_equal": bool(np.array_equal(
            np.sort(clean[clean != 0]), np.sort(adversarial[adversarial != 0])
        )),
        "realized_b_inf": int(absolute.max(initial=0)),
        "realized_b1": int(absolute.sum()),
        "realized_b0": int(np.count_nonzero(delta)),
    }


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the saved CUDA projection")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    manifest_path = SOURCE_DIR / "manifests/nmnist_binary_seed42_manifest.json"
    cache_path = SOURCE_DIR / "checkpoints/seed42_binary_manifest_cache.npz"
    checkpoint_path = ROOT / "checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path, allow_pickle=False)
    positions = {int(sample_id): index for index, sample_id in enumerate(cache["sample_ids"])}
    if any(sample_id not in positions for sample_id in SAMPLE_IDS):
        raise RuntimeError("fixed audit sample is absent from the frozen manifest cache")

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NMNISTConvSNN(0.5).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    records = []

    for budget_type, budget in CONDITIONS:
        artifact_path = SOURCE_DIR / "runs" / f"nmnist_binary_true_attacks_seed42_{budget_type}_{budget}.npz"
        payload = np.load(artifact_path, allow_pickle=False)
        artifact_positions = {int(sample_id): index for index, sample_id in enumerate(payload["sample_ids"])}
        for sample_id in SAMPLE_IDS:
            cache_index = positions[sample_id]
            artifact_index = artifact_positions[sample_id]
            clean_np = np.array(cache["binary"][cache_index], dtype=np.uint8, copy=True)
            label = int(cache["labels"][cache_index])
            clean = torch.from_numpy(clean_np.astype(np.float32, copy=False))[None].to(device)
            y = torch.tensor([label], dtype=torch.long, device=device)
            attack = PILPGDAttack(model, PILPGDConfig(budget_type, budget, projector="cuda"))
            adversarial, displacement = attack(clean, y)
            fresh_np = adversarial[0].detach().cpu().numpy().astype(np.uint8)
            displacement_np = displacement[0].detach().cpu().numpy().astype(np.int8)
            saved_np = reconstruct_saved(payload, artifact_index, clean_np)
            metrics = independent_metrics(clean_np, fresh_np, displacement_np)
            requested_value = metrics[{"B1": "realized_b1", "B0": "realized_b0"}[budget_type]]
            clean_prediction = int(model(clean).argmax(1).item())
            fresh_prediction = int(attack.last_logits.argmax(1).item())
            saved_prediction = int(payload["adv_predictions"][artifact_index])
            passed = all((
                metrics["packet_count_clean"] == metrics["packet_count_adversarial"],
                metrics["mass_clean"] == metrics["mass_adversarial"],
                metrics["domain_valid"], metrics["collision_free"],
                metrics["reconstruction_equal"], metrics["amplitudes_equal"],
                requested_value <= budget, clean_prediction == label,
                fresh_prediction != label, np.array_equal(fresh_np, saved_np),
                fresh_prediction == saved_prediction,
            ))
            records.append({
                "sample_id": sample_id,
                "label": label,
                "budget_type": budget_type,
                "requested_budget": budget,
                "clean_prediction": clean_prediction,
                "fresh_adversarial_prediction": fresh_prediction,
                "saved_adversarial_prediction": saved_prediction,
                "fresh_equals_saved_tensor": bool(np.array_equal(fresh_np, saved_np)),
                "fresh_attack_success": fresh_prediction != label,
                "passed": passed,
                **metrics,
            })

    output = {
        "status": "PASS" if all(record["passed"] for record in records) else "FAIL",
        "purpose": "fresh deterministic subset reproduction; not a paper-protocol replication",
        "seed": SEED,
        "sample_ids": list(SAMPLE_IDS),
        "conditions": [{"budget_type": kind, "budget": budget} for kind, budget in CONDITIONS],
        "command": f"{PYTHON} scripts/reproduce_nmnist_table1_audit_subset.py",
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest_path),
        "cache_path": str(cache_path.relative_to(ROOT)).replace("\\", "/"),
        "cache_sha256": sha256(cache_path),
        "records": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "small_subset_reproduction.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps({"status": output["status"], "records": len(records), "output": str(output_path)}))
    if output["status"] != "PASS":
        raise RuntimeError("subset reproduction failed")


if __name__ == "__main__":
    main()
