"""Independent read-only audit of all frozen seed-42 Binary PIL-PGD artifacts."""
from __future__ import annotations

import argparse
import hashlib
import inspect
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

from attacks.pil_pgd import PILPGDConfig, PILPGDAttack
from experiments.nmnist.snn_baseline import events_to_frames
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
BASE = ROOT / "Reports/results/nmnist_binary_true_attacks"
OUTPUT = ROOT / "Reports/results/nmnist_seed42_verification/seed42_attack_verification.json"
CHECKPOINT = ROOT / "checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt"
PREDICTIONS = ROOT / "Reports/results/nmnist_seed42_verification/seed42_clean_predictions.npz"
BUDGETS = (("B_inf", 1), ("B_inf", 2), ("B_inf", 3),
           ("B1", 500), ("B1", 750), ("B1", 1000),
           ("B0", 200), ("B0", 300), ("B0", 400))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} | {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def close(self) -> None:
        self.stream.close()


@torch.no_grad()
def infer(model, values: np.ndarray, device, log: Logger, phase: str) -> np.ndarray:
    predictions = []
    started = time.perf_counter()
    for start in range(0, len(values), 64):
        end = min(start + 64, len(values))
        tensor = torch.from_numpy(np.asarray(values[start:end], dtype=np.float32)).to(device)
        predictions.extend(model(tensor).argmax(1).cpu().tolist())
        if start == 0 or end % 256 == 0 or end == len(values):
            log(f"{phase}: samples={end}/{len(values)} elapsed={time.perf_counter() - started:.1f}s")
    return np.asarray(predictions, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    log = Logger(ROOT / args.log)
    try:
        if Path(sys.executable).resolve() != PYTHON.resolve():
            raise RuntimeError(f"use required interpreter {PYTHON}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        log("PHASE 2 START: read-only audit; no attacks will be executed")
        clean_verification = json.loads((OUTPUT.parent / "seed42_clean_verification.json").read_text(encoding="utf-8"))
        if clean_verification.get("status") != "PASS" or clean_verification["frozen_test_metrics_batch64"]["accuracy"] < .95:
            raise RuntimeError("clean-model gate is not PASS")

        manifest_path = BASE / "manifests/nmnist_binary_seed42_manifest.json"
        cache_path = BASE / "checkpoints/seed42_binary_manifest_cache.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = np.load(cache_path, allow_pickle=False)
        independent = np.load(PREDICTIONS, allow_pickle=False)
        labels_full = independent["labels"].astype(np.int64)
        predictions_full = independent["predictions_batch64"].astype(np.int64)
        eligible = np.flatnonzero(predictions_full == labels_full)
        expected_ids = sorted(map(int, eligible), key=lambda i: (
            hashlib.sha256(f"nmnist_binary_true_attacks:42:{i}".encode()).digest(), i))[:1000]
        manifest_ids = [int(row["sample_id"]) for row in manifest["samples"]]

        from tonic.datasets import NMNIST
        native = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
        raw_errors = []
        for position, sample_id in enumerate(manifest_ids):
            events, label = native[sample_id]
            fresh = (events_to_frames(events, 10) != 0).astype(np.uint8)
            if int(label) != int(cache["labels"][position]):
                raw_errors.append(f"label mismatch sample {sample_id}")
            if not np.array_equal(fresh, cache["binary"][position]):
                raw_errors.append(f"preprocessing/cache mismatch sample {sample_id}")
            if (position + 1) % 250 == 0:
                log(f"raw dataset/cache verification: samples={position + 1}/1000")

        device = torch.device("cuda")
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        model = NMNISTConvSNN(decay=0.5, n_classes=10).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        conditions, global_errors = [], []
        for budget_type, budget in BUDGETS:
            run_id = f"nmnist_binary_true_attacks_seed42_{budget_type}_{budget}"
            metadata_path = BASE / "runs" / f"{run_id}.json"
            artifact_path = BASE / "runs" / f"{run_id}.npz"
            historical_audit_path = BASE / "runs" / f"{run_id}.audit.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            historical_audit = json.loads(historical_audit_path.read_text(encoding="utf-8"))
            payload = np.load(artifact_path, allow_pickle=False)
            errors = []
            expected_hashes = ((artifact_path, metadata["artifact_sha256"], "artifact"),
                               (manifest_path, metadata["manifest_sha256_file"], "manifest"),
                               (cache_path, metadata["cache_sha256"], "cache"),
                               (CHECKPOINT, metadata["checkpoint_sha256"], "checkpoint"))
            for path, expected_hash, name in expected_hashes:
                if sha256(path) != expected_hash:
                    errors.append(f"{name} hash mismatch")
            if metadata.get("seed") != 42 or metadata.get("attack_batch_size") != 64 or metadata.get("audit_batch_size") != 64:
                errors.append("metadata seed/batch mismatch")
            if metadata.get("budget_type") != budget_type or metadata.get("budget") != budget:
                errors.append("metadata budget mismatch")
            if not np.array_equal(payload["sample_ids"], np.asarray(manifest_ids)):
                errors.append("sample order mismatch")
            if not np.array_equal(payload["labels"].astype(np.int64), cache["labels"].astype(np.int64)):
                errors.append("label mismatch")

            reconstructed = np.zeros_like(cache["binary"])
            computed = np.empty((1000, 3), dtype=np.int64)
            offsets = payload["offsets"].astype(np.int64)
            for index in range(1000):
                clean = cache["binary"][index]
                flat_clean = clean.reshape(10, -1)
                start, end = int(offsets[index]), int(offsets[index + 1])
                source_t = payload["source_t"][start:end].astype(np.int64)
                lines = payload["line"][start:end].astype(np.int64)
                target_t = payload["target_t"][start:end].astype(np.int64)
                values = payload["value"][start:end].astype(np.int64)
                sources = np.argwhere(flat_clean != 0)
                if not (np.array_equal(sources[:, 0], source_t) and np.array_equal(sources[:, 1], lines)):
                    errors.append(f"source identity mismatch at position {index}")
                    continue
                if not np.array_equal(flat_clean[source_t, lines].astype(np.int64), values):
                    errors.append(f"amplitude mismatch at position {index}")
                if np.any(target_t < 0) or np.any(target_t >= 10):
                    errors.append(f"domain violation at position {index}")
                if len(set(zip(lines.tolist(), target_t.tolist()))) != len(target_t):
                    errors.append(f"collision at position {index}")
                adv_flat = reconstructed[index].reshape(10, -1)
                adv_flat[target_t, lines] = values.astype(clean.dtype)
                delta = target_t - source_t
                absolute = np.abs(delta)
                computed[index] = (absolute.max(initial=0), absolute.sum(), np.count_nonzero(delta))
                if np.count_nonzero(reconstructed[index]) != np.count_nonzero(clean):
                    errors.append(f"packet count mismatch at position {index}")
                if int(reconstructed[index].astype(np.int64).sum()) != int(clean.astype(np.int64).sum()):
                    errors.append(f"mass mismatch at position {index}")
            reported = np.column_stack((payload["reported_b_inf"], payload["reported_b1"], payload["reported_b0"])).astype(np.int64)
            if not np.array_equal(computed, reported):
                errors.append("stored realized budgets differ from independent recomputation")
            active_column = {"B_inf": 0, "B1": 1, "B0": 2}[budget_type]
            if not np.all(computed[:, active_column] == budget):
                errors.append("active requested budget is not realized exactly for every sample")

            clean_predictions = infer(model, cache["binary"], device, log, f"{budget_type}={budget} clean")
            adv_predictions = infer(model, reconstructed, device, log, f"{budget_type}={budget} adversarial")
            labels = cache["labels"].astype(np.int64)
            if not np.array_equal(clean_predictions, payload["clean_predictions"].astype(np.int64)):
                errors.append("clean prediction mismatch")
            if not np.array_equal(adv_predictions, payload["adv_predictions"].astype(np.int64)):
                errors.append("adversarial prediction mismatch")
            if not np.all(clean_predictions == labels):
                errors.append("ASR denominator includes clean-error samples")
            successes = int(np.count_nonzero(adv_predictions != labels))
            asr = 100.0 * successes / len(labels)
            if successes != int(historical_audit["successful_attacks"]):
                errors.append("historical ASR numerator mismatch")
            condition = {
                "run_id": run_id, "budget_type": budget_type, "requested_budget": budget,
                "sample_count": len(labels), "clean_correct_denominator": int(np.count_nonzero(clean_predictions == labels)),
                "successful_attacks": successes, "asr_percent": asr,
                "realized_b_inf_min_max": [int(computed[:, 0].min()), int(computed[:, 0].max())],
                "realized_b1_min_max": [int(computed[:, 1].min()), int(computed[:, 1].max())],
                "realized_b0_min_max": [int(computed[:, 2].min()), int(computed[:, 2].max())],
                "artifact_sha256": sha256(artifact_path), "historical_audit_sha256": sha256(historical_audit_path),
                "errors": errors, "status": "PASS" if not errors else "FAIL",
            }
            conditions.append(condition)
            global_errors.extend(f"{run_id}: {error}" for error in errors)
            log(f"{budget_type}={budget}: samples=1000 ASR={asr:.2f}% budget_validation={condition['status']}")

        manifest_checks = {
            "seed_42": manifest.get("seed") == 42,
            "official_test_10000": manifest.get("official_test_samples") == 10000,
            "eligible_count_matches_independent_evaluation": manifest.get("eligible_binary_clean_correct") == len(eligible) == 9873,
            "selection_exactly_reproduced": manifest_ids == expected_ids,
            "same_manifest_all_conditions": all(json.loads((BASE / "runs" / f"nmnist_binary_true_attacks_seed42_{kind}_{budget}.json").read_text(encoding="utf-8"))["manifest_sha256_file"] == sha256(manifest_path) for kind, budget in BUDGETS),
            "raw_labels_and_preprocessing_match_cache": not raw_errors,
            "no_duplicate_selected_ids": len(set(manifest_ids)) == 1000,
        }
        source = inspect.getsource(PILPGDAttack.__call__)
        defaults = PILPGDConfig("B1", 500)
        attack_settings = {
            "objective": "untargeted white-box cross-entropy maximization",
            "iterations_b_inf": 20, "iterations_b1_b0": 40,
            "temperature": defaults.temperature, "alpha": defaults.alpha,
            "logit_clip": defaults.logit_clip, "capacity_weight": defaults.capacity_weight,
            "budget_weight": defaults.budget_weight, "strict_projected_forward": "hard + soft - soft.detach()" in source,
            "attack_source_sha256": sha256(ROOT / "attacks/pil_pgd.py"),
            "runner_source_sha256": sha256(ROOT / "scripts/run_nmnist_binary_true_attacks.py"),
            "limitation": "Historical metadata did not serialize a config/source hash; settings are established from the preserved runner/source and exact fresh reproduction evidence, not from metadata alone.",
        }
        if raw_errors:
            global_errors.extend(raw_errors)
        if not all(manifest_checks.values()):
            global_errors.extend(name for name, passed in manifest_checks.items() if not passed)
        status = "PASS" if not global_errors else "FAIL"
        result = {
            "status": status, "scope": "seed 42 Binary local protocol only", "seed": 42,
            "paper_comparability": "NON_COMPARABLE",
            "non_comparability_reasons": ["equal-duration local binning versus paper equal-event-count slicing",
                "custom compact SNN/checkpoint versus paper models/checkpoints",
                "SHA-256 ordered local attacked subset versus paper first 1000 clean-correct in test order"],
            "batch_size": 64, "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256(manifest_path), "manifest_checks": manifest_checks,
            "attack_settings": attack_settings, "conditions": conditions,
            "fresh_reproduction_evidence": "Reports/results/nmnist_table1_audit/small_subset_reproduction.json",
            "fresh_reproduction_records": 10, "errors": global_errors,
        }
        atomic_json(OUTPUT, result)
        log(f"PHASE 2 COMPLETE: status={status}; paper_comparability=NON_COMPARABLE")
        if status != "PASS":
            raise RuntimeError(f"attack verification failed with {len(global_errors)} errors")
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
