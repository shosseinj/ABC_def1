"""Versioned, event-level auditable Seed-42 N-MNIST attack protocol.

The frozen PGD and TEMP-DRIFT-v2 implementations are reused without changing
their search logic or hyperparameters. Only the existing canonical input
adapter is attached to their attack paths. Every record stores complete clean
and adversarial event fields so a separate post-hoc auditor can reconstruct
frames, rerun model predictions, and compute feasibility without trusting any
runner feasibility flag.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import events_to_frames
import scripts.run_nmnist_common_attack_protocol_seed42 as legacy
import scripts.run_nmnist_temp_drift_v2_seed42 as v2
from scripts.run_nmnist_attack_protocol_v2_canonical_seed42 import canonical_frames_torch

PROTOCOL_VERSION = "N-MNIST-attack-v3-auditable-canonical-preprocessing"
DEST = ROOT / "results/nmnist_attack_protocol_v3_auditable_seed42"
RECORD_DIR = DEST / "records"
EPS = (0.0001, 0.0025, 0.01, 0.05, 0.10)
ALL_EPS = (0.0,) + EPS
ATTACKS = ("PGD", "TEMP-DRIFT-v2")
MODELS = ("SNN", "QSNN")
BOUND_TOLERANCE = 1e-9
ZERO = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def epsilon_token(epsilon_fraction: float) -> str:
    return f"{int(round(float(epsilon_fraction) * 1_000_000)):06d}"


def attack_slug(attack: str) -> str:
    return attack.lower().replace("-", "_")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def frame_metrics(clean: torch.Tensor, adversarial: torch.Tensor) -> dict:
    difference = (adversarial - clean).float().flatten()
    return {
        "frame_l0": int((difference != 0).sum().item()),
        "frame_l1": float(difference.abs().sum().item()),
        "frame_l2": float(torch.linalg.vector_norm(difference, 2).item()),
        "frame_linf": float(difference.abs().max().item()),
        "changed_frame_elements": int((difference != 0).sum().item()),
    }


def objective_and_margin(logits: torch.Tensor, label: int, objective_name: str) -> tuple[float, float]:
    margin = float(v2.objective_values(logits, label).item())
    if objective_name == "cross_entropy_true_label":
        objective = float(F.cross_entropy(logits, torch.tensor([label], device=logits.device)).item())
    else:
        objective = margin
    return objective, margin


def record_payload(
    *,
    record_key: str,
    sample_id: int,
    label: int,
    model: str,
    attack: str,
    epsilon_fraction: float,
    epsilon_absolute: float,
    duration: float,
    clean_t_native: np.ndarray,
    clean_t: np.ndarray,
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    clean_p: np.ndarray,
    adversarial_t: np.ndarray,
    adversarial_x: np.ndarray,
    adversarial_y: np.ndarray,
    adversarial_p: np.ndarray,
    clean_prediction: int,
    adversarial_prediction: int,
    attack_success: bool,
    clean_objective: float,
    adversarial_objective: float,
    clean_margin: float,
    adversarial_margin: float,
    attack_returned_objective: float,
    attack_returned_prediction: int,
    objective_name: str,
    rng_seed: int,
    runtime_seconds: float,
    forward_evaluations: int,
    backward_evaluations: int,
    candidate_evaluations: int,
    clean_bins: np.ndarray,
    adversarial_bins: np.ndarray,
    clean_frame: torch.Tensor,
    adversarial_frame: torch.Tensor,
    metrics: dict,
) -> dict:
    return {
        "protocol_version": np.asarray(PROTOCOL_VERSION),
        "record_key": np.asarray(record_key),
        "sample_id": np.asarray(sample_id, dtype=np.int64),
        "label": np.asarray(label, dtype=np.int64),
        "model": np.asarray(model),
        "attack": np.asarray(attack),
        "epsilon_fraction": np.asarray(epsilon_fraction, dtype=np.float64),
        "epsilon_absolute": np.asarray(epsilon_absolute, dtype=np.float64),
        "bound_tolerance": np.asarray(BOUND_TOLERANCE, dtype=np.float64),
        "duration": np.asarray(duration, dtype=np.float64),
        "timestamp_domain_min": np.asarray(float(clean_t_native[0]), dtype=np.float64),
        "timestamp_domain_max": np.asarray(float(clean_t_native[-1]), dtype=np.float64),
        "clean_event_count": np.asarray(len(clean_t), dtype=np.int64),
        "adversarial_event_count": np.asarray(len(adversarial_t), dtype=np.int64),
        "clean_x": np.asarray(clean_x, dtype=np.int16),
        "clean_y": np.asarray(clean_y, dtype=np.int16),
        "clean_p": np.asarray(clean_p, dtype=np.int8),
        "clean_t_native": np.asarray(clean_t_native, dtype=np.int64),
        "clean_t": np.asarray(clean_t, dtype=np.float64),
        "adversarial_x": np.asarray(adversarial_x, dtype=np.int16),
        "adversarial_y": np.asarray(adversarial_y, dtype=np.int16),
        "adversarial_p": np.asarray(adversarial_p, dtype=np.int8),
        "adversarial_t": np.asarray(adversarial_t, dtype=np.float64),
        "clean_bins": np.asarray(clean_bins, dtype=np.int64),
        "adversarial_bins": np.asarray(adversarial_bins, dtype=np.int64),
        "clean_prediction": np.asarray(clean_prediction, dtype=np.int64),
        "adversarial_prediction": np.asarray(adversarial_prediction, dtype=np.int64),
        "stored_attack_success": np.asarray(bool(attack_success)),
        "clean_objective": np.asarray(clean_objective, dtype=np.float64),
        "adversarial_objective": np.asarray(adversarial_objective, dtype=np.float64),
        "clean_margin": np.asarray(clean_margin, dtype=np.float64),
        "adversarial_margin": np.asarray(adversarial_margin, dtype=np.float64),
        "attack_returned_objective": np.asarray(attack_returned_objective, dtype=np.float64),
        "attack_returned_prediction": np.asarray(attack_returned_prediction, dtype=np.int64),
        "objective_name": np.asarray(objective_name),
        "rng_seed": np.asarray(rng_seed, dtype=np.int64),
        "runtime_seconds": np.asarray(runtime_seconds, dtype=np.float64),
        "model_forward_evaluations": np.asarray(forward_evaluations, dtype=np.int64),
        "backward_evaluations": np.asarray(backward_evaluations, dtype=np.int64),
        "candidate_evaluations": np.asarray(candidate_evaluations, dtype=np.int64),
        "timestamps_changed": np.asarray(int(np.count_nonzero(adversarial_t != clean_t)), dtype=np.int64),
        "events_changing_bin": np.asarray(int(np.count_nonzero(adversarial_bins != clean_bins)), dtype=np.int64),
        "frame_l0": np.asarray(metrics["frame_l0"], dtype=np.int64),
        "frame_l1": np.asarray(metrics["frame_l1"], dtype=np.float64),
        "frame_l2": np.asarray(metrics["frame_l2"], dtype=np.float64),
        "frame_linf": np.asarray(metrics["frame_linf"], dtype=np.float64),
        "changed_frame_elements": np.asarray(metrics["changed_frame_elements"], dtype=np.int64),
        "clean_frame_sha256": np.asarray(tensor_hash(clean_frame)),
        "adversarial_frame_sha256": np.asarray(tensor_hash(adversarial_frame)),
        "clean_event_sha256": np.asarray(array_hash(clean_t)),
        "adversarial_event_sha256": np.asarray(array_hash(adversarial_t)),
    }


def main() -> None:
    if DEST.exists():
        raise RuntimeError(f"Refusing to overwrite existing auditable protocol directory: {DEST}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen attack experiment")

    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    (DEST / "RUNNING").write_text(
        f"started_utc={datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8"
    )

    # This is the same adapter-only correction used by the preserved v2 run.
    legacy.frames_torch = canonical_frames_torch
    v2.frames_torch = canonical_frames_torch

    from tonic.datasets import NMNIST

    manifest_path = legacy.OUT / "common_clean_correct_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shutil.copyfile(manifest_path, DEST / "common_clean_correct_manifest.json")
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=True)
    models = legacy.load_models()

    rows: list[dict] = []
    record_manifest: list[dict] = []
    started_at = datetime.now(timezone.utc)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "dataset": "N-MNIST",
        "partition": "train",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(manifest["samples"]),
        "models": list(MODELS),
        "attacks": list(ATTACKS),
        "epsilon_fractions": list(EPS),
        "zero_control_included": True,
        "epsilon_semantics": "absolute bound = epsilon_fraction * (last_clean_timestamp - first_clean_timestamp + 1)",
        "bound_tolerance": BOUND_TOLERANCE,
        "representation": "[time, polarity, y, x]; polarity*34*34+y*34+x",
        "canonical_adapter_source": "scripts/run_nmnist_attack_protocol_v2_canonical_seed42.py:canonical_frames_torch",
        "attack_code": {
            "PGD": "scripts/run_nmnist_common_attack_protocol_seed42.py:pgd",
            "TEMP-DRIFT-v2": "scripts/run_nmnist_temp_drift_v2_seed42.py:temp_drift_v2",
        },
        "attack_seeds": "42 + manifest_position*1009 + int(epsilon_fraction*1000000); reused for TEMP-DRIFT-v2 where applicable",
        "query_accounting": {
            "PGD": {"model_forward_evaluations": 41, "backward_evaluations": 20, "candidate_evaluations": 21},
            "TEMP-DRIFT-v2": {"model_forward_evaluations": 1600, "backward_evaluations": 0, "candidate_evaluations": 1600},
        },
        "record_schema": "one compressed NPZ per model/attack/sample/epsilon containing complete clean and adversarial event fields",
        "feasibility_source": "independent post-hoc auditor; runner does not emit feasibility flags",
        "five_seed_campaign_started": False,
        "supersedes_for_audit": "results/nmnist_attack_protocol_v2_canonical_seed42",
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
    }
    write_json_atomic(DEST / "protocol.json", protocol)

    csv_fields = [
        "record_key", "sample_id", "label", "model", "attack", "epsilon_fraction",
        "epsilon_absolute", "clean_prediction", "adversarial_prediction",
        "stored_attack_success", "clean_objective", "adversarial_objective",
        "objective_name", "model_forward_evaluations", "backward_evaluations",
        "candidate_evaluations", "runtime_seconds", "record_path",
    ]

    try:
        for position, item in enumerate(manifest["samples"]):
            sample_id = int(item["sample_id"])
            label = int(item["label"])
            events, dataset_label = dataset[sample_id]
            if int(dataset_label) != label:
                raise RuntimeError(f"Manifest/dataset label mismatch for sample {sample_id}")

            clean_t_native = np.asarray(events["t"], dtype=np.int64)
            clean_t = clean_t_native.astype(np.float64)
            clean_x = np.asarray(events["x"], dtype=np.int16)
            clean_y = np.asarray(events["y"], dtype=np.int16)
            clean_p = np.asarray(events["p"], dtype=np.int8)
            duration = float(clean_t_native[-1] - clean_t_native[0] + 1)
            clean_bins = v2.bins(events, clean_t)
            clean_frame = canonical_frames_torch(events, torch.as_tensor(clean_t, device="cuda", dtype=torch.float32))
            reference_clean_frame = torch.from_numpy(events_to_frames(events, 10)).float()[None].cuda()
            if not torch.equal(clean_frame, reference_clean_frame):
                raise RuntimeError(f"Canonical adapter mismatch for clean sample {sample_id}")

            for model_name in MODELS:
                model = models[model_name]
                with torch.no_grad():
                    clean_logits = model(clean_frame)
                    clean_prediction = int(clean_logits.argmax(1).item())
                if clean_prediction != label:
                    raise RuntimeError(f"Manifest sample is not clean-correct for {model_name}: {sample_id}")
                clean_margin = float(v2.objective_values(clean_logits, label).item())

                for epsilon_fraction in ALL_EPS:
                    rng_seed = legacy.SEED + position * 1009 + int(epsilon_fraction * 1_000_000)
                    epsilon_absolute = float(epsilon_fraction) * duration
                    for attack in ATTACKS:
                        attack_started = time.perf_counter()
                        rng = np.random.default_rng(rng_seed)
                        if attack == "PGD":
                            attacked = legacy.pgd(model, events, label, epsilon_absolute)
                            objective_name = "cross_entropy_true_label"
                            attack_returned_objective = np.nan
                            attack_returned_prediction = -1
                            forward_evaluations, backward_evaluations, candidate_evaluations = 41, 20, 21
                        else:
                            attacked, returned_objective, returned_prediction, query_count = v2.temp_drift_v2(
                                model, events, label, epsilon_absolute, rng
                            )
                            objective_name = "true_class_margin"
                            attack_returned_objective = float(returned_objective)
                            attack_returned_prediction = int(returned_prediction)
                            forward_evaluations, backward_evaluations, candidate_evaluations = query_count, 0, query_count

                        attacked = legacy.project(attacked, clean_t, epsilon_absolute)
                        adversarial_t = np.asarray(attacked, dtype=np.float64)
                        adversarial_x = clean_x.copy()
                        adversarial_y = clean_y.copy()
                        adversarial_p = clean_p.copy()
                        adversarial_bins = v2.bins(events, adversarial_t)
                        adversarial_frame = canonical_frames_torch(
                            events, torch.as_tensor(adversarial_t, device="cuda", dtype=torch.float32)
                        )
                        with torch.no_grad():
                            adversarial_logits = model(adversarial_frame)
                            adversarial_prediction = int(adversarial_logits.argmax(1).item())
                            adversarial_objective, adversarial_margin = objective_and_margin(
                                adversarial_logits, label, objective_name
                            )
                        attack_success = adversarial_prediction != label
                        runtime_seconds = time.perf_counter() - attack_started
                        metrics = frame_metrics(clean_frame, adversarial_frame)

                        if epsilon_fraction == ZERO:
                            if (
                                not np.array_equal(adversarial_t, clean_t)
                                or not np.array_equal(adversarial_bins, clean_bins)
                                or not torch.equal(adversarial_frame, clean_frame)
                                or adversarial_prediction != clean_prediction
                                or attack_success
                            ):
                                raise RuntimeError(f"Zero-control invariant failed for sample {sample_id}, {model_name}, {attack}")

                        record_key = (
                            f"sample={sample_id};label={label};model={model_name};"
                            f"attack={attack};epsilon={epsilon_fraction:.6f}"
                        )
                        filename = (
                            f"sample_{sample_id:06d}__{model_name}__{attack_slug(attack)}__"
                            f"eps_{epsilon_token(epsilon_fraction)}.npz"
                        )
                        record_path = RECORD_DIR / model_name / attack_slug(attack) / filename
                        record_path.parent.mkdir(parents=True, exist_ok=True)
                        payload = record_payload(
                            record_key=record_key,
                            sample_id=sample_id,
                            label=label,
                            model=model_name,
                            attack=attack,
                            epsilon_fraction=epsilon_fraction,
                            epsilon_absolute=epsilon_absolute,
                            duration=duration,
                            clean_t_native=clean_t_native,
                            clean_t=clean_t,
                            clean_x=clean_x,
                            clean_y=clean_y,
                            clean_p=clean_p,
                            adversarial_t=adversarial_t,
                            adversarial_x=adversarial_x,
                            adversarial_y=adversarial_y,
                            adversarial_p=adversarial_p,
                            clean_prediction=clean_prediction,
                            adversarial_prediction=adversarial_prediction,
                            attack_success=attack_success,
                            clean_objective=(
                                float(F.cross_entropy(clean_logits, torch.tensor([label], device=clean_logits.device)).item())
                                if objective_name == "cross_entropy_true_label" else clean_margin
                            ),
                            adversarial_objective=adversarial_objective,
                            clean_margin=clean_margin,
                            adversarial_margin=adversarial_margin,
                            attack_returned_objective=attack_returned_objective,
                            attack_returned_prediction=attack_returned_prediction,
                            objective_name=objective_name,
                            rng_seed=rng_seed,
                            runtime_seconds=runtime_seconds,
                            forward_evaluations=forward_evaluations,
                            backward_evaluations=backward_evaluations,
                            candidate_evaluations=candidate_evaluations,
                            clean_bins=clean_bins,
                            adversarial_bins=adversarial_bins,
                            clean_frame=clean_frame,
                            adversarial_frame=adversarial_frame,
                            metrics=metrics,
                        )
                        np.savez_compressed(record_path, **payload)
                        record_hash = sha256_file(record_path)
                        record_manifest.append({
                            "record_key": record_key,
                            "sample_id": sample_id,
                            "label": label,
                            "model": model_name,
                            "attack": attack,
                            "epsilon_fraction": epsilon_fraction,
                            "record_path": record_path.relative_to(DEST).as_posix(),
                            "sha256": record_hash,
                        })
                        row = {
                            "record_key": record_key,
                            "sample_id": sample_id,
                            "label": label,
                            "model": model_name,
                            "attack": attack,
                            "epsilon_fraction": epsilon_fraction,
                            "epsilon_absolute": epsilon_absolute,
                            "clean_prediction": clean_prediction,
                            "adversarial_prediction": adversarial_prediction,
                            "stored_attack_success": attack_success,
                            "clean_objective": payload["clean_objective"].item(),
                            "adversarial_objective": adversarial_objective,
                            "objective_name": objective_name,
                            "model_forward_evaluations": forward_evaluations,
                            "backward_evaluations": backward_evaluations,
                            "candidate_evaluations": candidate_evaluations,
                            "runtime_seconds": runtime_seconds,
                            "record_path": record_path.relative_to(DEST).as_posix(),
                        }
                        rows.append(row)
                        print(
                            f"{position + 1}/100 {model_name} {attack} "
                            f"eps={epsilon_fraction:.4%} success={attack_success}",
                            flush=True,
                        )

        expected_records = len(manifest["samples"]) * len(MODELS) * len(ATTACKS) * len(ALL_EPS)
        if len(rows) != expected_records or len(record_manifest) != expected_records:
            raise RuntimeError(f"Record count mismatch: {len(rows)} != {expected_records}")

        write_csv_atomic(DEST / "per_sample_results.csv", csv_fields, rows)
        write_json_atomic(DEST / "records_manifest.json", {
            "protocol_version": PROTOCOL_VERSION,
            "record_count": len(record_manifest),
            "records": record_manifest,
        })
        protocol["status"] = "raw_records_complete_pending_independent_audit"
        protocol["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(DEST / "protocol.json", protocol)
        (DEST / "RUNNING").unlink()
    except Exception:
        protocol["status"] = "failed"
        protocol["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(DEST / "protocol.json", protocol)
        raise


if __name__ == "__main__":
    main()
