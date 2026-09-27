"""Independent raw-data reconstruction audit for paper-aligned attack cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.paper_aligned import events_to_number_split_binary
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    metadata_path = ROOT / args.metadata
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = {
        "artifact": ROOT / meta["artifact_path"], "manifest": ROOT / meta["manifest_path"],
        "checkpoint": ROOT / meta["checkpoint_path"], "clean_cache": ROOT / meta["clean_cache_path"],
        "configuration": ROOT / meta["configuration_path"], "adapter": ROOT / meta["adapter_path"],
    }
    expected_hashes = {
        "artifact": meta["artifact_sha256"], "manifest": meta["manifest_sha256"],
        "checkpoint": meta["checkpoint_sha256"], "clean_cache": meta["clean_cache_sha256"],
        "configuration": meta["configuration_sha256"], "adapter": meta["adapter_sha256"],
    }
    errors = []
    for name, path in paths.items():
        actual = sha256(path)
        if actual != expected_hashes[name]:
            errors.append(f"{name} hash mismatch: {actual}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("selection") != "first 1000 clean-correct samples encountered in unshuffled official test order":
        errors.append("manifest selection policy mismatch")
    if canonical_hash(manifest.get("samples", [])) != manifest.get("samples_sha256"):
        errors.append("manifest sample hash mismatch")
    prediction_path = ROOT / manifest["clean_predictions_path"]
    if sha256(prediction_path) != manifest["clean_predictions_sha256"]:
        errors.append("clean prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as predictions:
        all_ids = predictions["sample_ids"].astype(np.int64)
        all_labels = predictions["labels"].astype(np.int64)
        all_predictions = predictions["predictions"].astype(np.int64)
    independently_selected = all_ids[np.flatnonzero(all_predictions == all_labels)[:1000]]
    expected_ids = np.asarray([row["sample_id"] for row in manifest["samples"]], dtype=np.int64)
    if not np.array_equal(independently_selected, expected_ids):
        errors.append("manifest is not the first 1000 clean-correct test samples")
    with np.load(paths["artifact"], allow_pickle=False) as source:
        payload = {key: source[key].copy() for key in source.files}
    if not np.array_equal(payload["sample_ids"].astype(np.int64), expected_ids):
        errors.append("artifact sample IDs/order differ from manifest")
    if len(expected_ids) != 1000:
        errors.append(f"expected denominator 1000, found {len(expected_ids)}")

    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    offsets = payload["offsets"].astype(np.int64)
    reconstructed_clean = np.zeros((1000, 10, 2, 34, 34), dtype=np.uint8)
    reconstructed_adv = np.zeros_like(reconstructed_clean)
    sample_evidence = []
    kind = meta["budget_type"]; requested_budget = int(meta["requested_budget"])
    for index, sample_id in enumerate(expected_ids):
        row = manifest["samples"][index]
        events, raw_label = dataset[int(sample_id)]
        event_digest = hashlib.sha256()
        for field in ("x", "y", "t", "p"):
            event_digest.update(np.ascontiguousarray(events[field]).tobytes())
        clean = events_to_number_split_binary(events, 10)
        reconstructed_clean[index] = clean
        start, end = int(offsets[index]), int(offsets[index + 1])
        source_t = payload["source_t"][start:end].astype(np.int64)
        line = payload["line"][start:end].astype(np.int64)
        target_t = payload["target_t"][start:end].astype(np.int64)
        values = payload["value"][start:end].astype(np.uint8)
        clean_flat = clean.reshape(10, -1)
        source = np.argwhere(clean_flat != 0)
        identity_ok = (np.array_equal(source[:, 0], source_t)
                       and np.array_equal(source[:, 1], line))
        values_ok = identity_ok and np.array_equal(clean_flat[source_t, line], values)
        domain_ok = bool(np.all((target_t >= 0) & (target_t < 10)))
        collision_free = len(set(zip(line.tolist(), target_t.tolist()))) == len(target_t)
        if domain_ok and collision_free:
            reconstructed_adv[index].reshape(10, -1)[target_t, line] = values
        delta = target_t - source_t
        absolute = np.abs(delta)
        realized = {
            "B_inf": int(absolute.max(initial=0)), "B1": int(absolute.sum()),
            "B0": int(np.count_nonzero(delta)),
        }
        stored_budget_ok = (
            realized["B_inf"] == int(payload["realized_b_inf"][index])
            and realized["B1"] == int(payload["realized_b1"][index])
            and realized["B0"] == int(payload["realized_b0"][index])
        )
        active_budget_respected = realized[kind] <= requested_budget
        active_budget_saturated = realized[kind] == requested_budget
        packet_count_ok = np.count_nonzero(clean) == len(values) == np.count_nonzero(reconstructed_adv[index])
        mass_ok = int(clean.sum(dtype=np.int64)) == int(reconstructed_adv[index].sum(dtype=np.int64))
        amplitude_ok = np.array_equal(np.sort(clean[clean != 0]), np.sort(reconstructed_adv[index][reconstructed_adv[index] != 0]))
        raw_identity_ok = event_digest.hexdigest() == row["raw_event_sha256"]
        label_ok = int(raw_label) == int(row["true_label"]) == int(payload["labels"][index])
        structural_pass = all((identity_ok, values_ok, domain_ok, collision_free,
                               stored_budget_ok, active_budget_respected, packet_count_ok,
                               mass_ok, amplitude_ok, raw_identity_ok, label_ok))
        if not structural_pass:
            errors.append(f"sample {int(sample_id)} structural audit failed")
        sample_evidence.append({
            "position": index, "sample_id": int(sample_id), "true_label": int(raw_label),
            "raw_event_sha256": event_digest.hexdigest(), "packet_count": int(len(values)),
            "total_mass": int(clean.sum(dtype=np.int64)), "realized_b_inf": realized["B_inf"],
            "realized_b1": realized["B1"], "realized_b0": realized["B0"],
            "identity_ok": identity_ok, "amplitudes_ok": values_ok and amplitude_ok,
            "mass_ok": mass_ok, "spatial_polarity_line_ok": identity_ok,
            "temporal_domain_ok": domain_ok, "collision_free_capacity_1": collision_free,
            "active_budget_respected": active_budget_respected,
            "active_budget_saturated": active_budget_saturated, "structural_pass": structural_pass,
        })
    clean_cache = np.load(paths["clean_cache"], mmap_mode="r")
    if not np.array_equal(reconstructed_clean, np.asarray(clean_cache[expected_ids])):
        errors.append("raw-data reconstruction differs from frozen clean cache")

    device = torch.device("cuda")
    checkpoint = torch.load(paths["checkpoint"], map_location=device, weights_only=True)
    if int(checkpoint.get("seed", -1)) != 42:
        errors.append("checkpoint embedded seed mismatch")
    model = NMNISTConvSNN(0.5, 10).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    clean_audit, adv_audit = [], []
    for start in range(0, 1000, 64):
        end = min(start + 64, 1000)
        clean_tensor = torch.from_numpy(reconstructed_clean[start:end].astype(np.float32)).to(device)
        adv_tensor = torch.from_numpy(reconstructed_adv[start:end].astype(np.float32)).to(device)
        with torch.no_grad():
            clean_audit.extend(model(clean_tensor).argmax(1).cpu().tolist())
            adv_audit.extend(model(adv_tensor).argmax(1).cpu().tolist())
        print(f"INDEPENDENT AUDIT {kind}={requested_budget} predictions={end}/1000", flush=True)
    successes = 0
    for index, sample_id in enumerate(expected_ids):
        label = int(payload["labels"][index]); clean_prediction = int(clean_audit[index]); adv_prediction = int(adv_audit[index])
        prediction_ok = (clean_prediction == int(payload["clean_predictions"][index])
                         and adv_prediction == int(payload["adv_predictions"][index]))
        clean_correct = clean_prediction == label
        success = adv_prediction != label
        success_ok = success == bool(payload["success"][index])
        successes += int(success)
        sample_evidence[index].update({"clean_prediction": clean_prediction,
                                       "adversarial_prediction": adv_prediction,
                                       "clean_correct": clean_correct, "attack_success": success,
                                       "prediction_records_ok": prediction_ok and success_ok})
        if not (prediction_ok and clean_correct and success_ok):
            errors.append(f"sample {int(sample_id)} prediction audit failed")
    if successes != int(meta["successful_attack_numerator_runner"]):
        errors.append("runner/auditor successful-attack numerator mismatch")
    status = "PASS" if not errors else "FAIL"
    result = {
        "status": status, "passed": not errors, "run_id": meta["run_id"], "dataset": "N-MNIST",
        "seed": 42, "budget_type": kind, "requested_budget": requested_budget,
        "clean_correct_denominator": 1000, "successful_attacks": successes,
        "asr": successes / 1000.0, "asr_percent": successes / 10.0,
        "audit_batch_size": 64, "errors": errors[:200], "samples": sample_evidence,
        "verified": ["raw sample identity", "seed", "true label", "clean prediction",
                     "clean correctness", "equal-event-count Binary reconstruction",
                     "packet count and identity", "packet amplitudes", "total mass",
                     "spatial coordinates and polarity via fixed event line", "temporal domain",
                     "collision-free capacity-1", "requested and all realized budgets",
                     "adversarial prediction", "attack success", "serialization reconstruction",
                     "checkpoint/configuration/manifest/source hashes"],
    }
    atomic_json(ROOT / args.output, result)
    print(f"INDEPENDENT AUDIT COMPLETE {kind}={requested_budget} status={status} "
          f"numerator={successes} denominator=1000 ASR={successes/10:.2f}% errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"independent audit failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
