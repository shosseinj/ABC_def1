"""Independent reconstruction audit for corrected N-MNIST Integer attacks."""
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

from models.nmnist_snn import NMNISTConvSNN


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata_path = ROOT / args.metadata
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_path = ROOT / meta["artifact_path"]
    manifest_path = ROOT / meta["manifest_path"]
    cache_path = ROOT / meta["cache_path"]
    checkpoint_path = ROOT / meta["checkpoint_path"]
    errors, samples = [], []
    for path, expected, name in (
        (artifact_path, meta["artifact_sha256"], "artifact"),
        (manifest_path, meta["manifest_sha256_file"], "manifest file"),
        (cache_path, meta["cache_sha256"], "cache"),
        (checkpoint_path, meta["checkpoint_sha256"], "checkpoint"),
    ):
        if sha256(path) != expected:
            errors.append(f"{name} hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_dependency") != "integer_clean_correct_only":
        errors.append("manifest is not Integer-only")
    if manifest.get("binary_dependency") is not False:
        errors.append("manifest has Binary dependency")
    if int(manifest.get("seed", -1)) != int(meta["seed"]):
        errors.append("manifest seed mismatch")
    payload = np.load(artifact_path, allow_pickle=False)
    cache = np.load(cache_path, allow_pickle=False)
    expected_ids = np.asarray([r["sample_id"] for r in manifest["samples"]], dtype=np.int64)
    if not np.array_equal(payload["sample_ids"], expected_ids):
        errors.append("artifact sample order differs from manifest")
    if not np.array_equal(cache["sample_ids"], expected_ids):
        errors.append("cache sample order differs from manifest")
    if len(expected_ids) != 1000:
        errors.append(f"expected 1000 samples, got {len(expected_ids)}")

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    stored_seed = checkpoint.get("seed", checkpoint.get("config", {}).get("seed"))
    if int(stored_seed) != int(meta["seed"]):
        errors.append("checkpoint embedded seed mismatch")
    model = NMNISTConvSNN(0.5).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    clean_all = cache["integer"]
    labels = payload["labels"].astype(np.int64)
    offsets = payload["offsets"].astype(np.int64)
    reconstructed = np.zeros_like(clean_all)
    audit_predictions = []
    successes = 0
    for i in range(len(expected_ids)):
        clean = clean_all[i]
        start, end = int(offsets[i]), int(offsets[i + 1])
        source_t = payload["source_t"][start:end].astype(np.int64)
        line = payload["line"][start:end].astype(np.int64)
        target_t = payload["target_t"][start:end].astype(np.int64)
        values = payload["value"][start:end].astype(np.int64)
        flat_clean = clean.reshape(10, -1)
        source = np.argwhere(flat_clean != 0)
        source_ok = np.array_equal(source[:, 0], source_t) and np.array_equal(source[:, 1], line)
        values_ok = np.array_equal(flat_clean[source_t, line].astype(np.int64), values)
        domain_ok = bool(np.all((target_t >= 0) & (target_t < 10)))
        no_collision = len(set(zip(line.tolist(), target_t.tolist()))) == len(target_t)
        adv_flat = reconstructed[i].reshape(10, -1)
        if domain_ok and no_collision:
            adv_flat[target_t, line] = values.astype(clean.dtype)
        delta = target_t - source_t
        absolute = np.abs(delta)
        b_inf = int(absolute.max(initial=0)); b1 = int(absolute.sum()); b0 = int(np.count_nonzero(delta))
        requested = {"B_inf": b_inf, "B1": b1, "B0": b0}[meta["budget_type"]]
        budget_ok = requested <= int(meta["budget"])
        packet_count_ok = int(np.count_nonzero(clean)) == len(values) == int(np.count_nonzero(reconstructed[i]))
        amplitude_ok = np.array_equal(np.sort(clean[clean != 0].astype(np.int64)),
                                      np.sort(reconstructed[i][reconstructed[i] != 0].astype(np.int64)))
        mass_ok = int(clean.astype(np.int64).sum()) == int(reconstructed[i].astype(np.int64).sum())
        manifest_row = manifest["samples"][i]
        label_ok = int(manifest_row["true_label"]) == int(labels[i]) == int(cache["labels"][i])
        clean_correct_recorded = int(manifest_row["integer_clean_prediction"]) == int(labels[i])
        sample_pass = all((source_ok, values_ok, domain_ok, no_collision, budget_ok,
                           packet_count_ok, amplitude_ok, mass_ok, label_ok, clean_correct_recorded))
        samples.append({"sample_id": int(expected_ids[i]), "passed_structural": sample_pass,
                        "realized_b_inf": b_inf, "realized_b1": b1, "realized_b0": b0,
                        "packet_count": len(values), "mass": int(clean.astype(np.int64).sum())})
        if not sample_pass:
            errors.append(f"sample {int(expected_ids[i])} structural audit failed")

    with torch.no_grad():
        for i in range(len(expected_ids)):
            clean_tensor = torch.from_numpy(np.array(clean_all[i:i+1], dtype=np.float32, copy=True)).to(device)
            adv_tensor = torch.from_numpy(np.array(reconstructed[i:i+1], dtype=np.float32, copy=True)).to(device)
            clean_prediction = int(model(clean_tensor).argmax(1))
            adv_prediction = int(model(adv_tensor).argmax(1))
            audit_predictions.append(adv_prediction)
            if clean_prediction != int(labels[i]):
                errors.append(f"sample {int(expected_ids[i])} not clean-correct under attack arithmetic")
            if clean_prediction != int(payload["clean_predictions"][i]):
                errors.append(f"sample {int(expected_ids[i])} clean prediction mismatch")
            if adv_prediction != int(payload["adv_predictions"][i]):
                errors.append(f"sample {int(expected_ids[i])} adversarial prediction mismatch")
            success = adv_prediction != int(labels[i])
            successes += int(success)
            samples[i]["clean_prediction"] = clean_prediction
            samples[i]["adv_prediction"] = adv_prediction
            samples[i]["attack_success"] = success
    passed = not errors
    atomic_json(ROOT / args.output, {"passed": passed, "status": "PASS" if passed else "FAIL",
        "run_id": meta["run_id"], "seed": meta["seed"], "representation": "integer",
        "manifest_policy": "integer_clean_correct_only", "sample_count": len(expected_ids),
        "successful_attacks": successes, "asr_percent": 100.0 * successes / max(len(expected_ids), 1),
        "errors": errors[:100], "samples": samples,
        "verified": ["seed", "checkpoint hash", "manifest hash and Integer-only policy", "sample identity",
          "clean correctness", "packet identity/count/amplitude", "mass", "temporal domain", "capacity-1",
          "B_inf", "B1", "B0", "reconstruction", "clean prediction", "adversarial prediction", "success"]})
    if not passed:
        raise RuntimeError(f"independent audit failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
