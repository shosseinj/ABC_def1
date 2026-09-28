"""Independently audit one common-manifest four-model attack from raw events."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.controlled_models import build_model
from experiments.nmnist.paper_aligned import events_to_number_split_binary, events_to_number_split_frames
from scripts.run_nmnist_seed42_paper_aligned_integer_attacks import sha256

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")


def main(metadata_path: Path, output_path: Path) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors = []
    for prefix in ("manifest", "clean_result", "model_source", "checkpoint", "clean_cache", "artifact", "adapter", "configuration"):
        path = ROOT / meta[f"{prefix}_path"]
        if sha256(path) != meta[f"{prefix}_sha256"]:
            errors.append(f"{prefix} hash mismatch")
    manifest = json.loads((ROOT / meta["manifest_path"]).read_text(encoding="utf-8"))
    result = json.loads((ROOT / meta["clean_result_path"]).read_text(encoding="utf-8"))
    rep = meta["representation"]
    name = meta["model"]
    if manifest.get("selection") != "first 1000 jointly clean-correct official-test samples, in test order" or manifest.get("representation") != rep:
        errors.append("joint manifest policy/representation mismatch")
    if manifest["models"][name]["checkpoint_sha256"] != meta["checkpoint_sha256"]:
        errors.append("manifest checkpoint mismatch")
    ids = np.asarray(manifest["sample_ids"], dtype=np.int64)
    labels = np.asarray(manifest["labels"], dtype=np.int64)
    if len(ids) != 1000 or len(np.unique(ids)) != 1000 or np.any(np.diff(ids) <= 0):
        errors.append("manifest IDs are not 1000 unique ordered test IDs")
    jointly_eligible = np.ones(10000, dtype=bool)
    for joint_name in ("custom", "convnet", "resnet18", "vggsnn"):
        item = manifest["models"][joint_name]
        joint_result_path = ROOT / item["result_path"]
        if sha256(joint_result_path) != item["result_sha256"]:
            errors.append(f"{joint_name} clean-result hash mismatch")
        joint_result = json.loads(joint_result_path.read_text(encoding="utf-8"))
        joint_prediction_path = ROOT / joint_result["predictions_path"]
        if sha256(joint_prediction_path) != item["predictions_sha256"]:
            errors.append(f"{joint_name} prediction hash mismatch")
        with np.load(joint_prediction_path, allow_pickle=False) as joint_data:
            joint_ids = joint_data["sample_ids"].astype(np.int64)
            joint_labels = joint_data["labels"].astype(np.int64)
            joint_predictions = joint_data["predictions"].astype(np.int64)
        if not np.array_equal(joint_ids, np.arange(10000)) or not np.array_equal(joint_labels[ids], labels):
            errors.append(f"{joint_name} official test order/labels mismatch")
        jointly_eligible &= joint_predictions == joint_labels
    if not np.array_equal(np.flatnonzero(jointly_eligible)[:1000], ids):
        errors.append("joint manifest is not first 1000 clean-correct intersection")
    with np.load(ROOT / result["predictions_path"], allow_pickle=False) as data:
        recorded_ids = data["sample_ids"].astype(np.int64)
        recorded_labels = data["labels"].astype(np.int64)
        recorded_predictions = data["predictions"].astype(np.int64)
    if sha256(ROOT / result["predictions_path"]) != result["predictions_sha256"]:
        errors.append("prediction artifact hash mismatch")
    if not np.array_equal(recorded_ids, np.arange(10000)) or not np.array_equal(labels, recorded_labels[ids]) or not np.all(recorded_predictions[ids] == labels):
        errors.append("model clean-correct selection mismatch")
    with np.load(ROOT / meta["artifact_path"], allow_pickle=False) as source:
        payload = {key: source[key].copy() for key in source.files}
    if not np.array_equal(payload["sample_ids"].astype(np.int64), ids) or not np.array_equal(payload["labels"].astype(np.int64), labels):
        errors.append("attack sample IDs/labels differ from joint manifest")
    offsets = payload["offsets"].astype(np.int64)
    if len(offsets) != 1001 or offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(payload["value"]):
        errors.append("invalid packet offsets")
    from tonic.datasets import NMNIST
    dataset = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    converter = events_to_number_split_frames if rep == "integer" else events_to_number_split_binary
    reconstructed_clean = np.zeros((1000, 10, 2, 34, 34), dtype=np.uint16)
    reconstructed_adv = np.zeros_like(reconstructed_clean)
    evidence = []
    kind, requested = meta["budget_type"], int(meta["requested_budget"])
    for index, sample_id in enumerate(ids):
        events, label = dataset[int(sample_id)]
        if len(events["x"]) > np.iinfo(np.uint16).max:
            errors.append(f"sample {int(sample_id)} may overflow uint16 packet counts")
        clean = converter(events, 10).astype(np.uint16)
        reconstructed_clean[index] = clean
        start, end = int(offsets[index]), int(offsets[index + 1])
        source_t = payload["source_t"][start:end].astype(np.int64)
        line = payload["line"][start:end].astype(np.int64)
        target_t = payload["target_t"][start:end].astype(np.int64)
        values = payload["value"][start:end].astype(np.uint16)
        source = np.argwhere(clean.reshape(10, -1) != 0)
        identities = np.array_equal(source[:, 0], source_t) and np.array_equal(source[:, 1], line)
        amplitudes = identities and np.array_equal(clean.reshape(10, -1)[source_t, line], values)
        domain = bool(np.all((target_t >= 0) & (target_t < 10)))
        collisions = len(set(zip(line.tolist(), target_t.tolist()))) == len(target_t)
        if domain and collisions:
            reconstructed_adv[index].reshape(10, -1)[target_t, line] = values
        displacement = np.abs(target_t - source_t)
        realized = {"B_inf": int(displacement.max(initial=0)), "B1": int(displacement.sum()), "B0": int(np.count_nonzero(displacement))}
        budgets = (realized["B_inf"] == int(payload["realized_b_inf"][index]) and
                   realized["B1"] == int(payload["realized_b1"][index]) and
                   realized["B0"] == int(payload["realized_b0"][index]) and realized[kind] <= requested)
        packets = np.count_nonzero(clean) == len(values) == np.count_nonzero(reconstructed_adv[index])
        mass = int(clean.sum(dtype=np.int64)) == int(reconstructed_adv[index].sum(dtype=np.int64))
        amplitude_multiset = np.array_equal(np.sort(clean[clean != 0]), np.sort(reconstructed_adv[index][reconstructed_adv[index] != 0]))
        passed = bool(identities and amplitudes and domain and collisions and budgets and packets and mass and amplitude_multiset and int(label) == int(labels[index]))
        if not passed:
            errors.append(f"sample {int(sample_id)} packet/budget/label audit failed")
        evidence.append({"sample_id": int(sample_id), "packet_count": int(len(values)), "mass": int(clean.sum(dtype=np.int64)),
                         "realized_b_inf": realized["B_inf"], "realized_b1": realized["B1"], "realized_b0": realized["B0"],
                         "packet_audit_pass": passed})
    cache = np.load(ROOT / meta["clean_cache_path"], mmap_mode="r")
    if not np.array_equal(reconstructed_clean, np.asarray(cache[ids])):
        errors.append("raw-event reconstruction differs from frozen cache")
    model = build_model(name).cuda().eval()
    checkpoint = torch.load(ROOT / meta["checkpoint_path"], map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    clean_preds, adv_preds = [], []
    for begin in range(0, 1000, 64):
        end = min(begin + 64, 1000)
        with torch.no_grad():
            clean_preds.extend(model(torch.from_numpy(reconstructed_clean[begin:end].astype(np.float32)).cuda()).argmax(1).cpu().tolist())
            adv_preds.extend(model(torch.from_numpy(reconstructed_adv[begin:end].astype(np.float32)).cuda()).argmax(1).cpu().tolist())
        print(f"AUDIT {meta['run_id']} {end}/1000", flush=True)
    successes = 0
    for index, sample_id in enumerate(ids):
        success = adv_preds[index] != int(labels[index])
        successes += int(success)
        valid = (clean_preds[index] == int(labels[index]) == int(payload["clean_predictions"][index])
                 and clean_preds[index] == int(recorded_predictions[sample_id])
                 and adv_preds[index] == int(payload["adv_predictions"][index])
                 and success == bool(payload["success"][index]))
        evidence[index]["clean_prediction"] = clean_preds[index]
        evidence[index]["attacked_prediction"] = adv_preds[index]
        evidence[index]["prediction_audit_pass"] = valid
        if not valid:
            errors.append(f"sample {int(sample_id)} prediction audit failed")
    if successes != int(meta["successful_attack_numerator_runner"]):
        errors.append("runner/auditor numerator mismatch")
    audit = {"status": "PASS" if not errors else "FAIL", "run_id": meta["run_id"], "dataset": "N-MNIST",
             "representation": rep, "model": name, "budget_type": kind, "requested_budget": requested,
             "clean_correct_denominator": 1000, "successful_attacks": successes, "asr_percent": successes / 10,
             "errors": errors[:200], "samples": evidence,
             "verified": ["joint manifest model predictions", "raw event reconstruction", "packet identity/amplitude/mass",
                          "collision-free temporal occupancy", "all realized budgets", "clean and attacked model predictions",
                          "checkpoint/cache/configuration/manifest/artifact/adapter hashes"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)
    print(f"AUDIT COMPLETE {meta['run_id']} {audit['status']} ASR={audit['asr_percent']:.2f}% errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"controlled attack audit failed with {len(errors)} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(ROOT / args.metadata, ROOT / args.output)
