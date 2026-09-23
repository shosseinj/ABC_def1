"""Independently re-audit and rerun frozen SHD resume350 test inference."""
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import environment_metadata, select_device, set_determinism, sha256, write_json
from experiments.shd.snn_baseline import evaluate, make_tensor_loader, preprocess_partition
from models.shd_snn import SHDRecurrentSNN

SEEDS = [42, 123, 777, 2026, 6543]
TOLERANCE = 1e-12
EXPECTED_TEST_SHA256 = "47d8621a092cb483bea9448a93c0a3375a6726d370045090c2f7a77f70f95df3"
EXPECTED_SPLIT_SHA256 = "043dc832ebb6130a301224068c166294224caa29dfa13c831dcc12492dc0c08b"


def mean_sample_sd(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def audit_before_evaluation(config):
    test_h5 = ROOT / config["data_root"] / "SHD" / "shd_test.h5"
    split_path = ROOT / "results" / "shd_snn_multiseed_split.json"
    if sha256(test_h5) != EXPECTED_TEST_SHA256 or sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Official SHD test or training split identity changed.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(split["train_indices"])
    validation_ids = set(split["validation_indices"])
    if train_ids & validation_ids or len(train_ids | validation_ids) != 8156:
        raise RuntimeError("Training/validation split integrity failed.")
    with h5py.File(test_h5, "r") as handle:
        labels = np.asarray(handle["labels"], dtype=np.int64)
        class_keys = [value.decode() if isinstance(value, bytes) else str(value)
                      for value in np.asarray(handle["extra/keys"])]
    if len(labels) != 2264 or labels.min() != 0 or labels.max() != 19:
        raise RuntimeError("Official SHD test labels are invalid.")
    expected_keys = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                     "null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun"]
    if class_keys != expected_keys:
        raise RuntimeError("SHD class mapping changed.")
    if (config["time_steps"], config["input_channels"], config["duration_us"], config["representation"]) != (
            100, 700, 1_400_000,
            "dense binary occupancy preserving ordered spike-time bins and cochlear channel identity"):
        raise RuntimeError("SHD preprocessing contract changed.")
    validations, previous = [], []
    for seed in SEEDS:
        validation = json.loads((ROOT / "results" / f"shd_snn_seed{seed}_resume350_validation.json").read_text())
        prior = json.loads((ROOT / "results" / f"shd_snn_seed{seed}_resume350_test.json").read_text())
        checkpoint_path, config_path = ROOT / validation["checkpoint"], ROOT / validation["config"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (validation["seed"] != seed or checkpoint["config"]["seed"] != seed
                or sha256(checkpoint_path) != validation["checkpoint_sha256"]
                or sha256(config_path) != validation["config_sha256"]):
            raise RuntimeError(f"Seed {seed} checkpoint/config provenance failed.")
        validations.append(validation)
        previous.append(prior)
    return validations, previous, labels, class_keys


def main():
    from tonic.datasets import SHD

    output_path = ROOT / "results" / "shd_snn_test_reaudit.json"
    if output_path.exists():
        raise RuntimeError("SHD test re-audit artifact exists; refusing another rerun.")
    config = json.loads((ROOT / "configs" / "shd_snn_multiseed.json").read_text(encoding="utf-8"))
    validations, previous_results, h5_labels, class_keys = audit_before_evaluation(config)
    dataset = SHD(save_to=str(ROOT / config["data_root"]), train=False)
    frames, labels, preprocessing = preprocess_partition(dataset, config, "official_test_reaudit")
    if not np.array_equal(labels, h5_labels) or preprocessing["clipped_events"] != 0:
        raise RuntimeError("Tonic labels or test preprocessing disagree with audited HDF5 data.")
    results = []
    for validation, previous in zip(validations, previous_results):
        seed = validation["seed"]
        seed_config = json.loads((ROOT / validation["config"]).read_text(encoding="utf-8"))
        set_determinism(seed)
        device = select_device(seed_config["device"])
        model = SHDRecurrentSNN(seed_config["input_channels"], seed_config["hidden_size"],
                                seed_config["n_classes"], seed_config["lif_decay"]).to(device)
        model.load_state_dict(torch.load(ROOT / validation["checkpoint"], map_location=device,
                                         weights_only=True)["model_state"], strict=True)
        model.eval()
        if model.training or any(module.training for module in model.modules()):
            raise RuntimeError(f"Seed {seed} model has active training-time state.")
        loader = make_tensor_loader(frames, labels, np.arange(len(labels)), seed_config["batch_size"], False, seed)
        recomputed = evaluate(model, loader, device, seed_config["n_classes"], True)
        differences = {
            "accuracy": recomputed["accuracy"] - previous["test"]["accuracy"],
            "macro_f1": recomputed["macro_f1"] - previous["test"]["macro_f1"],
            "loss": recomputed["loss"] - previous["test"]["loss"],
        }
        exact_matrix_match = recomputed["confusion_matrix"] == previous["test"]["confusion_matrix"]
        match = exact_matrix_match and all(abs(value) <= TOLERANCE for value in differences.values())
        row = {"seed": seed, "checkpoint": validation["checkpoint"],
               "checkpoint_sha256": validation["checkpoint_sha256"], "test_samples": recomputed["sample_count"],
               "previous_test_accuracy": previous["test"]["accuracy"],
               "recomputed_test_accuracy": recomputed["accuracy"], "macro_f1": recomputed["macro_f1"],
               "per_class_accuracy": recomputed["per_class_accuracy"],
               "confusion_matrix": recomputed["confusion_matrix"], "differences": differences,
               "confusion_matrix_match": exact_matrix_match, "match": match}
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)
    aggregate = {
        "test_accuracy": mean_sample_sd(row["recomputed_test_accuracy"] for row in results),
        "macro_f1": mean_sample_sd(row["macro_f1"] for row in results),
    }
    prior_summary = json.loads((ROOT / "results" / "shd_snn_resume350_test_summary.json").read_text())
    aggregate_match = all(
        abs(aggregate[metric][stat] - prior_summary[metric][stat]) <= TOLERANCE
        for metric in ("test_accuracy", "macro_f1") for stat in ("mean", "sample_sd"))
    verdict = "VERIFIED" if aggregate_match and all(row["match"] for row in results) else "CORRECTED"
    payload = {
        "status": verdict, "audit_type": "independent reproducibility rerun on previously accessed official test",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "tolerance": TOLERANCE,
        "dataset_audit": {"official_test_h5_sha256": EXPECTED_TEST_SHA256, "test_samples": 2264,
                          "labels_aligned": True, "class_mapping": class_keys,
                          "split_sha256": EXPECTED_SPLIT_SHA256, "validation_mixed_into_test": False},
        "preprocessing_audit": preprocessing,
        "model_audit": {"architecture": config["architecture"], "parameters": 108692,
                        "eval_mode": True, "dropout_present": False, "batch_norm_present": False,
                        "lif_state_reset": "membrane, recurrent spikes, and logits zero-initialized per forward call"},
        "results": results, "aggregate": aggregate, "previous_aggregate": {
            "test_accuracy": prior_summary["test_accuracy"], "macro_f1": prior_summary["macro_f1"]},
        "aggregate_match": aggregate_match, "environment": environment_metadata(),
        "official_test_rerun_count": 1, "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(output_path, payload)
    with (ROOT / "results" / "shd_snn_test_reaudit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "checkpoint", "checkpoint_sha256", "test_samples", "previous_test_accuracy",
                  "recomputed_test_accuracy", "macro_f1", "accuracy_difference", "macro_f1_difference", "match"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({"seed": row["seed"], "checkpoint": row["checkpoint"],
                             "checkpoint_sha256": row["checkpoint_sha256"], "test_samples": row["test_samples"],
                             "previous_test_accuracy": row["previous_test_accuracy"],
                             "recomputed_test_accuracy": row["recomputed_test_accuracy"],
                             "macro_f1": row["macro_f1"], "accuracy_difference": row["differences"]["accuracy"],
                             "macro_f1_difference": row["differences"]["macro_f1"], "match": row["match"]})
    lines = ["# SHD SNN Test Re-Audit", "", f"Verdict: **{verdict}**", "",
             "This was an independent reproducibility rerun on a previously accessed official test partition.", "",
             "| Seed | Previous Test | Recomputed Test | Macro-F1 | Match |", "|---:|---:|---:|---:|---|"]
    for row in results:
        lines.append(f"| {row['seed']} | {row['previous_test_accuracy']:.6f} | "
                     f"{row['recomputed_test_accuracy']:.6f} | {row['macro_f1']:.6f} | {row['match']} |")
    lines.extend(["", f"Aggregate test accuracy: {aggregate['test_accuracy']['mean']:.6f} +/- "
                  f"{aggregate['test_accuracy']['sample_sd']:.6f}",
                  f"Aggregate macro-F1: {aggregate['macro_f1']['mean']:.6f} +/- "
                  f"{aggregate['macro_f1']['sample_sd']:.6f}"])
    (ROOT / "results" / "shd_snn_test_reaudit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"SHD TEST EVALUATION: {verdict}", flush=True)


if __name__ == "__main__":
    main()
