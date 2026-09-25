"""Independent, read-only verification of the frozen seed-42 N-MNIST model."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import events_to_frames, stratified_train_validation_indices
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
CHECKPOINT = ROOT / "checkpoints/nmnist_binary_true/nmnist_binary_seed42_best.pt"
HISTORY = ROOT / "Reports/logs/nmnist_binary_true/seed42_history.csv"
EXTERNAL_CONFIG = ROOT / "configs/nmnist_snn_clean_seed42.json"
OUTPUT_DIR = ROOT / "Reports/results/nmnist_seed42_verification"


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


class BinarySet(Dataset):
    def __init__(self, native):
        self.native = native

    def __len__(self):
        return len(self.native)

    def __getitem__(self, index):
        events, label = self.native[index]
        binary = (events_to_frames(events, 10) != 0).astype(np.float32)
        if binary.shape != (10, 2, 34, 34) or not np.all((binary == 0) | (binary == 1)):
            raise RuntimeError(f"invalid Binary preprocessing at sample {index}")
        return torch.from_numpy(binary), int(label), int(index)


def make_loader(dataset, indices, batch_size: int):
    selected = dataset if indices is None else Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
    return DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, data, device, phase: str):
    labels, predictions, sample_ids, logits_all = [], [], [], []
    loss_sum = 0.0
    correct = 0
    processed = 0
    started = time.perf_counter()
    model.eval()
    for batch_index, (frames, target, sample_id) in enumerate(data, start=1):
        frames = frames.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(frames)
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(1) == target).sum())
        processed += len(target)
        labels.extend(target.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        sample_ids.extend(sample_id.tolist())
        logits_all.append(logits.cpu().numpy())
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(data):
            print(f"eval {phase}: batch {batch_index}/{len(data)} samples={processed} "
                  f"running_accuracy={correct / processed:.6f} elapsed={time.perf_counter() - started:.1f}s",
                  flush=True)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    matrix = confusion_matrix(labels, predictions, labels=range(10))
    totals = matrix.sum(axis=1)
    metrics = {
        "samples": int(len(labels)),
        "correct": int(np.count_nonzero(labels == predictions)),
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "loss": loss_sum / len(labels),
        "per_class_accuracy": [float(matrix[i, i] / totals[i]) for i in range(10)],
        "confusion_matrix": matrix.tolist(),
        "true_histogram": np.bincount(labels, minlength=10).tolist(),
        "prediction_histogram": np.bincount(predictions, minlength=10).tolist(),
    }
    print(f"eval {phase}: COMPLETE samples={metrics['samples']} accuracy={metrics['accuracy']:.6f} "
          f"macro_f1={metrics['macro_f1']:.6f} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return metrics, labels, predictions, np.asarray(sample_ids, dtype=np.int64), np.concatenate(logits_all)


def event_hash(events) -> str:
    digest = hashlib.sha256()
    for field in ("x", "y", "t", "p"):
        array = np.ascontiguousarray(events[field])
        digest.update(field.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def binary_hash(events) -> str:
    binary = np.ascontiguousarray(events_to_frames(events, 10) != 0)
    digest = hashlib.sha256()
    digest.update(binary.tobytes())
    return digest.hexdigest()


def duplicate_summary(values: list[str]) -> dict:
    counts = Counter(values)
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "unique": len(counts),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_samples": int(sum(count - 1 for count in duplicate_groups)),
        "maximum_multiplicity": max(counts.values(), default=0),
    }


def independent_binary(events) -> np.ndarray:
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    output = np.zeros((10, 2, 34, 34), dtype=np.bool_)
    if len(t):
        duration = max(int(t[-1]) - int(t[0]) + 1, 1)
        temporal = np.minimum(((t - int(t[0])) * 10) // duration, 9)
        output[temporal, p, y, x] = True
    return output


def scan_partition(dataset, name: str, audit_indices: set[int]) -> tuple[dict, list[str], list[str]]:
    raw_hashes, binary_hashes = [], []
    target_mismatches, preprocessing_mismatches = [], []
    target_array = np.asarray(dataset.targets, dtype=np.int64)
    for index in range(len(dataset)):
        events, label = dataset[index]
        label = int(label)
        if label != int(target_array[index]):
            target_mismatches.append(index)
        # Input-only hashes intentionally exclude labels so duplicate inputs with
        # conflicting labels cannot evade the duplicate/overlap audit.
        raw_hashes.append(event_hash(events))
        binary_hashes.append(binary_hash(events))
        if index in audit_indices:
            production = events_to_frames(events, 10) != 0
            if not np.array_equal(production, independent_binary(events)):
                preprocessing_mismatches.append(index)
        if (index + 1) % 10000 == 0:
            print(f"scan {name}: {index + 1}/{len(dataset)}", flush=True)
    result = {
        "samples": len(dataset),
        "target_mismatch_count": len(target_mismatches),
        "target_mismatch_examples": target_mismatches[:20],
        "independent_preprocessing_samples": len(audit_indices),
        "preprocessing_mismatch_count": len(preprocessing_mismatches),
        "preprocessing_mismatch_examples": preprocessing_mismatches[:20],
        "raw_duplicates": duplicate_summary(raw_hashes),
        "binary_duplicates": duplicate_summary(binary_hashes),
    }
    return result, raw_hashes, binary_hashes


def main() -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)

    from tonic.datasets import NMNIST

    print("PHASE 1 START: seed=42, attacks prohibited until clean gate passes", flush=True)
    train_native = NMNIST(save_to=str(ROOT / "data/nmnist"), train=True)
    test_native = NMNIST(save_to=str(ROOT / "data/nmnist"), train=False)
    train_ids, validation_ids = stratified_train_validation_indices(train_native.targets, 500, 42)
    split_checks = {
        "official_train_samples": len(train_native),
        "official_test_samples": len(test_native),
        "training_samples": len(train_ids),
        "validation_samples": len(validation_ids),
        "index_intersection": int(np.intersect1d(train_ids, validation_ids).size),
        "index_union_complete": len(np.union1d(train_ids, validation_ids)) == len(train_native),
        "training_class_histogram": np.bincount(np.asarray(train_native.targets)[train_ids], minlength=10).tolist(),
        "validation_class_histogram": np.bincount(np.asarray(train_native.targets)[validation_ids], minlength=10).tolist(),
        "test_class_histogram": np.bincount(np.asarray(test_native.targets), minlength=10).tolist(),
    }

    rng = np.random.default_rng(420042)
    train_audit = set(map(int, rng.choice(len(train_native), size=1000, replace=False)))
    test_audit = set(map(int, rng.choice(len(test_native), size=1000, replace=False)))
    train_scan, train_raw, train_binary = scan_partition(train_native, "official-train", train_audit)
    test_scan, test_raw, test_binary = scan_partition(test_native, "official-test", test_audit)
    overlap = {
        "raw_hash_intersection": len(set(train_raw).intersection(test_raw)),
        "binary_hash_intersection": len(set(train_binary).intersection(test_binary)),
    }

    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = NMNISTConvSNN(decay=0.5, n_classes=10).to(device).eval()
    load_result = model.load_state_dict(checkpoint["model_state"], strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_data, test_data = BinarySet(train_native), BinarySet(test_native)

    train_metrics, *_ = evaluate(model, make_loader(train_data, train_ids, 64), device, "frozen-train-batch64")
    validation_metrics, *_ = evaluate(model, make_loader(train_data, validation_ids, 64), device, "frozen-validation-batch64")
    test_metrics, labels64, predictions64, ids64, logits64 = evaluate(
        model, make_loader(test_data, None, 64), device, "official-test-batch64")
    test_metrics_b1, labels1, predictions1, ids1, logits1 = evaluate(
        model, make_loader(test_data, None, 1), device, "official-test-batch1-consistency")
    consistency = {
        "samples": len(labels64),
        "labels_equal": bool(np.array_equal(labels64, labels1)),
        "sample_order_equal": bool(np.array_equal(ids64, ids1)),
        "prediction_mismatch_count": int(np.count_nonzero(predictions64 != predictions1)),
        "prediction_mismatch_sample_ids": ids64[predictions64 != predictions1].tolist()[:100],
        "maximum_absolute_logit_difference": float(np.max(np.abs(logits64 - logits1))),
        "batch64_accuracy": test_metrics["accuracy"],
        "batch1_accuracy": test_metrics_b1["accuracy"],
    }

    history = list(csv.DictReader(HISTORY.open(encoding="utf-8")))
    for row in history:
        for key in ("epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"):
            row[key] = int(row[key]) if key == "epoch" else float(row[key])
    selected = sorted(history, key=lambda row: (-row["validation_accuracy"], row["validation_loss"], row["epoch"]))[0]
    checkpoint_selection = {
        "history_epochs": len(history),
        "history_selected_epoch": selected["epoch"],
        "history_selected_validation_accuracy": selected["validation_accuracy"],
        "history_selected_validation_loss": selected["validation_loss"],
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "checkpoint_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "matches_history_rule": (
            int(checkpoint["best_epoch"]) == selected["epoch"]
            and float(checkpoint["validation_accuracy"]) == selected["validation_accuracy"]
            and abs(float(checkpoint["validation_loss"]) - selected["validation_loss"]) < 1e-12
        ),
    }

    external_config = json.loads(EXTERNAL_CONFIG.read_text(encoding="utf-8"))
    embedded_config = checkpoint["config"]
    config_provenance = {
        "embedded_config": embedded_config,
        "embedded_config_sha256": canonical_hash(embedded_config),
        "external_config_path": str(EXTERNAL_CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "external_config_sha256": sha256(EXTERNAL_CONFIG),
        "external_config_used_by_binary_runner": False,
        "external_representation": external_config.get("input_representation"),
        "external_batch_size": external_config.get("batch_size"),
        "provenance_warning": "Binary runner hard-coded its configuration; the standalone config describes count frames and was not loaded.",
    }

    checks = {
        "checkpoint_seed_42": int(checkpoint["seed"]) == 42,
        "strict_checkpoint_load": not load_result.missing_keys and not load_result.unexpected_keys,
        "parameter_count_25482": parameters == 25482,
        "official_partition_sizes": len(train_native) == 60000 and len(test_native) == 10000,
        "split_disjoint_complete": split_checks["index_intersection"] == 0 and split_checks["index_union_complete"],
        "validation_500_per_class": split_checks["validation_class_histogram"] == [500] * 10,
        "labels_match_targets": train_scan["target_mismatch_count"] == test_scan["target_mismatch_count"] == 0,
        "independent_preprocessing_matches": train_scan["preprocessing_mismatch_count"] == test_scan["preprocessing_mismatch_count"] == 0,
        "no_exact_raw_train_test_overlap": overlap["raw_hash_intersection"] == 0,
        "no_binary_train_test_overlap": overlap["binary_hash_intersection"] == 0,
        "checkpoint_selection_valid": checkpoint_selection["matches_history_rule"],
        "full_test_10000": test_metrics["samples"] == 10000,
        "reported_accuracy_reproduced": test_metrics["correct"] == 9873 and test_metrics["accuracy"] == 0.9873,
        "batch1_batch64_predictions_equal": consistency["prediction_mismatch_count"] == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status == "PASS":
        print("CLEAN GATE PASS: accuracy >= 95% and all integrity checks passed", flush=True)
    else:
        failed = [name for name, passed in checks.items() if not passed]
        print(f"STOP — CLEAN GATE FAIL: {failed}; attacks remain prohibited", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / "seed42_clean_predictions.npz"
    temporary = prediction_path.with_name(prediction_path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, sample_ids=ids64.astype(np.int32), labels=labels64.astype(np.int8),
                            predictions_batch64=predictions64.astype(np.int8), predictions_batch1=predictions1.astype(np.int8))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, prediction_path)

    result = {
        "status": status,
        "seed": 42,
        "new_evaluation_batch_size": 64,
        "batch_size_1_used_only_for_required_consistency_check": True,
        "architecture": "NMNISTConvSNN(decay=0.5, n_classes=10)",
        "trainable_parameters": parameters,
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "model_source_sha256": sha256(ROOT / "models/nmnist_snn.py"),
        "training_runner_sha256": sha256(ROOT / "scripts/train_nmnist_binary_true.py"),
        "history_path": str(HISTORY.relative_to(ROOT)).replace("\\", "/"),
        "history_sha256": sha256(HISTORY),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "split_checks": split_checks,
        "official_train_scan": train_scan,
        "official_test_scan": test_scan,
        "train_test_overlap": overlap,
        "checkpoint_selection": checkpoint_selection,
        "config_provenance": config_provenance,
        "frozen_train_metrics_batch64": train_metrics,
        "frozen_validation_metrics_batch64": validation_metrics,
        "frozen_test_metrics_batch64": test_metrics,
        "test_metrics_batch1_consistency_only": test_metrics_b1,
        "batch_consistency": consistency,
        "predictions_artifact": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
        "checks": checks,
    }
    result_path = OUTPUT_DIR / "seed42_clean_verification.json"
    atomic_json(result_path, result)
    print(json.dumps({"status": status, "test_accuracy": test_metrics["accuracy"],
                      "macro_f1": test_metrics["macro_f1"], "parameters": parameters,
                      "batch_prediction_mismatches": consistency["prediction_mismatch_count"],
                      "output": str(result_path)}), flush=True)
    if status != "PASS":
        raise RuntimeError("seed-42 clean verification failed")


if __name__ == "__main__":
    main()
