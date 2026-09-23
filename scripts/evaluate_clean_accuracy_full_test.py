"""Evaluate frozen benchmark checkpoints on each complete benchmark test split.

This is a clean-only evaluator. It deliberately does not read attack manifests and
does not select samples based on predictions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import events_to_frames as nmnist_frames
from models.cifar10_dvs_snn import CIFAR10DVSConvSNN
from models.dvs_gesture_snn import DVSGestureConvSNN
from models.nmnist_snn import NMNISTConvSNN

SEEDS = (42, 123, 777)
REPRESENTATIONS = ("binary", "integer")
T_CRITICAL_DF2_975 = 4.302652729911275


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class NMNISTOfficialTest(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        events, label = self.dataset[index]
        return torch.from_numpy(nmnist_frames(events, 10)).float(), int(label), int(index)


class ArrayTestSet(Dataset):
    def __init__(self, frames, labels, indices=None):
        self.frames = frames
        self.labels = labels
        self.indices = np.arange(len(labels), dtype=np.int64) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source_index = int(self.indices[position])
        frame = torch.from_numpy(np.array(self.frames[source_index], dtype=np.float32, copy=True))
        return frame, int(self.labels[source_index]), source_index


def set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)


def checkpoint_seed(payload: dict) -> int:
    value = payload.get("seed")
    if value is None:
        value = payload.get("config", {}).get("seed", payload.get("config", {}).get("model_seed"))
    if value is None:
        raise RuntimeError("checkpoint has no seed provenance")
    return int(value)


def load_models(dataset: str, factory, device: torch.device, expected_hashes: dict) -> tuple[dict, dict]:
    stems = {
        "N-MNIST": "nmnist_snn_clean_seed{seed}_best.pt",
        "DVS-Gesture": "dvs_gesture_snn_seed{seed}_best.pt",
        "CIFAR10-DVS": "cifar10_dvs_snn_seed{seed}_best.pt",
    }
    models, checkpoints = {}, {}
    for seed in SEEDS:
        path = ROOT / "checkpoints" / stems[dataset].format(seed=seed)
        actual_hash = sha256(path)
        if actual_hash != expected_hashes[seed]:
            raise RuntimeError(f"checkpoint hash mismatch for {dataset} seed {seed}")
        payload = torch.load(path, map_location=device, weights_only=True)
        if checkpoint_seed(payload) != seed:
            raise RuntimeError(f"checkpoint seed mismatch for {dataset} seed {seed}")
        model = factory().to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        if model.training:
            raise RuntimeError("model did not enter eval mode")
        models[seed] = model
        checkpoints[seed] = str(path.relative_to(ROOT)).replace("\\", "/")
    return models, checkpoints


@torch.no_grad()
def evaluate_dataset(dataset_name: str, data: Dataset, models: dict, checkpoints: dict,
                     batch_size: int, expected_samples: int, split_identifier: str,
                     device: torch.device) -> tuple[list[dict], dict]:
    if len(data) != expected_samples:
        raise RuntimeError(f"{dataset_name}: expected {expected_samples} test samples, found {len(data)}")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    correct = defaultdict(int)
    total = 0
    source_ids = []
    nonbinary_cells = 0
    differing_cells = 0
    validation_batch = None
    validation_predictions = {}
    for frames, labels, ids in loader:
        if validation_batch is None:
            validation_batch = (frames[:min(8, len(frames))].clone(), labels[:min(8, len(labels))].clone())
        source_ids.extend(map(int, ids.tolist()))
        total += len(labels)
        # The integer path retains benchmark count/count-derived amplitudes; the
        # binary path is occupancy. DVS/CIFAR counts have the frozen normalization.
        binary = frames.gt(0).to(torch.float32)
        nonbinary_cells += int(((frames != 0) & (frames != 1)).sum())
        differing_cells += int((frames != binary).sum())
        labels_gpu = labels.to(device, non_blocking=True)
        for representation, inputs in (("integer", frames), ("binary", binary)):
            inputs_gpu = inputs.to(device, non_blocking=True)
            for seed, model in models.items():
                prediction = model(inputs_gpu).argmax(1)
                correct[(representation, seed)] += int((prediction == labels_gpu).sum())
                if total == len(labels):
                    validation_predictions[(representation, seed)] = prediction[:min(8, len(prediction))].cpu()
    if total != expected_samples or len(source_ids) != expected_samples or len(set(source_ids)) != expected_samples:
        raise RuntimeError(f"{dataset_name}: test traversal was incomplete or duplicated")

    # Repeat a fixed, unfiltered prefix after the complete pass.
    repeat_ok = True
    frames, _ = validation_batch
    binary = frames.gt(0).to(torch.float32)
    for representation, inputs in (("integer", frames), ("binary", binary)):
        inputs_gpu = inputs.to(device)
        for seed, model in models.items():
            repeated = model(inputs_gpu).argmax(1).cpu()
            repeat_ok &= torch.equal(repeated, validation_predictions[(representation, seed)])
    if not repeat_ok:
        raise RuntimeError(f"{dataset_name}: repeated-prefix predictions were not deterministic")
    if differing_cells == 0:
        raise RuntimeError(f"{dataset_name}: binary and integer preprocessing were identical")

    rows = []
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            value = correct[(representation, seed)]
            rows.append({
                "dataset": dataset_name,
                "representation": representation,
                "seed": seed,
                "checkpoint": checkpoints[seed],
                "test_samples": total,
                "correct": value,
                "accuracy_percent": 100.0 * value / total,
            })
    audit = {
        "test_split_identifier": split_identifier,
        "expected_test_samples": expected_samples,
        "evaluated_test_samples": total,
        "source_index_min": min(source_ids),
        "source_index_max": max(source_ids),
        "unique_source_indices": len(set(source_ids)),
        "attack_manifest_used": False,
        "prediction_filtering_used": False,
        "model_eval_mode": all(not model.training for model in models.values()),
        "checkpoint_seed_verified": True,
        "checkpoint_hash_verified_against_frozen_specification": True,
        "binary_integer_differing_cells": differing_cells,
        "integer_nonbinary_cells": nonbinary_cells,
        "repeated_unfiltered_prefix_samples": len(validation_batch[1]),
        "repeated_prefix_predictions_deterministic": repeat_ok,
    }
    return rows, audit


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    configured = Path(json.loads((ROOT / "ResearchLoop/config.json").read_text())["python"]).resolve()
    if Path(sys.executable).resolve() != configured:
        raise RuntimeError(f"wrong interpreter: expected {configured}")
    if not torch.cuda.is_available():
        raise RuntimeError("frozen benchmark device is CUDA, but CUDA is unavailable")
    set_determinism()
    device = torch.device("cuda")
    specification = json.loads((ROOT / "Reports/benchmark_specification.json").read_text())
    expected_hashes = {
        dataset: {int(item["seed"]): item["checkpoint_sha256"] for item in details["checkpoints"]}
        for dataset, details in specification["datasets"].items()
    }

    from tonic.datasets import NMNIST
    nmnist = NMNISTOfficialTest(NMNIST(save_to=str(ROOT / "data/nmnist"), train=False))
    dvs_payload = np.load(ROOT / "Reports/checkpoints/dvs_gesture_test_t10_64_float16.npz")
    dvs = ArrayTestSet(dvs_payload["frames"], dvs_payload["labels"])
    cifar_frames = np.load(ROOT / "Reports/checkpoints/cifar10_dvs_t10_128_float16.npy", mmap_mode="r")
    cifar_labels = np.load(ROOT / "Reports/checkpoints/cifar10_dvs_labels.npy", mmap_mode="r")
    cifar_split = json.loads((ROOT / "results/cifar10_dvs_snn_seed42_split.json").read_text())
    cifar = ArrayTestSet(cifar_frames, cifar_labels, cifar_split["test_indices"])

    definitions = (
        ("N-MNIST", nmnist, lambda: NMNISTConvSNN(0.5), 128, 10000,
         "Tonic N-MNIST official test partition (train=False)"),
        ("DVS-Gesture", dvs, lambda: DVSGestureConvSNN(0.5), 16, 264,
         "Tonic DVS-Gesture official test partition (train=False)"),
        ("CIFAR10-DVS", cifar, lambda: CIFAR10DVSConvSNN(0.5), 16, 1000,
         "Frozen seed-42 class-stratified 10% test split (dataset has no official train/test partition)"),
    )
    rows, audits = [], {}
    for name, data, factory, batch, expected, split_id in definitions:
        models, checkpoints = load_models(name, factory, device, expected_hashes[name])
        dataset_rows, audit = evaluate_dataset(name, data, models, checkpoints, batch, expected,
                                               split_id, device)
        rows.extend(dataset_rows)
        audits[name] = audit
        print(f"{name}: PASS ({expected} complete test samples)", flush=True)
        del models
        torch.cuda.empty_cache()

    by_condition = defaultdict(list)
    for row in rows:
        by_condition[(row["dataset"], row["representation"])].append(float(row["accuracy_percent"]))
    summaries = []
    for (dataset, representation), values in by_condition.items():
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_CRITICAL_DF2_975 * sd / math.sqrt(len(values))
        summaries.append({
            "dataset": dataset,
            "representation": representation,
            "mean_accuracy_percent": mean,
            "std_accuracy_percent": sd,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "seeds": "42;123;777",
        })

    result_fields = ["dataset", "representation", "seed", "checkpoint", "test_samples",
                     "correct", "accuracy_percent"]
    summary_fields = ["dataset", "representation", "mean_accuracy_percent", "std_accuracy_percent",
                      "ci95_low", "ci95_high", "seeds"]
    write_csv(ROOT / "Reports/results/clean_accuracy_by_seed.csv", rows, result_fields)
    write_csv(ROOT / "Reports/results/clean_accuracy_summary.csv", summaries, summary_fields)

    row_index = {(row["dataset"], row["representation"], row["seed"]): row for row in rows}
    summary_index = {(row["dataset"], row["representation"]): row for row in summaries}
    report = [
        "# Full-test clean accuracy report", "",
        "Clean accuracy was evaluated without attacks on every sample in each frozen benchmark test split. "
        "No attack clean-correct manifest, prediction filtering, class balancing, or subsampling was used.", "",
        "## Results", "",
        "| Dataset | Representation | Seed 42 | Seed 123 | Seed 777 | Mean ± sample SD | 95% t CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, *_ in definitions:
        for representation in REPRESENTATIONS:
            summary = summary_index[(dataset, representation)]
            values = [row_index[(dataset, representation, seed)]["accuracy_percent"] for seed in SEEDS]
            report.append(
                f"| {dataset} | {representation.title()}-grid | "
                f"{values[0]:.4f}% | {values[1]:.4f}% | {values[2]:.4f}% | "
                f"{summary['mean_accuracy_percent']:.4f}% ± {summary['std_accuracy_percent']:.4f}% | "
                f"[{summary['ci95_low']:.4f}%, {summary['ci95_high']:.4f}%] |"
            )
    report += ["", "## Run details", ""]
    for row in rows:
        report.append(
            f"- **{row['dataset']} / {row['representation']} / seed {row['seed']}**: "
            f"{row['correct']}/{row['test_samples']} correct = {row['accuracy_percent']:.4f}%; "
            f"checkpoint `{row['checkpoint']}`; split: {audits[row['dataset']]['test_split_identifier']}."
        )
    report += ["", "## Sanity checks", ""]
    for dataset, audit in audits.items():
        report.append(
            f"- **{dataset}: PASS.** Evaluated {audit['evaluated_test_samples']}/"
            f"{audit['expected_test_samples']} expected unique samples; no manifest or prediction filtering; "
            f"models in eval mode; checkpoint seed/hash verified; binary and integer tensors differed in "
            f"{audit['binary_integer_differing_cells']:,} cells; predictions were identical on a repeated "
            f"unfiltered {audit['repeated_unfiltered_prefix_samples']}-sample prefix."
        )
    report += [
        "", "## Statistical note", "",
        "The summary uses the arithmetic mean and sample standard deviation (`ddof=1`) across the three "
        "requested seeds. The 95% interval is a two-sided Student-t interval with df=2. With only three "
        "seeds it is mathematically defined but very imprecise and should be interpreted cautiously.", "",
        "CIFAR10-DVS has no official train/test partition. Accordingly, its complete test set here is the "
        "frozen class-stratified 10% benchmark split, not an attack subset.", "",
        "## Statistical workflow reference", "",
        "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: "
        "A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. "
        "https://doi.org/10.48550/arXiv.2609.00065", "",
    ]
    atomic_text(ROOT / "Reports/clean_accuracy_report.md", "\n".join(report))
    atomic_text(ROOT / "Reports/results/clean_accuracy_sanity_checks.json",
                json.dumps({"status": "PASS", "device": str(device), "audits": audits}, indent=2) + "\n")
    print("Wrote clean-accuracy CSVs and report", flush=True)


if __name__ == "__main__":
    main()
