"""Frozen five-seed QSNN-v3 validation campaign for N-MNIST.

The architecture and hyperparameters are fixed to the selected Seed-42
wide4 + project_measure2 configuration. This script only trains/evaluates on
the frozen official-training split and never instantiates the official test set.
"""
import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.hybrid_data import CachedFrameDataset
from models.nmnist_hybrid_qsnn import NMNISTHybridQSNNV2, NMNISTSpatialLIFExtractor
from scripts.run_nmnist_hybrid_qsnn_seed42 import (
    CACHE, CHECKPOINT_DIR, RESULT_DIR, SPLIT_SHA256, seed_everything, sha256,
)

SEEDS = (42, 123, 777, 2026, 6543)
FRONTEND_CONFIG = {
    "channels": [16, 32],
    "spatial_size": 4,
    "latent_dim": 32,
}
QUANTUM_CONFIG = {
    "n_blocks": 2,
    "learned_projection": True,
    "two_axis_encoding": True,
    "learned_measurement": True,
    "topology": "ring",
}
CLASSICAL_FRONTEND_CHECKPOINT = CHECKPOINT_DIR / "v3_wide4_classical_full_best.pt"
CLASSICAL_FRONTEND_CHECKPOINT_SHA256 = (
    "11e03f479bcc028882b28d865ec9ae291a05553bc4b1eeada9dd454a57bcfe0a"
)
CAMPAIGN_DIR = RESULT_DIR / "nmnist_qsnn_v3_multiseed"
CAMPAIGN_CHECKPOINT_DIR = CHECKPOINT_DIR / "nmnist_qsnn_v3_multiseed"


def frozen_config():
    return {
        "campaign": "N-MNIST QSNN-v3 frozen five-seed validation campaign",
        "dataset": "N-MNIST",
        "source_partition": "official training",
        "split_manifest": "results/nmnist_snn_multiseed_split.json",
        "split_sha256": SPLIT_SHA256,
        "seeds": list(SEEDS),
        "seed_42_classical_control": {
            "checkpoint": str(CLASSICAL_FRONTEND_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": CLASSICAL_FRONTEND_CHECKPOINT_SHA256,
            "validation_accuracy": 0.978,
            "validation_macro_f1": 0.978022148742341,
        },
        "frontend": "wide4",
        "frontend_config": FRONTEND_CONFIG,
        "quantum_architecture": QUANTUM_CONFIG,
        "latent_to_quantum_projection": "learned Linear(32, 8) with no classical bypass",
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-4,
            "parameter_groups": {
                "extractor": 5e-5,
                "quantum_projection": 2.5e-4,
                "quantum": 2.5e-4,
                "classifier": 1e-3,
            },
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 2,
            "min_lr": 1e-5,
        },
        "training": {
            "batch_size": 256,
            "max_epochs": 40,
            "minimum_epochs": 8,
            "early_stopping_patience": 6,
            "gradient_clip_norm": 1.0,
            "amp": "torch.float16 for Conv/LIF forward; quantum state remains float32/complex64",
            "device": "cuda:0",
            "checkpoint_selection": "highest validation accuracy, then lowest validation loss",
        },
        "attacks_run": False,
        "defenses_run": False,
        "official_test_accessed": False,
    }


def make_loader(dataset, batch_size, shuffle, seed, epoch):
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator,
        num_workers=4, persistent_workers=True, pin_memory=True,
        prefetch_factor=3, drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum = 0.0
    targets, predictions = [], []
    for frames, labels in loader:
        frames = frames.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames)
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "loss": loss_sum / len(targets),
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def build_frozen_model():
    extractor = NMNISTSpatialLIFExtractor(**FRONTEND_CONFIG)
    return NMNISTHybridQSNNV2(
        extractor=extractor,
        latent_dim=FRONTEND_CONFIG["latent_dim"],
        **QUANTUM_CONFIG,
    )


def load_fixed_frontend(model):
    if not CLASSICAL_FRONTEND_CHECKPOINT.exists():
        raise FileNotFoundError(f"Fixed classical frontend checkpoint missing: {CLASSICAL_FRONTEND_CHECKPOINT}")
    actual_hash = sha256(CLASSICAL_FRONTEND_CHECKPOINT)
    if actual_hash != CLASSICAL_FRONTEND_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Fixed frontend checkpoint hash mismatch: {actual_hash} != "
            f"{CLASSICAL_FRONTEND_CHECKPOINT_SHA256}"
        )
    payload = torch.load(CLASSICAL_FRONTEND_CHECKPOINT, map_location="cpu", weights_only=True)
    extractor_state = {
        key.removeprefix("extractor."): value
        for key, value in payload["model_state"].items()
        if key.startswith("extractor.")
    }
    model.extractor.load_state_dict(extractor_state)
    return payload


def train_seed(seed, train_set, validation_set, config):
    seed_everything(seed)
    model = build_frozen_model()
    load_fixed_frontend(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.extractor.parameters(), "lr": 5e-5},
        {"params": model.quantum_projection.parameters(), "lr": 2.5e-4},
        {"params": model.quantum.parameters(), "lr": 2.5e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    validation_loader = make_loader(validation_set, 256, False, seed, 0)
    checkpoint = CAMPAIGN_CHECKPOINT_DIR / f"seed{seed}_best.pt"
    history_path = CAMPAIGN_DIR / f"seed{seed}_history.csv"
    result_path = CAMPAIGN_DIR / f"seed{seed}_result.json"

    best_accuracy, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    history = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, 41):
        warmup = epoch <= 2
        model.extractor.requires_grad_(not warmup)
        model.train()
        model.extractor.eval()
        train_loader = make_loader(train_set, 256, True, seed, epoch)
        loss_sum = correct = seen = 0
        for frames, labels in train_loader:
            frames = frames.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frames)
            loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        validation = evaluate(model, validation_loader)
        scheduler.step(validation["accuracy"])
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation": validation,
                "seed": seed,
                "config": config,
                "official_test_accessed": False,
            }, checkpoint)
        else:
            stale += 1
        torch.cuda.synchronize()
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "quantum_learning_rate": optimizer.param_groups[1]["lr"],
            "classifier_learning_rate": optimizer.param_groups[3]["lr"],
        }
        history.append(row)
        print(
            f"[seed={seed}] epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"train_accuracy={row['train_accuracy']:.6f} "
            f"validation_loss={row['validation_loss']:.6f} "
            f"validation_accuracy={row['validation_accuracy']:.6f} "
            f"validation_macro_f1={row['validation_macro_f1']:.6f} "
            f"best_validation_accuracy={best_accuracy:.6f}",
            flush=True,
        )
        if epoch >= 8 and stale >= 6:
            break
        del train_loader
    runtime = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(payload["model_state"])
    final_validation = evaluate(model, validation_loader)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    result = {
        "seed": seed,
        "frontend": "wide4",
        "frontend_config": FRONTEND_CONFIG,
        "quantum_architecture": QUANTUM_CONFIG,
        "validation": final_validation,
        "best_epoch": payload["epoch"],
        "stopping_epoch": history[-1]["epoch"],
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "batch_size": 256,
        "runtime_seconds": runtime,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "official_test_accessed": False,
        "attacks_run": False,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    del model, optimizer, scaler, validation_loader
    torch.cuda.empty_cache()
    gc.collect()
    return result


def finalize_seed42():
    """Evaluate the already-saved Seed-42 checkpoint after the timeout."""
    checkpoint = CAMPAIGN_CHECKPOINT_DIR / "seed42_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_frozen_model()
    load_fixed_frontend(model)
    model.load_state_dict(payload["model_state"])
    model = model.cuda()
    manifest = json.loads((ROOT / "results/nmnist_snn_multiseed_split.json").read_text(encoding="utf-8"))
    validation_set = CachedFrameDataset(
        CACHE, np.asarray(manifest["validation_indices"], dtype=np.int64))
    validation_loader = DataLoader(
        validation_set, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    validation = evaluate(model, validation_loader)
    result = {
        "seed": 42,
        "frontend": "wide4",
        "frontend_config": FRONTEND_CONFIG,
        "quantum_architecture": QUANTUM_CONFIG,
        "validation": validation,
        "stored_validation": payload["validation"],
        "validation_matches_checkpoint": validation == payload["validation"],
        "best_epoch": payload["epoch"],
        "stopping_epoch": 38,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "batch_size": 256,
        "runtime_seconds": 1200.0,
        "peak_gpu_bytes": None,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "official_test_accessed": False,
        "attacks_run": False,
        "note": "The Seed-42 run reached epoch 38 before the execution timeout; the saved best checkpoint is epoch 37.",
    }
    (CAMPAIGN_DIR / "seed42_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def aggregate(results, config):
    accuracies = np.asarray([row["validation"]["accuracy"] for row in results], dtype=np.float64)
    macro_f1 = np.asarray([row["validation"]["macro_f1"] for row in results], dtype=np.float64)
    runtimes = np.asarray([row["runtime_seconds"] for row in results], dtype=np.float64)
    epochs = np.asarray([row["best_epoch"] for row in results], dtype=np.int64)
    snn = json.loads((ROOT / "results/nmnist_snn_multiseed_summary.json").read_text(encoding="utf-8"))
    summary = {
        "campaign": config["campaign"],
        "seeds": list(SEEDS),
        "runs": results,
        "validation_accuracy": {
            "mean": float(accuracies.mean()),
            "sample_sd": float(accuracies.std(ddof=1)),
        },
        "macro_f1": {
            "mean": float(macro_f1.mean()),
            "sample_sd": float(macro_f1.std(ddof=1)),
        },
        "runtime_seconds": {
            "mean": float(runtimes.mean()),
            "sample_sd": float(runtimes.std(ddof=1)),
        },
        "best_epoch": {
            "mean": float(epochs.mean()),
            "values": {str(row["seed"]): row["best_epoch"] for row in results},
        },
        "snn_reference": {
            "validation_accuracy": snn["validation_accuracy"],
            "macro_f1": snn["macro_f1"],
        },
        "difference_from_snn": {
            "validation_accuracy": float(accuracies.mean() - snn["validation_accuracy"]["mean"]),
            "macro_f1": float(macro_f1.mean() - snn["macro_f1"]["mean"]),
        },
        "parameters": results[0]["parameters"],
        "official_test_accessed": False,
        "attacks_run": False,
        "defenses_run": False,
    }
    (CAMPAIGN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (CAMPAIGN_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "validation_accuracy", "macro_f1", "best_epoch", "runtime_seconds", "parameters", "checkpoint"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "seed": row["seed"],
                "validation_accuracy": row["validation"]["accuracy"],
                "macro_f1": row["validation"]["macro_f1"],
                "best_epoch": row["best_epoch"],
                "runtime_seconds": row["runtime_seconds"],
                "parameters": row["parameters"],
                "checkpoint": row["checkpoint"],
            })
    lines = [
        "# N-MNIST QSNN-v3 Frozen Five-Seed Validation Campaign",
        "",
        "Architecture and hyperparameters were frozen to the selected `wide4 + project_measure2` configuration. No architecture tuning was performed between seeds. The official test partition and attacks were not used.",
        "",
        "| Seed | Validation Accuracy | Macro-F1 | Best Epoch | Runtime (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['seed']} | {100 * row['validation']['accuracy']:.2f}% | "
            f"{100 * row['validation']['macro_f1']:.2f}% | {row['best_epoch']} | "
            f"{row['runtime_seconds']:.2f} |"
        )
    lines.extend([
        "",
        f"Validation accuracy mean +/- sample SD: {100 * summary['validation_accuracy']['mean']:.4f}% +/- {100 * summary['validation_accuracy']['sample_sd']:.4f}%",
        f"Macro-F1 mean +/- sample SD: {100 * summary['macro_f1']['mean']:.4f}% +/- {100 * summary['macro_f1']['sample_sd']:.4f}%",
        "",
        f"SNN reference validation accuracy mean +/- sample SD: {100 * snn['validation_accuracy']['mean']:.4f}% +/- {100 * snn['validation_accuracy']['sample_sd']:.4f}%",
        f"SNN reference Macro-F1 mean +/- sample SD: {100 * snn['macro_f1']['mean']:.4f}% +/- {100 * snn['macro_f1']['sample_sd']:.4f}%",
        "",
        f"QSNN minus SNN validation accuracy: {100 * summary['difference_from_snn']['validation_accuracy']:.4f} percentage points.",
        f"QSNN minus SNN Macro-F1: {100 * summary['difference_from_snn']['macro_f1']:.4f} percentage points.",
    ])
    (CAMPAIGN_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--finalize-seed42", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"Seeds must be selected from {SEEDS}")
    if args.finalize_seed42:
        finalize_seed42()
        return
    if args.aggregate_only:
        results = []
        for seed in SEEDS:
            path = CAMPAIGN_DIR / f"seed{seed}_result.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing completed seed result: {path}")
            results.append(json.loads(path.read_text(encoding="utf-8")))
        if len(results) != len(SEEDS):
            raise RuntimeError("Aggregate requires all five seed result files.")
        aggregate(results, frozen_config())
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results/nmnist_snn_multiseed_split.json"
    if sha256(manifest_path) != SPLIT_SHA256:
        raise RuntimeError("Frozen split hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if len(train_ids) != 55000 or len(validation_ids) != 5000 or np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Frozen split integrity check failed.")
    if not CACHE.exists():
        raise FileNotFoundError(f"Cached event frames missing: {CACHE}")
    config = frozen_config()
    (CAMPAIGN_DIR / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_set = CachedFrameDataset(CACHE, train_ids)
    validation_set = CachedFrameDataset(CACHE, validation_ids)
    results = []
    for seed in args.seeds:
        results.append(train_seed(seed, train_set, validation_set, config))
    if len(results) == len(SEEDS):
        aggregate(results, config)
    else:
        print(f"[PARTIAL] {len(results)} seed result(s) completed; aggregate waits for all five.", flush=True)


if __name__ == "__main__":
    main()
