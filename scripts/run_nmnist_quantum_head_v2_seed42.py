"""Targeted Seed-42 validation-only quantum-head optimization for N-MNIST."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.hybrid_data import CachedFrameDataset
from models.nmnist_hybrid_qsnn import NMNISTHybridQSNNV2
from scripts.run_nmnist_hybrid_qsnn_seed42 import (
    CACHE, CHECKPOINT_DIR, RESULT_DIR, SEED, SPLIT_SHA256, make_loader, seed_everything, sha256,
)

VARIANTS = {
    "opt2": {"n_blocks": 2},
    "measure2": {"n_blocks": 2, "learned_measurement": True},
    "project_measure2": {
        "n_blocks": 2, "learned_projection": True,
        "two_axis_encoding": True, "learned_measurement": True,
    },
    "project_measure3_alt": {
        "n_blocks": 3, "learned_projection": True, "two_axis_encoding": True,
        "learned_measurement": True, "topology": "alternating",
    },
}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, targets, predictions = 0.0, [], []
    for frames, labels in loader:
        frames = frames.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames)
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    return {"loss": loss_sum / len(targets),
            "accuracy": float(np.mean(targets == predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0))}


def train(variant, epochs, stage, train_set, validation_set):
    seed_everything()
    model = NMNISTHybridQSNNV2(**VARIANTS[variant])
    classical = torch.load(CHECKPOINT_DIR / "classical_best.pt", map_location="cpu", weights_only=True)
    extractor = {key.removeprefix("extractor."): value for key, value in classical["model_state"].items()
                 if key.startswith("extractor.")}
    model.extractor.load_state_dict(extractor)
    model = model.cuda()
    epoch_offset = 0
    resume_payload = None
    if stage == "extend":
        source = CHECKPOINT_DIR / f"v2_{variant}_full_best.pt"
        resume_payload = torch.load(source, map_location="cuda", weights_only=True)
        model.load_state_dict(resume_payload["model_state"])
        epoch_offset = int(resume_payload["epoch"])
    extractor_lr, quantum_lr, head_lr = ((5e-5, 2.5e-4, 5e-4) if stage == "extend"
                                         else (1e-4, 5e-4, 3e-3))
    optimizer = torch.optim.AdamW([
        {"params": model.extractor.parameters(), "lr": extractor_lr},
        {"params": model.quantum.parameters(), "lr": quantum_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    validation_loader = make_loader(validation_set, 256, False)
    name = f"v2_{variant}_{stage}"
    checkpoint = CHECKPOINT_DIR / f"{name}_best.pt"
    if resume_payload is None:
        best_accuracy, best_loss, best_epoch = -1.0, float("inf"), 0
    else:
        best_accuracy = float(resume_payload["validation"]["accuracy"])
        best_loss = float(resume_payload["validation"]["loss"])
        best_epoch = epoch_offset
        torch.save({**resume_payload, "continued_from": str(source.relative_to(ROOT))}, checkpoint)
    stale = 0
    history = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        absolute_epoch = epoch_offset + epoch
        warmup = stage != "extend" and epoch <= 2
        if stage != "extend" and epoch == 3:
            optimizer.param_groups[2]["lr"] = 1e-3
        model.extractor.requires_grad_(not warmup)
        model.train()
        # Preserve the validated classical extractor's BatchNorm statistics.
        model.extractor.eval()
        loader = make_loader(train_set, 256, True, absolute_epoch)
        loss_sum = correct = seen = 0
        grad_sums = {"extractor": 0.0, "quantum": 0.0, "head": 0.0}
        for frames, labels in loader:
            frames, labels = frames.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frames)
            loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for group_name, module in (("extractor", model.extractor), ("quantum", model.quantum),
                                       ("head", model.head)):
                grad_sums[group_name] += sum(float(p.grad.float().norm()) for p in module.parameters()
                                             if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        validation = evaluate(model, validation_loader)
        if not warmup:
            scheduler.step(validation["accuracy"])
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], absolute_epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": absolute_epoch, "validation": validation,
                        "variant": variant, "architecture": VARIANTS[variant], "seed": SEED,
                        "official_test_accessed": False}, checkpoint)
        else:
            stale += 1
        torch.cuda.synchronize()
        row = {"epoch": absolute_epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1": validation["macro_f1"], "extractor_lr": optimizer.param_groups[0]["lr"],
               "quantum_lr": optimizer.param_groups[1]["lr"], "head_lr": optimizer.param_groups[2]["lr"],
               **{f"{key}_grad_sum": value for key, value in grad_sums.items()}}
        history.append(row)
        print(f"[{name}] " + " ".join(f"{key}={value:.6f}" if isinstance(value, float)
                                       else f"{key}={value}" for key, value in row.items()), flush=True)
        if epoch >= 8 and stale >= (3 if stage == "screen" else 6):
            break
    runtime = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(payload["model_state"])
    validation = evaluate(model, validation_loader)
    with (RESULT_DIR / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    result = {"variant": variant, "stage": stage, "architecture": VARIANTS[variant],
              "best_epoch": payload["epoch"], "stopping_epoch": history[-1]["epoch"],
              "validation": validation,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "runtime_seconds": runtime, "batch_size": 256,
              "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
              "official_test_accessed": False}
    (RESULT_DIR / f"{name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--stage", choices=("screen", "full", "extend"), default="screen")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    manifest_path = ROOT / "results" / "nmnist_snn_multiseed_split.json"
    if sha256(manifest_path) != SPLIT_SHA256:
        raise RuntimeError("Frozen split hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Train/validation overlap.")
    train_set = CachedFrameDataset(CACHE, train_ids)
    validation_set = CachedFrameDataset(CACHE, validation_ids)
    train(args.variant, args.epochs or (12 if args.stage == "screen" else 30),
          args.stage, train_set, validation_set)


if __name__ == "__main__":
    main()
