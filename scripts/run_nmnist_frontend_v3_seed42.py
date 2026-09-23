"""Screen stronger N-MNIST Conv/LIF frontends, then train project_measure2 once."""
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
from models.nmnist_hybrid_qsnn import (
    NMNISTHybridQSNNV2, NMNISTSpatialLIFExtractor, SpatialClassicalControl,
)
from scripts.run_nmnist_hybrid_qsnn_seed42 import (
    CACHE, CHECKPOINT_DIR, RESULT_DIR, SPLIT_SHA256, make_loader, seed_everything, sha256,
)

FRONTENDS = {
    "spatial4": {"channels": (12, 24), "spatial_size": 4, "latent_dim": 16},
    "wide4": {"channels": (16, 32), "spatial_size": 4, "latent_dim": 32},
    "spatial8": {"channels": (16, 32), "spatial_size": 8, "latent_dim": 32},
}


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); loss_sum = 0.0; targets = []; predictions = []
    for frames, labels in loader:
        frames, labels = frames.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames)
        loss_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum"))
        targets.extend(labels.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    return {"loss": loss_sum / len(targets), "accuracy": float(np.mean(targets == predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0))}


def build_model(frontend, model_type):
    config = FRONTENDS[frontend]
    if model_type == "classical":
        return SpatialClassicalControl(**config)
    extractor = NMNISTSpatialLIFExtractor(**config)
    return NMNISTHybridQSNNV2(
        extractor=extractor, latent_dim=config["latent_dim"], n_blocks=2,
        learned_projection=True, two_axis_encoding=True, learned_measurement=True,
    )


def train(frontend, model_type, stage, epochs, train_set, validation_set):
    seed_everything()
    model = build_model(frontend, model_type)
    if model_type == "quantum":
        source = CHECKPOINT_DIR / f"v3_{frontend}_classical_full_best.pt"
        payload = torch.load(source, map_location="cpu", weights_only=True)
        state = {key.removeprefix("extractor."): value for key, value in payload["model_state"].items()
                 if key.startswith("extractor.")}
        model.extractor.load_state_dict(state)
    model = model.cuda()
    if model_type == "classical":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.extractor.parameters(), "lr": 5e-5},
            {"params": model.quantum_projection.parameters(), "lr": 2.5e-4},
            {"params": model.quantum.parameters(), "lr": 2.5e-4},
            {"params": model.head.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    validation_loader = make_loader(validation_set, 256, False)
    name = f"v3_{frontend}_{model_type}_{stage}"
    checkpoint = CHECKPOINT_DIR / f"{name}_best.pt"
    best_accuracy, best_loss, stale = -1.0, float("inf"), 0
    history = []; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        warmup = model_type == "quantum" and epoch <= 2
        if model_type == "quantum":
            model.extractor.requires_grad_(not warmup)
        model.train()
        if model_type == "quantum":
            model.extractor.eval()
        loader = make_loader(train_set, 256, True, epoch)
        loss_sum = correct = seen = 0
        for frames, labels in loader:
            frames, labels = frames.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frames)
            loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum()); seen += len(labels)
        validation = evaluate(model, validation_loader)
        scheduler.step(validation["accuracy"])
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)
        if improved:
            best_accuracy, best_loss, stale = validation["accuracy"], validation["loss"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation": validation,
                        "frontend": frontend, "frontend_config": FRONTENDS[frontend],
                        "model_type": model_type, "official_test_accessed": False}, checkpoint)
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1": validation["macro_f1"]}
        history.append(row)
        print(f"[{name}] " + " ".join(f"{key}={value:.6f}" if isinstance(value, float)
                                       else f"{key}={value}" for key, value in row.items()), flush=True)
        if epoch >= 8 and stale >= (3 if stage == "screen" else 6):
            break
    runtime = time.perf_counter() - started
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(payload["model_state"]); validation = evaluate(model, validation_loader)
    with (RESULT_DIR / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    result = {"frontend": frontend, "frontend_config": FRONTENDS[frontend], "model_type": model_type,
              "stage": stage, "validation": validation, "best_epoch": payload["epoch"],
              "stopping_epoch": history[-1]["epoch"],
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "runtime_seconds": runtime, "batch_size": 256,
              "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
              "official_test_accessed": False}
    (RESULT_DIR / f"{name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=FRONTENDS, required=True)
    parser.add_argument("--model", choices=("classical", "quantum"), default="classical")
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    manifest_path = ROOT / "results" / "nmnist_snn_multiseed_split.json"
    if sha256(manifest_path) != SPLIT_SHA256: raise RuntimeError("Frozen split hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if np.intersect1d(train_ids, validation_ids).size: raise RuntimeError("Train/validation overlap.")
    train_set = CachedFrameDataset(CACHE, train_ids); validation_set = CachedFrameDataset(CACHE, validation_ids)
    default_epochs = 12 if args.stage == "screen" else (30 if args.model == "classical" else 40)
    train(args.frontend, args.model, args.stage, args.epochs or default_epochs, train_set, validation_set)


if __name__ == "__main__":
    main()
