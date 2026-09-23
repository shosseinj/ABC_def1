"""Train the validation-only seed-42 N-MNIST classical control and hybrid QSNN."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.hybrid_data import CachedFrameDataset, build_frame_cache
from models.nmnist_hybrid_qsnn import ClassicalLatentClassifier, NMNISTHybridQSNN

SEED = 42
SPLIT_SHA256 = "a12176ce117ab9a85dd29d277901f8617ad2b5427dd3f976dbba436942f70198"
CACHE = ROOT / "data" / "nmnist" / "nmnist_train_t10_frames.npy"
RESULT_DIR = ROOT / "results" / "nmnist_hybrid_qsnn_seed42"
CHECKPOINT_DIR = ROOT / "checkpoints" / "nmnist_hybrid_qsnn_seed42"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(dataset, batch_size, shuffle, epoch=0):
    generator = torch.Generator().manual_seed(SEED + epoch)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=4, persistent_workers=True, pin_memory=True,
        prefetch_factor=3, drop_last=False,
    )


def amp_context():
    return torch.autocast("cuda", dtype=torch.float16)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, labels_all, predictions = 0.0, [], []
    for frames, labels in loader:
        frames = frames.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with amp_context():
            logits = model(frames)
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        labels_all.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    labels_all = np.asarray(labels_all)
    predictions = np.asarray(predictions)
    return {
        "loss": loss_sum / len(labels_all),
        "accuracy": float(np.mean(labels_all == predictions)),
        "macro_f1": float(f1_score(labels_all, predictions, average="macro", zero_division=0)),
    }


def benchmark_batches(model_factory, dataset, candidates=(32, 64, 128, 256), steps=30):
    rows = []
    for batch_size in candidates:
        seed_everything()
        model = model_factory().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        loader = make_loader(dataset, batch_size, True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        iterator = iter(loader)
        started = None
        samples = 0
        try:
            for step in range(steps + 5):
                frames, labels = next(iterator)
                frames = frames.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                if step == 5:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with amp_context():
                    loss = F.cross_entropy(model(frames), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if step >= 5:
                    samples += len(labels)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append({"batch_size": batch_size, "samples_per_second": samples / elapsed,
                         "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "oom": False})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": batch_size, "samples_per_second": 0.0,
                         "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "oom": True})
        del model, optimizer, loader
        torch.cuda.empty_cache()
    valid = [row for row in rows if not row["oom"]]
    return max(valid, key=lambda row: row["samples_per_second"])["batch_size"], rows


def train_model(name, model, train_set, validation_set, batch_size, epochs, learning_rate,
                patience, initial_extractor=None, freeze_extractor=False):
    seed_everything()
    model = model.cuda()
    if initial_extractor is not None:
        model.extractor.load_state_dict(initial_extractor)
    if freeze_extractor:
        model.extractor.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=2, min_lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    validation_loader = make_loader(validation_set, batch_size, False)
    checkpoint = CHECKPOINT_DIR / f"{name}_best.pt"
    history, best_accuracy, best_loss, stale = [], -1.0, float("inf"), 0
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_loader = make_loader(train_set, batch_size, True, epoch)
        model.train()
        if freeze_extractor:
            model.extractor.eval()
        loss_sum = correct = seen = 0
        for frames, labels in train_loader:
            frames = frames.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context():
                logits = model(frames)
                loss = F.cross_entropy(logits, labels)
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
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)
        if improved:
            best_accuracy, best_loss, stale = validation["accuracy"], validation["loss"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "validation": validation, "batch_size": batch_size, "seed": SEED,
                        "official_test_accessed": False}, checkpoint)
        else:
            stale += 1
        torch.cuda.synchronize()
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "validation_macro_f1": validation["macro_f1"],
               "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        print(f"[{name}] " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in row.items()), flush=True)
        if epoch >= 8 and stale >= patience:
            break
    runtime = time.perf_counter() - training_started
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(payload["model_state"])
    final_validation = evaluate(model, validation_loader)
    with (RESULT_DIR / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    result = {"name": name, "best_epoch": payload["epoch"], "stopping_epoch": history[-1]["epoch"],
              "validation": final_validation,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "batch_size": batch_size, "training_seconds": runtime,
              "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
              "official_test_accessed": False}
    (RESULT_DIR / f"{name}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return model, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cache", "classical", "quantum", "all"), default="all")
    parser.add_argument("--classical-epochs", type=int, default=25)
    parser.add_argument("--quantum-epochs", type=int, default=25)
    parser.add_argument("--quantum-variant", choices=("joint", "prob1-frozen"), default="joint")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results" / "nmnist_snn_multiseed_split.json"
    if sha256(manifest_path) != SPLIT_SHA256:
        raise RuntimeError("Frozen split hash mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = np.asarray(manifest["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_indices"], dtype=np.int64)
    if np.intersect1d(train_ids, validation_ids).size:
        raise RuntimeError("Train/validation overlap.")
    if not CACHE.exists():
        from tonic.datasets import NMNIST
        # Intentionally instantiate only the official training partition.
        dataset = NMNIST(save_to=str(ROOT / "data" / "nmnist"), train=True)
        build_frame_cache(dataset, CACHE, temporal_bins=10)
    if args.mode == "cache":
        return
    train_set = CachedFrameDataset(CACHE, train_ids)
    validation_set = CachedFrameDataset(CACHE, validation_ids)
    summary = {"seed": SEED, "split_sha256": SPLIT_SHA256, "official_test_accessed": False,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "amp": "float16"}
    classical_checkpoint = CHECKPOINT_DIR / "classical_best.pt"
    if args.mode in ("classical", "all"):
        classical_batch, benchmark = benchmark_batches(ClassicalLatentClassifier, train_set)
        summary["classical_batch_benchmark"] = benchmark
        classical_model, classical_result = train_model(
            "classical", ClassicalLatentClassifier(), train_set, validation_set,
            classical_batch, args.classical_epochs, 1e-3, 5)
        summary["classical"] = classical_result
    if args.mode in ("quantum", "all"):
        if not classical_checkpoint.exists():
            raise RuntimeError("Train the classical control before the QSNN.")
        classical_payload = torch.load(classical_checkpoint, map_location="cpu", weights_only=True)
        extractor_state = {key.removeprefix("extractor."): value for key, value in
                           classical_payload["model_state"].items() if key.startswith("extractor.")}
        frozen = args.quantum_variant == "prob1-frozen"
        quantum_factory = lambda: NMNISTHybridQSNN(quantum_blocks=1 if frozen else 2)
        quantum_batch, benchmark = benchmark_batches(quantum_factory, train_set)
        summary["quantum_batch_benchmark"] = benchmark
        run_name = "quantum_prob1_frozen" if frozen else "quantum_joint"
        _, quantum_result = train_model(
            run_name, quantum_factory(), train_set, validation_set,
            quantum_batch, args.quantum_epochs, 1e-3 if frozen else 5e-4,
            5 if frozen else 6, extractor_state, frozen)
        summary["quantum"] = quantum_result
    suffix = args.quantum_variant.replace("-", "_") if args.mode in ("quantum", "all") else "classical"
    (RESULT_DIR / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
