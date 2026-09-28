"""Train the three published architectures under the local custom model's frozen recipe.

The existing custom checkpoints are reused only after their data/configuration gates pass.
This is a new controlled local experiment, not reproduction of paper checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.controlled_models import BatchMajorReference
from experiments.nmnist.reference_models.resnet import ResNet18
from experiments.nmnist.reference_models.simplenet import SimpleNet_v2
from experiments.nmnist.reference_models.vgg import VGGSNN
from experiments.nmnist.snn_baseline import stratified_train_validation_indices
from models.nmnist_snn import NMNISTConvSNN
from scripts.train_nmnist_seed42_paper_aligned import (
    CachedBinary, Logger, atomic_json, evaluate, make_loader, set_determinism, sha256, train_epoch,
)

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
OUT = ROOT / "Reports/results/nmnist_controlled_four_models_lr1e4"
STATE = ROOT / "Reports/checkpoints/nmnist_controlled_four_models_lr1e4"
CONFIG = ROOT / "configs/nmnist_controlled_four_models_lr1e4.json"
MODELS = {"custom": NMNISTConvSNN, "convnet": SimpleNet_v2, "resnet18": ResNet18, "vggsnn": VGGSNN}


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def run(model_name: str, representation: str) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_result = ROOT / f"Reports/results/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}/seed42_clean_result.json"
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if source["seed"] != 42 or not source["clean_gate_pass"]:
        raise RuntimeError("custom model clean gate missing")
    train_path = ROOT / source["train_cache"]["path"]
    test_path = ROOT / source["test_cache"]["path"]
    for path, item in ((train_path, source["train_cache"]), (test_path, source["test_cache"])):
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"cache hash mismatch: {path}")
    source_state = ROOT / f"Reports/checkpoints/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}"
    train_set = CachedBinary(train_path, source_state / "train_labels_int8.npy")
    test_set = CachedBinary(test_path, source_state / "test_labels_int8.npy")
    train_ids, val_ids = stratified_train_validation_indices(np.asarray(train_set.labels), 500, 42)
    if len(train_ids) != 55000 or len(val_ids) != 5000:
        raise RuntimeError("controlled split mismatch")
    run_id = f"{representation}_{model_name}_seed42"
    result_path = OUT / f"{run_id}.json"
    if result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if (old.get("status") == "PASS" and old["test"]["accuracy"] >= 0.95
                and sha256(ROOT / old["checkpoint_path"]) == old["checkpoint_sha256"]):
            print(f"SKIP {run_id}: complete checkpoint hash verified", flush=True)
            return
        if old.get("test", {}).get("accuracy", 1.0) < 0.95:
            raise RuntimeError(f"{run_id}: existing clean accuracy below 95%; revise the common recipe")
    log = Logger(ROOT / f"Reports/logs/nmnist_controlled_four_models_lr1e4/{run_id}.log")
    try:
        set_determinism(42)
        model = (NMNISTConvSNN(0.5, 10) if model_name == "custom" else
                 BatchMajorReference(MODELS[model_name](num_classes=10, img_size=(2, 34, 34)))).cuda()
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        history = []
        best_accuracy = -1.0
        best_loss = float("inf")
        best_path = None
        stale = 0
        started = time.perf_counter()
        log(f"CONTROLLED START {run_id} parameters={parameters} train=55000 val=5000 batch=64")
        last_path = STATE / f"{run_id}_last.pt"
        first_epoch = 1
        if last_path.exists():
            last = torch.load(last_path, map_location="cuda", weights_only=True)
            if (last.get("run_id") != run_id or last.get("config_sha256") != sha256(CONFIG)
                    or last.get("train_cache_sha256") != source["train_cache"]["sha256"]):
                raise RuntimeError("last checkpoint provenance mismatch")
            model.load_state_dict(last["model_state"], strict=True)
            optimizer.load_state_dict(last["optimizer_state"])
            history = last["history"]
            best_path = ROOT / last["best_path"]
            best_accuracy, best_loss, stale = last["best_accuracy"], last["best_loss"], last["stale"]
            first_epoch = int(last["epoch"]) + 1
            log(f"CONTROLLED RESUME {run_id} epoch={first_epoch} previous_best={best_path}")
        remaining_epochs = range(first_epoch, config["epochs"] + 1)
        if first_epoch - 1 >= config["minimum_epochs"] and stale >= config["patience"]:
            remaining_epochs = ()
        for epoch in remaining_epochs:
            training = train_epoch(model, make_loader(train_set, train_ids, True, 42 + epoch), optimizer, torch.device("cuda"), epoch, config["epochs"], log)
            validation, *_ = evaluate(model, make_loader(train_set, val_ids, False, 42), torch.device("cuda"), f"CONTROLLED VALIDATION epoch={epoch}", log)
            improved = validation["accuracy"] > best_accuracy or (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss)
            history.append({"epoch": epoch, "train_accuracy": training["accuracy"], "train_loss": training["loss"], "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"]})
            if improved:
                best_accuracy, best_loss, stale = validation["accuracy"], validation["loss"], 0
                best_path = STATE / f"{run_id}_best_epoch{epoch:02d}.pt"
                atomic_torch(best_path, {"model_state": model.state_dict(), "seed": 42, "epoch": epoch, "model": model_name, "representation": representation, "config_sha256": sha256(CONFIG), "validation_accuracy": best_accuracy, "validation_loss": best_loss})
            else:
                stale += 1
            atomic_torch(last_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history, "best_path": str(best_path.relative_to(ROOT)), "stale": stale, "best_accuracy": best_accuracy, "best_loss": best_loss, "run_id": run_id, "config_sha256": sha256(CONFIG), "train_cache_sha256": source["train_cache"]["sha256"]})
            log(f"CONTROLLED EPOCH {epoch} best_accuracy={best_accuracy:.5f} stale={stale}")
            if epoch >= config["minimum_epochs"] and stale >= config["patience"]:
                break
        if best_path is None:
            raise RuntimeError("no validation-selected checkpoint")
        checkpoint = torch.load(best_path, map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        test, labels, predictions, sample_ids = evaluate(model, make_loader(test_set, None, False, 42), torch.device("cuda"), "CONTROLLED FINAL TEST", log)
        OUT.mkdir(parents=True, exist_ok=True)
        prediction_path = OUT / f"{run_id}_predictions.npz"
        temp_prediction = prediction_path.with_name(prediction_path.name + f".{os.getpid()}.tmp")
        with temp_prediction.open("wb") as stream:
            np.savez_compressed(stream, sample_ids=sample_ids, labels=labels.astype(np.int8), predictions=predictions.astype(np.int8))
        os.replace(temp_prediction, prediction_path)
        history_path = OUT / f"{run_id}_history.csv"
        with history_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        source_model_path = (ROOT / "models/nmnist_snn.py" if model_name == "custom" else
                             ROOT / f"experiments/nmnist/reference_models/{'simplenet' if model_name == 'convnet' else 'resnet' if model_name == 'resnet18' else 'vgg'}.py")
        result = {"status": "PASS" if test["accuracy"] >= 0.95 else "INSUFFICIENT_CLEAN_ACCURACY", "run_id": run_id, "model": model_name, "representation": representation,
                  "seed": 42, "parameters": parameters, "test": test, "best_epoch": checkpoint["epoch"],
                  "checkpoint_path": str(best_path.relative_to(ROOT)).replace("\\", "/"), "checkpoint_sha256": sha256(best_path),
                  "predictions_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"), "predictions_sha256": sha256(prediction_path),
                  "train_cache": source["train_cache"], "test_cache": source["test_cache"],
                  "train_samples": 55000, "validation_samples": 5000, "test_samples": 10000,
                  "training_config_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"), "training_config_sha256": sha256(CONFIG),
                  "history_path": str(history_path.relative_to(ROOT)).replace("\\", "/"), "history_sha256": sha256(history_path),
                  "model_source_path": str(source_model_path.relative_to(ROOT)).replace("\\", "/"), "model_source_sha256": sha256(source_model_path),
                  "runtime_seconds": time.perf_counter() - started, "exact_command": " ".join([str(PYTHON), *sys.argv])}
        atomic_json(result_path, result)
        log(f"CONTROLLED COMPLETE {run_id} test_accuracy={test['accuracy']:.6f} checkpoint={sha256(best_path)}")
        if result["status"] != "PASS":
            raise RuntimeError(f"clean accuracy {test['accuracy']:.4%} below 95% gate; revise the common training recipe before attack comparison")
    finally:
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--representation", required=True, choices=("binary", "integer"))
    args = parser.parse_args()
    run(args.model, args.representation)
