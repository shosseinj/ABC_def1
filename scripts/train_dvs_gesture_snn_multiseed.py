"""Resumable five-seed DVS-Gesture clean SNN training."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dvs_gesture_snn import DVSGestureConvSNN


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def frames(events, bins=10, size=64):
    out = np.zeros((bins, 2, size, size), dtype=np.float32)
    if not len(events): return out.astype(np.float16)
    x = np.asarray(events["x"], dtype=np.int64) // 2
    y = np.asarray(events["y"], dtype=np.int64) // 2
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    duration = max(int(t.max()) - int(t.min()) + 1, 1)
    tb = np.minimum(((t - t.min()) * bins) // duration, bins - 1)
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size) & ((p == 0) | (p == 1))
    np.add.at(out, (tb[valid], p[valid], y[valid], x[valid]), 1)
    maximum = float(out.max())
    if maximum: out /= maximum
    return out.astype(np.float16)


def cache_partition(dataset, name, config):
    path = ROOT / "Reports/checkpoints" / f"dvs_gesture_{name}_t10_64_float16.npz"
    if path.exists():
        saved = np.load(path)
        if len(saved["labels"]) == len(dataset):
            print(f"[{name}] reusing cache {path} sha256={sha256(path)}", flush=True)
            return saved["frames"], saved["labels"], path
    xs = np.empty((len(dataset), 10, 2, 64, 64), dtype=np.float16)
    ys = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        events, label = dataset[i]; xs[i] = frames(events); ys[i] = label
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"[{name} preprocessing] {i+1}/{len(dataset)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, frames=xs, labels=ys); os.replace(temporary, path)
    print(f"[{name}] cache sha256={sha256(path)}", flush=True)
    return xs, ys, path


def loader(xs, ys, ids, batch, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
    return DataLoader(Subset(ds, list(map(int, ids))), batch_size=batch, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def evaluate(model, data, device):
    model.eval(); total = correct = 0; loss_sum = 0.; truth = []; pred = []
    with torch.no_grad():
        for x, y in data:
            x, y = x.float().to(device), y.to(device); logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum")); total += len(y)
            prediction = logits.argmax(1); correct += int((prediction == y).sum())
            truth += y.cpu().tolist(); pred += prediction.cpu().tolist()
    return {"loss": loss_sum/total, "accuracy": correct/total,
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)), "samples": total}


def main():
    from tonic.datasets import DVSGesture
    config_path = ROOT / "configs/dvs_gesture_snn_multiseed.json"
    config = json.loads(config_path.read_text())
    train_native = DVSGesture(save_to=str(ROOT/config["data_root"]), train=True)
    test_native = DVSGesture(save_to=str(ROOT/config["data_root"]), train=False)
    train_x, train_y, train_cache = cache_partition(train_native, "train", config)
    test_x, test_y, test_cache = cache_partition(test_native, "test", config)
    indices = np.arange(len(train_y))
    train_ids, validation_ids = train_test_split(indices, test_size=config["validation_fraction"],
        random_state=config["split_seed"], stratify=train_y)
    split = {"train_indices":sorted(map(int,train_ids)), "validation_indices":sorted(map(int,validation_ids)),
             "official_test_indices":list(range(len(test_y))), "split_seed":config["split_seed"]}
    split_path = ROOT/"Reports/results/dvs_gesture_split.json"; atomic_json(split_path, split)
    device=torch.device(config["device"]); summaries=[]
    for seed in config["seeds"]:
        checkpoint=ROOT/f"checkpoints/dvs_gesture_snn_seed{seed}_best.pt"
        marker=ROOT/f"Reports/checkpoints/dvs_gesture_seed{seed}.complete.json"
        if marker.exists() and checkpoint.exists():
            done=json.loads(marker.read_text())
            if done.get("checkpoint_sha256")==sha256(checkpoint):
                print(f"[seed={seed}] valid completion marker; skipping training",flush=True); summaries.append(done); continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model=DVSGestureConvSNN().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        validation=loader(train_x,train_y,validation_ids,config["batch_size"],False,seed)
        best_acc=-1.; best_loss=float("inf"); stale=0; history=[]; started=time.perf_counter()
        for epoch in range(1,config["epochs"]+1):
            model.train(); seen=correct=0; loss_sum=0.; epoch_start=time.perf_counter()
            for x,y in loader(train_x,train_y,train_ids,config["batch_size"],True,seed+epoch):
                x,y=x.float().to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=F.cross_entropy(logits,y); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"]); optimizer.step()
                seen+=len(y); correct+=int((logits.argmax(1)==y).sum()); loss_sum+=float(loss.detach())*len(y)
            val=evaluate(model,validation,device); improved=val["accuracy"]>best_acc or (val["accuracy"]==best_acc and val["loss"]<best_loss)
            if improved:
                best_acc,best_loss,stale=val["accuracy"],val["loss"],0
                torch.save({"model_state":model.state_dict(),"seed":seed,"epoch":epoch,"config":config},checkpoint)
            else: stale+=1
            row={"epoch":epoch,"train_loss":loss_sum/seen,"train_accuracy":correct/seen,"validation_loss":val["loss"],"validation_accuracy":val["accuracy"],"seconds":time.perf_counter()-epoch_start};history.append(row)
            print(f"seed={seed} epoch={epoch} train_acc={row['train_accuracy']:.4f} val_acc={val['accuracy']:.4f} best={best_acc:.4f} seconds={row['seconds']:.1f}",flush=True)
            if epoch>=config["minimum_epochs"] and stale>=config["patience"]: break
        payload=torch.load(checkpoint,map_location=device,weights_only=True); model.load_state_dict(payload["model_state"])
        test=evaluate(model,loader(test_x,test_y,range(len(test_y)),config["batch_size"],False,seed),device)
        done={"status":"COMPLETE","seed":seed,"best_epoch":payload["epoch"],"validation_accuracy":best_acc,"test":test,"checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":sha256(checkpoint),"config_sha256":sha256(config_path),"split_sha256":sha256(split_path),"train_cache_sha256":sha256(train_cache),"test_cache_sha256":sha256(test_cache),"runtime_seconds":time.perf_counter()-started,"parameters":model.trainable_parameter_count()}
        atomic_json(marker,done); summaries.append(done)
        with (ROOT/f"Reports/logs/dvs_gesture_seed{seed}_history.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
        print(json.dumps(done,indent=2),flush=True)
    atomic_json(ROOT/"Reports/results/dvs_gesture_training_summary.json",summaries)


if __name__ == "__main__": main()
