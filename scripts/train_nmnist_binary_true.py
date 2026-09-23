"""Train true Binary-grid N-MNIST SNNs from scratch and evaluate full official test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import events_to_frames, stratified_train_validation_indices
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
SEEDS = (42, 123, 777)
OUT = ROOT / "Reports/results/nmnist_binary_true"
CHECKPOINTS = ROOT / "checkpoints/nmnist_binary_true"
LOGS = ROOT / "Reports/logs/nmnist_binary_true"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class BinaryNMNIST(Dataset):
    """Exact benchmark Binary semantics: nonzero temporal cell -> 1."""
    def __init__(self, native): self.native = native
    def __len__(self): return len(self.native)
    def __getitem__(self, index):
        events, label = self.native[index]
        counts = events_to_frames(events, 10)
        binary = (counts != 0).astype(np.float32)
        if not np.all((binary == 0) | (binary == 1)):
            raise RuntimeError("Binary preprocessing leaked a non-binary amplitude")
        return torch.from_numpy(binary), int(label)


def loader(dataset, ids, batch, shuffle, seed):
    return DataLoader(Subset(dataset, list(map(int, ids))), batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, data, device, classes=10):
    model.eval(); labels=[]; predictions=[]; loss=0.0
    for x, y in data:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True); logits=model(x)
        loss += float(F.cross_entropy(logits, y, reduction="sum")); labels.extend(y.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
    labels=np.asarray(labels,dtype=np.int64);predictions=np.asarray(predictions,dtype=np.int64)
    matrix=confusion_matrix(labels,predictions,labels=range(classes))
    totals=matrix.sum(1)
    return {"samples":len(labels),"correct":int((labels==predictions).sum()),"accuracy":float((labels==predictions).mean()),
      "loss":loss/len(labels),"confusion_matrix":matrix.tolist(),
      "per_class_accuracy":[float(matrix[i,i]/totals[i]) for i in range(classes)],
      "prediction_histogram":np.bincount(predictions,minlength=classes).tolist(),
      "true_histogram":np.bincount(labels,minlength=classes).tolist(),
      "predicted_classes_used":int(np.count_nonzero(np.bincount(predictions,minlength=classes)))}


def train_seed(seed, train_native, test_native, train_ids, validation_ids, device):
    checkpoint=CHECKPOINTS/f"nmnist_binary_seed{seed}_best.pt"
    result_path=OUT/f"seed{seed}_result.json"; marker=OUT/f"seed{seed}.complete.json"
    config={"dataset":"N-MNIST","representation":"binary_occupancy","seed":seed,"split_seed":42,
      "temporal_bins":10,"sensor_size":[34,34,2],"layout":"[T,polarity,y,x]","binary_rule":"count_cell != 0",
      "architecture":"NMNISTConvSNN","lif_decay":0.5,"optimizer":"Adam","learning_rate":0.001,
      "weight_decay":0.0001,"batch_size":128,"epochs":15,"minimum_epochs":8,"patience":4,
      "gradient_clip_norm":1.0,"initialization":"independent random initialization; no count checkpoint loaded","device":"cuda"}
    if marker.exists() and result_path.exists() and checkpoint.exists():
        done=json.loads(marker.read_text())
        if done.get("checkpoint_sha256")==sha256(checkpoint):
            result=json.loads(result_path.read_text());print(f"seed {seed}: hash-verified completion reused",flush=True);return result
    set_determinism(seed); model=NMNISTConvSNN(config["lif_decay"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    train_data=BinaryNMNIST(train_native); validation=loader(train_data,validation_ids,config["batch_size"],False,seed)
    best_acc=-1.0;best_loss=float("inf");stale=0;history=[];started=time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        model.train();seen=correct=0;loss_sum=0.0
        for x,y in loader(train_data,train_ids,config["batch_size"],True,seed+epoch):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=F.cross_entropy(logits,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"]);optimizer.step()
            seen+=len(y);correct+=int((logits.argmax(1)==y).sum());loss_sum+=float(loss.detach())*len(y)
        val=evaluate(model,validation,device);improved=val["accuracy"]>best_acc or (val["accuracy"]==best_acc and val["loss"]<best_loss)
        if improved:
            best_acc,best_loss,stale=val["accuracy"],val["loss"],0
            torch.save({"model_state":model.state_dict(),"seed":seed,"best_epoch":epoch,"validation_accuracy":best_acc,"validation_loss":best_loss,"config":config},checkpoint)
        else:stale+=1
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_accuracy":correct/seen,"validation_loss":val["loss"],"validation_accuracy":val["accuracy"]};history.append(row)
        print(f"binary seed={seed} epoch={epoch} train={row['train_accuracy']:.4f} val={val['accuracy']:.4f} best={best_acc:.4f}",flush=True)
        if epoch>=config["minimum_epochs"] and stale>=config["patience"]:break
    payload=torch.load(checkpoint,map_location=device,weights_only=True);model.load_state_dict(payload["model_state"],strict=True);model.eval()
    # Official test is first used only after checkpoint selection is frozen.
    test=evaluate(model,DataLoader(BinaryNMNIST(test_native),batch_size=config["batch_size"],shuffle=False,num_workers=0,pin_memory=True),device)
    collapse_pass=test["predicted_classes_used"]==10 and min(test["per_class_accuracy"])>0.5
    result={"status":"PASS" if collapse_pass else "FAIL_CLASS_COLLAPSE","seed":seed,"config":config,"checkpoint":str(checkpoint.relative_to(ROOT)).replace("\\","/"),
      "checkpoint_sha256":sha256(checkpoint),"best_epoch":payload["best_epoch"],"validation_accuracy":payload["validation_accuracy"],
      "test":test,"class_collapse_check_pass":collapse_pass,"parameters":model.trainable_parameter_count(),"runtime_seconds":time.perf_counter()-started}
    atomic_json(result_path,result);atomic_json(marker,{"status":result["status"],"checkpoint_sha256":result["checkpoint_sha256"],"result_sha256":sha256(result_path)})
    with (LOGS/f"seed{seed}_history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(f"N-MNIST | binary | seed {seed} | full test | {test['samples']} | Acc {100*test['accuracy']:.2f}% | {result['status']}",flush=True)
    return result


def aggregate_results():
    results=[]
    for seed in SEEDS:
        path=OUT/f"seed{seed}_result.json"
        if path.exists():results.append(json.loads(path.read_text()))
    rows=[{"dataset":"N-MNIST","representation":"binary","seed":r["seed"],"checkpoint":r["checkpoint"],
           "test_samples":r["test"]["samples"],"correct":r["test"]["correct"],
           "accuracy_percent":100*r["test"]["accuracy"],"status":r["status"]} for r in results]
    fields=["dataset","representation","seed","checkpoint","test_samples","correct","accuracy_percent","status"]
    with (OUT/"clean_accuracy_by_seed.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={"status":"RUNNING","completed_seeds":len(results),"remaining_seeds":3-len(results)}
    if len(results)==3 and all(r["status"]=="PASS" for r in results):
        values=[100*r["test"]["accuracy"] for r in results];mean=statistics.mean(values);sd=statistics.stdev(values)
        summary={"status":"PASS","seeds":list(SEEDS),"mean_accuracy_percent":mean,"sample_std_accuracy_percent":sd,
                 "ci95_low":mean-4.302652729911275*sd/(3**0.5),"ci95_high":mean+4.302652729911275*sd/(3**0.5)}
    atomic_json(OUT/"clean_accuracy_summary.json",summary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,choices=SEEDS);parser.add_argument("--all",action="store_true");args=parser.parse_args()
    if Path(sys.executable).resolve()!=PYTHON.resolve():raise RuntimeError(f"wrong interpreter: {PYTHON}")
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    from tonic.datasets import NMNIST
    train_native=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True);test_native=NMNIST(save_to=str(ROOT/"data/nmnist"),train=False)
    train_ids,validation_ids=stratified_train_validation_indices(train_native.targets,500,42)
    if (args.seed is None)==(not args.all):raise ValueError("select exactly one of --seed or --all")
    seeds=SEEDS if args.all else (args.seed,)
    for seed in seeds:
        train_seed(seed,train_native,test_native,train_ids,validation_ids,torch.device("cuda"));aggregate_results()


if __name__=="__main__":main()
