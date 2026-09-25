"""Train and verify the seed-42 custom SNN on the paper-aligned N-MNIST Binary grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from experiments.nmnist.paper_aligned import events_to_number_split_binary, events_to_number_split_frames
from experiments.nmnist.snn_baseline import stratified_train_validation_indices
from models.nmnist_snn import NMNISTConvSNN

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
CONFIG_PATH = ROOT / "configs/nmnist_seed42_paper_aligned_binary.json"
STATE = ROOT / "Reports/checkpoints/nmnist_seed42_paper_aligned"
RESULTS = ROOT / "Reports/results/nmnist_seed42_paper_aligned"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} | {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def close(self):
        self.stream.close()


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class CachedBinary(Dataset):
    def __init__(self, frames_path: Path, labels_path: Path):
        self.frames = np.load(frames_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.frames[index], dtype=np.float32, copy=True)), int(self.labels[index]), int(index)


def make_loader(dataset, indices, shuffle: bool, seed: int):
    selected = dataset if indices is None else Subset(dataset, list(map(int, indices)))
    return DataLoader(selected, batch_size=64, shuffle=shuffle, num_workers=0, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


def duplicates(hashes: list[str]) -> dict:
    counts = Counter(hashes)
    groups = [value for value in counts.values() if value > 1]
    return {"unique": len(counts), "duplicate_groups": len(groups),
            "duplicate_extra_samples": sum(value - 1 for value in groups)}


def build_cache(native, partition: str, log: Logger) -> tuple[Path, Path, dict]:
    frames_path = STATE / f"{partition}_t10_number_split_binary_uint8.npy"
    labels_path = STATE / f"{partition}_labels_int8.npy"
    metadata_path = STATE / f"{partition}_cache.json"
    if frames_path.exists() and labels_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("frames_sha256") == sha256(frames_path)
                and metadata.get("labels_sha256") == sha256(labels_path)
                and metadata.get("samples") == len(native)):
            log(f"PHASE 1 CACHE: reusing hash-verified {partition} cache ({len(native)} samples)")
            return frames_path, labels_path, metadata

    STATE.mkdir(parents=True, exist_ok=True)
    frames_tmp = frames_path.with_suffix(f".npy.{os.getpid()}.tmp")
    labels_tmp = labels_path.with_suffix(f".npy.{os.getpid()}.tmp")
    frames = np.lib.format.open_memmap(frames_tmp, mode="w+", dtype=np.uint8, shape=(len(native), 10, 2, 34, 34))
    labels = np.lib.format.open_memmap(labels_tmp, mode="w+", dtype=np.int8, shape=(len(native),))
    hashes = []
    official_mismatches = []
    from spikingjelly.datasets import integrate_events_by_fixed_frames_number
    started = time.perf_counter()
    rng = np.random.default_rng(420042 if partition == "train" else 420043)
    official_indices = set(map(int, rng.choice(len(native), size=min(1000, len(native)), replace=False)))
    for index in range(len(native)):
        events, label = native[index]
        binary = events_to_number_split_binary(events, 10)
        frames[index] = binary
        labels[index] = int(label)
        hashes.append(hashlib.sha256(binary.tobytes()).hexdigest())
        if index in official_indices:
            official_counts = integrate_events_by_fixed_frames_number(
                {key: np.asarray(events[key]) for key in ("t", "x", "y", "p")}, "number", 10, 34, 34)
            if not np.array_equal(binary, (official_counts >= 1).astype(np.uint8)):
                official_mismatches.append(index)
        if (index + 1) % 1000 == 0 or index + 1 == len(native):
            log(f"PHASE 1 CACHE {partition}: {index + 1}/{len(native)} elapsed={time.perf_counter()-started:.1f}s")
    frames.flush(); labels.flush(); del frames, labels
    os.replace(frames_tmp, frames_path); os.replace(labels_tmp, labels_path)
    metadata = {
        "partition": partition, "samples": len(native), "shape": [len(native), 10, 2, 34, 34],
        "dtype": "uint8", "representation": "binary_occupancy", "normalization": "none",
        "temporal_split": "SpikingJelly split_by=number", "official_version": "spikingjelly==0.0.0.0.14",
        "official_comparison_samples": len(official_indices), "official_comparison_mismatches": official_mismatches,
        "input_only_binary_duplicates": duplicates(hashes), "input_only_hashes": hashes,
        "frames_sha256": sha256(frames_path), "labels_sha256": sha256(labels_path),
    }
    atomic_json(metadata_path, metadata)
    log(f"PHASE 1 CACHE {partition}: COMPLETE official_mismatches={len(official_mismatches)}")
    return frames_path, labels_path, metadata


@torch.no_grad()
def evaluate(model, loader, device, phase: str, log: Logger):
    model.eval(); labels=[]; predictions=[]; sample_ids=[]; loss_sum=0.0; correct=0; seen=0
    started=time.perf_counter()
    for batch, (frames, targets, ids) in enumerate(loader, start=1):
        frames=frames.to(device, non_blocking=True); targets=targets.to(device, non_blocking=True)
        logits=model(frames); loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        predicted=logits.argmax(1); correct += int((predicted == targets).sum()); seen += len(targets)
        labels.extend(targets.cpu().tolist()); predictions.extend(predicted.cpu().tolist()); sample_ids.extend(ids.tolist())
        if batch == 1 or batch % 50 == 0 or batch == len(loader):
            log(f"{phase}: batch={batch}/{len(loader)} samples={seen} loss={loss_sum/seen:.6f} "
                f"accuracy={correct/seen:.6f} elapsed={time.perf_counter()-started:.1f}s")
    labels=np.asarray(labels,dtype=np.int64); predictions=np.asarray(predictions,dtype=np.int64)
    matrix=confusion_matrix(labels,predictions,labels=range(10)); totals=matrix.sum(1)
    metrics={"samples":len(labels),"correct":correct,"accuracy":correct/len(labels),
             "macro_f1":float(f1_score(labels,predictions,average="macro",zero_division=0)),
             "loss":loss_sum/len(labels),"confusion_matrix":matrix.tolist(),
             "per_class_accuracy":[float(matrix[i,i]/totals[i]) for i in range(10)],
             "true_histogram":np.bincount(labels,minlength=10).tolist(),
             "prediction_histogram":np.bincount(predictions,minlength=10).tolist()}
    log(f"{phase}: COMPLETE samples={len(labels)} accuracy={metrics['accuracy']:.6f} "
        f"macro_f1={metrics['macro_f1']:.6f} loss={metrics['loss']:.6f}")
    return metrics, labels, predictions, np.asarray(sample_ids,dtype=np.int32)


def train_epoch(model, loader, optimizer, device, epoch: int, total_epochs: int, log: Logger):
    model.train(); seen=correct=0; loss_sum=0.0; started=time.perf_counter()
    for batch,(frames,targets,_) in enumerate(loader,start=1):
        frames=frames.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True);logits=model(frames);loss=F.cross_entropy(logits,targets)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        seen+=len(targets);correct+=int((logits.argmax(1)==targets).sum());loss_sum+=float(loss.detach())*len(targets)
        if batch == 1 or batch % 100 == 0 or batch == len(loader):
            log(f"PHASE 1 TRAIN epoch={epoch}/{total_epochs} batch={batch}/{len(loader)} samples={seen} "
                f"loss={loss_sum/seen:.6f} accuracy={correct/seen:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"elapsed={time.perf_counter()-started:.1f}s")
    return {"loss":loss_sum/seen,"accuracy":correct/seen,"samples":seen}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--log",required=True);args=parser.parse_args()
    log=Logger(ROOT/args.log)
    try:
        if Path(sys.executable).resolve()!=PYTHON.resolve(): raise RuntimeError(f"use required interpreter {PYTHON}")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        config=json.loads(CONFIG_PATH.read_text(encoding="utf-8")); set_determinism(42)
        log("PHASE 1 START: paper-aligned equal-event-count Binary pipeline; seed=42; batch_size=64")
        log("WARNING: old equal-duration checkpoint is representation-mismatched and will not be loaded")
        from tonic.datasets import NMNIST
        train_native=NMNIST(save_to=str(ROOT/"data/nmnist"),train=True)
        test_native=NMNIST(save_to=str(ROOT/"data/nmnist"),train=False)
        train_frames,train_labels,train_cache=build_cache(train_native,"train",log)
        test_frames,test_labels,test_cache=build_cache(test_native,"test",log)
        overlap=len(set(train_cache["input_only_hashes"]).intersection(test_cache["input_only_hashes"]))
        train_ids,val_ids=stratified_train_validation_indices(train_native.targets,500,42)
        labels_array=np.load(train_labels,mmap_mode="r")
        integrity={"train_samples":len(train_native),"test_samples":len(test_native),
          "training_samples":len(train_ids),"validation_samples":len(val_ids),
          "train_validation_intersection":int(np.intersect1d(train_ids,val_ids).size),
          "train_validation_union_complete":len(np.union1d(train_ids,val_ids))==len(train_native),
          "validation_class_histogram":np.bincount(labels_array[val_ids],minlength=10).tolist(),
          "train_test_binary_hash_intersection":overlap,
          "spikingjelly_train_mismatches":len(train_cache["official_comparison_mismatches"]),
          "spikingjelly_test_mismatches":len(test_cache["official_comparison_mismatches"])}
        integrity_pass=(integrity["train_samples"]==60000 and integrity["test_samples"]==10000
          and integrity["train_validation_intersection"]==0 and integrity["train_validation_union_complete"]
          and integrity["validation_class_histogram"]==[500]*10 and overlap==0
          and integrity["spikingjelly_train_mismatches"]==integrity["spikingjelly_test_mismatches"]==0)
        if not integrity_pass: raise RuntimeError(f"integrity gate failed: {integrity}")
        log(f"PHASE 1 INTEGRITY PASS: {json.dumps(integrity,sort_keys=True)}")

        training_set=CachedBinary(train_frames,train_labels); test_set=CachedBinary(test_frames,test_labels)
        device=torch.device("cuda"); model=NMNISTConvSNN(0.5,10).to(device)
        parameters=model.trainable_parameter_count()
        if parameters!=25482: raise RuntimeError(f"parameter count mismatch: {parameters}")
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
        history=[]; best_acc=-1.0;best_loss=float("inf");best_path=None;stale=0;started=time.perf_counter()
        for epoch in range(1,config["epochs"]+1):
            train_metrics=train_epoch(model,make_loader(training_set,train_ids,True,42+epoch),optimizer,device,epoch,config["epochs"],log)
            validation_metrics,*_=evaluate(model,make_loader(training_set,val_ids,False,42),device,
                                           f"PHASE 1 VALIDATION epoch={epoch}/{config['epochs']}",log)
            improved=(validation_metrics["accuracy"]>best_acc or
                      (validation_metrics["accuracy"]==best_acc and validation_metrics["loss"]<best_loss))
            row={"epoch":epoch,"train_loss":train_metrics["loss"],"train_accuracy":train_metrics["accuracy"],
                 "validation_loss":validation_metrics["loss"],"validation_accuracy":validation_metrics["accuracy"],
                 "learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.perf_counter()-started}
            history.append(row)
            if improved:
                best_acc=validation_metrics["accuracy"];best_loss=validation_metrics["loss"];stale=0
                best_path=STATE/f"seed42_best_epoch{epoch:02d}.pt"
                torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"seed":42,
                            "best_epoch":epoch,"validation_accuracy":best_acc,"validation_loss":best_loss,
                            "config":config,"config_sha256":sha256(CONFIG_PATH)},best_path)
                log(f"PHASE 1 CHECKPOINT: new best epoch={epoch} validation_accuracy={best_acc:.6f} path={best_path.relative_to(ROOT)}")
            else: stale+=1
            last_path=STATE/f"seed42_last_epoch{epoch:02d}.pt"
            torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"epoch":epoch,
                        "history":history,"best_path":str(best_path.relative_to(ROOT)) if best_path else None},last_path)
            log(f"PHASE 1 EPOCH COMPLETE: epoch={epoch}/{config['epochs']} train_accuracy={train_metrics['accuracy']:.6f} "
                f"validation_accuracy={validation_metrics['accuracy']:.6f} best={best_acc:.6f} stale={stale}")
            if epoch>=config["minimum_epochs"] and stale>=config["patience"]: break
        if best_path is None: raise RuntimeError("no checkpoint selected")
        checkpoint=torch.load(best_path,map_location=device,weights_only=True);model.load_state_dict(checkpoint["model_state"],strict=True)
        log(f"PHASE 1 CHECKPOINT FROZEN: epoch={checkpoint['best_epoch']} sha256={sha256(best_path)}")
        test_metrics,labels,predictions,sample_ids=evaluate(model,make_loader(test_set,None,False,42),device,
                                                            "PHASE 1 FINAL OFFICIAL TEST",log)
        gate_pass=test_metrics["accuracy"]>=0.95 and integrity_pass
        status="PASS" if gate_pass else "FAILED — MANUAL REVIEW REQUIRED"
        RESULTS.mkdir(parents=True,exist_ok=True)
        history_path=RESULTS/"seed42_training_history.csv"
        with history_path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
        prediction_path=RESULTS/"seed42_test_predictions.npz"
        with prediction_path.open("wb") as stream:
            np.savez_compressed(stream,sample_ids=sample_ids,labels=labels.astype(np.int8),predictions=predictions.astype(np.int8))
        result={"status":status,"clean_gate_pass":gate_pass,"seed":42,"batch_size":64,"parameters":parameters,
          "architecture":"NMNISTConvSNN(decay=0.5, n_classes=10)","config":config,
          "config_path":str(CONFIG_PATH.relative_to(ROOT)).replace("\\","/"),"config_sha256":sha256(CONFIG_PATH),
          "checkpoint_path":str(best_path.relative_to(ROOT)).replace("\\","/"),"checkpoint_sha256":sha256(best_path),
          "best_epoch":checkpoint["best_epoch"],"best_validation_accuracy":checkpoint["validation_accuracy"],
          "best_validation_loss":checkpoint["validation_loss"],"test":test_metrics,"integrity":integrity,
          "train_cache":{"path":str(train_frames.relative_to(ROOT)).replace("\\","/"),"sha256":train_cache["frames_sha256"]},
          "test_cache":{"path":str(test_frames.relative_to(ROOT)).replace("\\","/"),"sha256":test_cache["frames_sha256"]},
          "history_path":str(history_path.relative_to(ROOT)).replace("\\","/"),"history_sha256":sha256(history_path),
          "predictions_path":str(prediction_path.relative_to(ROOT)).replace("\\","/"),"predictions_sha256":sha256(prediction_path),
          "runtime_seconds":time.perf_counter()-started}
        atomic_json(RESULTS/"seed42_clean_result.json",result)
        if gate_pass: log(f"MANDATORY CLEAN GATE PASS: accuracy={test_metrics['accuracy']:.6f}; Phase 2 authorized")
        else:
            log(f"STOP — FAILED — MANUAL REVIEW REQUIRED: accuracy={test_metrics['accuracy']:.6f}; attacks prohibited")
            raise SystemExit(2)
    except Exception as error:
        log(f"ERROR: {type(error).__name__}: {error}");raise
    finally: log.close()


if __name__=="__main__": main()
