"""Freeze the same 1,000 clean-correct N-MNIST test IDs for four models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")
OUT = ROOT / "Reports/results/nmnist_controlled_four_models_lr1e4"
NAMES = ("custom", "convnet", "resnet18", "vggsnn")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(representation: str) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    prior = ROOT / f"Reports/results/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}/seed42_clean_result.json"
    results = {}
    for name in NAMES:
        path = OUT / f"{representation}_{name}_seed42.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if not (result.get("clean_gate_pass", True) and result.get("status") == "PASS"
                and result["seed"] == 42 and result["test"]["samples"] == 10000
                and result["test"]["accuracy"] >= 0.95):
            raise RuntimeError(f"{name}: clean result invalid")
        prediction_path = ROOT / result["predictions_path"]
        checkpoint_path = ROOT / result["checkpoint_path"]
        if sha256(prediction_path) != result["predictions_sha256"] or sha256(checkpoint_path) != result["checkpoint_sha256"]:
            raise RuntimeError(f"{name}: frozen artifact hash mismatch")
        with np.load(prediction_path, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(np.int64)
            labels = data["labels"].astype(np.int64)
            predictions = data["predictions"].astype(np.int64)
        if not np.array_equal(ids, np.arange(10000)):
            raise RuntimeError(f"{name}: predictions are not in official test order")
        results[name] = {"labels": labels, "predictions": predictions, "result_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "result_sha256": sha256(path), "checkpoint_sha256": result["checkpoint_sha256"],
                         "predictions_sha256": result["predictions_sha256"], "clean_accuracy": result["test"]["accuracy"]}
    labels = results["custom"]["labels"]
    if any(not np.array_equal(labels, results[name]["labels"]) for name in NAMES):
        raise RuntimeError("model test labels differ")
    eligible = np.ones(10000, dtype=bool)
    for name in NAMES:
        eligible &= results[name]["predictions"] == labels
    selected = np.flatnonzero(eligible)[:1000]
    if len(selected) != 1000:
        raise RuntimeError(f"only {len(selected)} jointly clean-correct test samples")
    manifest = {"status": "FROZEN", "dataset": "N-MNIST", "representation": representation, "seed": 42,
                "selection": "first 1000 jointly clean-correct official-test samples, in test order",
                "eligible_intersection_size": int(eligible.sum()), "sample_ids": selected.tolist(),
                "labels": labels[selected].tolist(), "models": {name: {key: value for key, value in row.items() if key not in ("labels", "predictions")} for name, row in results.items()}}
    path = OUT / f"{representation}_joint_clean_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("frozen joint manifest differs from newly computed manifest")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    print(f"{representation}: joint manifest frozen; eligible={int(eligible.sum())}; selected=1000; sha256={sha256(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True, choices=("binary", "integer"))
    args = parser.parse_args()
    main(args.representation)
