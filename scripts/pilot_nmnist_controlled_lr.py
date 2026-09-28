"""One-epoch validation-only stability check for a common N-MNIST learning rate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.controlled_models import build_model
from experiments.nmnist.snn_baseline import stratified_train_validation_indices
from scripts.train_nmnist_seed42_paper_aligned import CachedBinary, Logger, evaluate, make_loader, set_determinism, train_epoch

PYTHON = Path(r"C:\Users\jafari.h.SPADANACO\Desktop\ai_project\.venv\Scripts\python.exe")


def main(model_name: str, representation: str, learning_rate: float, epochs: int) -> None:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"use required interpreter {PYTHON}")
    source = json.loads((ROOT / f"Reports/results/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}/seed42_clean_result.json").read_text(encoding="utf-8"))
    state = ROOT / f"Reports/checkpoints/nmnist_seed42_paper_aligned{'_integer' if representation == 'integer' else ''}"
    data = CachedBinary(ROOT / source["train_cache"]["path"], state / "train_labels_int8.npy")
    train_ids, val_ids = stratified_train_validation_indices(np.asarray(data.labels), 500, 42)
    log_path = ROOT / f"Reports/logs/nmnist_controlled_four_models/pilot_{representation}_{model_name}_lr{learning_rate:g}.log"
    log = Logger(log_path)
    try:
        set_determinism(42)
        model = build_model(model_name).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        history = []
        for epoch in range(1, epochs + 1):
            train = train_epoch(model, make_loader(data, train_ids, True, 42 + epoch), optimizer, torch.device("cuda"), epoch, epochs, log)
            val, *_ = evaluate(model, make_loader(data, val_ids, False, 42), torch.device("cuda"), f"PILOT VALIDATION epoch={epoch}", log)
            history.append({"epoch": epoch, "train_accuracy": train["accuracy"], "train_loss": train["loss"],
                            "validation_accuracy": val["accuracy"], "validation_loss": val["loss"]})
        record = {"model": model_name, "representation": representation, "learning_rate": learning_rate,
                  "epoch": epochs, "history": history, "train_accuracy": train["accuracy"], "train_loss": train["loss"],
                  "validation_accuracy": val["accuracy"], "validation_loss": val["loss"],
                  "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/")}
        output = ROOT / f"Reports/results/nmnist_controlled_four_models/pilot_{representation}_{model_name}_lr{learning_rate:g}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"PILOT COMPLETE {json.dumps(record,sort_keys=True)}")
    finally:
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("custom", "convnet", "resnet18", "vggsnn"), required=True)
    parser.add_argument("--representation", choices=("binary", "integer"), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.model, args.representation, args.learning_rate, args.epochs)
