"""Evaluate frozen 4x4 MNIST checkpoints once on the official held-out partition."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist.data import load_mnist_held_out
from experiments.mnist.training import evaluate_in_batches, to_theta
from models.mnist_qsnn import MNIST4x4QSNN


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = ROOT / "results" / "mnist_4x4_clean_frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_HELD_OUT" or protocol["held_out_accessed"]:
        raise RuntimeError("Held-out evaluation requires an unevaluated frozen protocol.")
    config = protocol["config"]
    held_x, held_y, held_ids = load_mnist_held_out(config)
    rows = []
    for spec in protocol["checkpoints"]:
        checkpoint = ROOT / spec["checkpoint"]
        if sha256(checkpoint) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        model = MNIST4x4QSNN(config["n_qubits"], config["reupload_blocks"], config["n_classes"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        metrics = evaluate_in_batches(model, to_theta(held_x, config["time_window"]), held_y, config["batch_size"])
        rows.append({"seed": spec["seed"], **metrics})
        print(f"[HELD OUT] seed={spec['seed']} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
    accuracy = np.asarray([row["accuracy"] for row in rows])
    macro_f1 = np.asarray([row["macro_f1"] for row in rows])
    summary = {
        "accuracy": {"mean": float(accuracy.mean()), "sample_sd": float(accuracy.std(ddof=1))},
        "macro_f1": {"mean": float(macro_f1.mean()), "sample_sd": float(macro_f1.std(ddof=1))},
    }
    final = {**protocol, "status": "FROZEN_AND_HELD_OUT_EVALUATED_ONCE",
             "held_out_accessed": True, "held_out_indices": held_ids.tolist(),
             "held_out_rows": rows, "held_out_summary": summary}
    (ROOT / "results" / "mnist_4x4_clean_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    with (ROOT / "results" / "mnist_4x4_clean_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = [{**row, "class_accuracy": json.dumps(row["class_accuracy"])} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    protocol["status"] = "FROZEN_AND_HELD_OUT_EVALUATED_ONCE"
    protocol["held_out_accessed"] = True
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"MNIST CLEAN BASELINE: ACCEPTABLE", flush=True)


if __name__ == "__main__":
    main()
