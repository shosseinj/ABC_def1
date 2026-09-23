"""Restart each SHD SNN from its own frozen best weights for validation-only extension."""
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nmnist.snn_baseline import print_runtime_configuration, select_device, set_determinism, sha256, write_json
from experiments.shd.snn_baseline import evaluate, make_tensor_loader, preprocess_partition
from models.shd_snn import SHDRecurrentSNN

SEEDS = [42, 123, 777, 2026, 6543]
TOTAL_EPOCHS = 150
PATIENCE = 30


def incumbent_from_history(history):
    best = max(history, key=lambda row: (float(row["validation_accuracy"]),
                                         -float(row["validation_loss"]), -int(row["epoch"])))
    last_improvement_index = max(
        index for index, row in enumerate(history)
        if int(row["epoch"]) == int(best["epoch"])
    )
    return best, len(history) - last_improvement_index - 1


def load_history(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_combined_history(path, old_history, extension_history):
    rows = old_history + extension_history
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def resume_seed(base_config, frames, labels, train_ids, validation_ids, seed):
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    prior_result_path = results_dir / f"shd_snn_seed{seed}_validation.json"
    prior_history_path = results_dir / f"shd_snn_seed{seed}_history.csv"
    prior = json.loads(prior_result_path.read_text(encoding="utf-8"))
    prior_history = load_history(prior_history_path)
    incumbent_row, stale = incumbent_from_history(prior_history)
    starting_epoch = int(prior["stopping_epoch"]) + 1
    prior_checkpoint_path = ROOT / prior["checkpoint"]
    if sha256(prior_checkpoint_path) != prior["checkpoint_sha256"]:
        raise RuntimeError(f"Seed {seed} prior checkpoint hash mismatch.")
    checkpoint = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["config"]["seed"]) != seed:
        raise RuntimeError(f"Seed {seed} checkpoint belongs to another seed.")

    config = dict(base_config)
    config.pop("seeds")
    config.update({"seed": seed, "epochs": TOTAL_EPOCHS, "patience": PATIENCE})
    config_path = results_dir / f"shd_snn_seed{seed}_resume150_config.json"
    write_json(config_path, config)
    output_checkpoint = checkpoints_dir / f"shd_snn_seed{seed}_resume150_best.pt"
    torch.save(checkpoint, output_checkpoint)

    set_determinism(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(config["device"])
    model = SHDRecurrentSNN(config["input_channels"], config["hidden_size"],
                            config["n_classes"], config["lif_decay"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    validation_loader = make_tensor_loader(frames, labels, validation_ids,
                                           config["batch_size"], False, seed)
    best_accuracy = float(prior["best_validation"]["accuracy"])
    best_loss = float(prior["best_validation"]["loss"])
    best_epoch = int(prior["best_epoch"])
    previous_best_accuracy = best_accuracy
    extension_history = []
    started = time.perf_counter()
    stopping_reason = "max_epochs"

    print(f"seed={seed} checkpoint_loaded={prior['checkpoint']} starting_epoch={starting_epoch} "
          f"previous_best_validation_accuracy={best_accuracy:.6f} optimizer_state=fresh_adam", flush=True)
    for epoch in range(starting_epoch, TOTAL_EPOCHS + 1):
        epoch_started = time.perf_counter()
        train_loader = make_tensor_loader(frames, labels, train_ids, config["batch_size"], True, seed + epoch)
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for spikes, targets in train_loader:
            spikes, targets = spikes.float().to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(spikes)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            seen += len(targets)
        validation = evaluate(model, validation_loader, device, config["n_classes"])
        improved = (validation["accuracy"] > best_accuracy or
                    (validation["accuracy"] == best_accuracy and validation["loss"] < best_loss))
        if improved:
            best_accuracy, best_loss, best_epoch, stale = validation["accuracy"], validation["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                        "config": config, "optimizer_state": optimizer.state_dict()}, output_checkpoint)
        else:
            stale += 1
        epoch_runtime = time.perf_counter() - epoch_started
        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        row = {"seed": seed, "epoch": epoch, "train_loss": total_loss / seen,
               "train_accuracy": correct / seen, "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
               "best_validation_accuracy": best_accuracy, "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_runtime_seconds": epoch_runtime, "gpu_memory_mib": gpu_memory}
        extension_history.append(row)
        print(f"seed={seed} epoch={epoch}/{TOTAL_EPOCHS} train_loss={row['train_loss']:.4f} "
              f"train_accuracy={row['train_accuracy']:.4f} validation_loss={row['validation_loss']:.4f} "
              f"validation_accuracy={row['validation_accuracy']:.4f} best_validation_accuracy={best_accuracy:.4f} "
              f"learning_rate={row['learning_rate']:.6f} epoch_runtime={epoch_runtime:.2f}s "
              f"GPU_memory={gpu_memory:.1f}MiB", flush=True)
        if stale >= PATIENCE:
            stopping_reason = "early_stopping"
            break

    frozen = torch.load(output_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(frozen["model_state"])
    final_validation = evaluate(model, validation_loader, device, config["n_classes"], True)
    history_path = results_dir / f"shd_snn_seed{seed}_resume150_history.csv"
    write_combined_history(history_path, prior_history, extension_history)
    result = {
        "status": "VALIDATION_EXTENSION_COMPLETE", "dataset": "SHD", "seed": seed,
        "checkpoint_loaded": str(prior_checkpoint_path.relative_to(ROOT)),
        "checkpoint_loaded_sha256": prior["checkpoint_sha256"], "starting_epoch": starting_epoch,
        "previous_stopping_epoch": prior["stopping_epoch"], "previous_best_epoch": prior["best_epoch"],
        "previous_best_validation_accuracy": previous_best_accuracy,
        "new_best_validation_accuracy": final_validation["accuracy"], "new_best_validation_loss": final_validation["loss"],
        "new_best_epoch": best_epoch, "stopping_epoch": epoch, "stopping_reason": stopping_reason,
        "total_epoch_budget": TOTAL_EPOCHS, "early_stopping_patience": PATIENCE,
        "historical_stale_carried": int(stale if not extension_history else incumbent_from_history(prior_history)[1]),
        "optimizer_resume": "fresh Adam; prior checkpoint did not contain optimizer state",
        "checkpoint": str(output_checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(output_checkpoint),
        "extension_runtime_seconds": time.perf_counter() - started, "parameters": model.trainable_parameter_count(),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(results_dir / f"shd_snn_seed{seed}_resume150_validation.json", result)
    return result


def main():
    from tonic.datasets import SHD

    base_config = json.loads((ROOT / "configs" / "shd_snn_multiseed.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "results" / "shd_snn_multiseed_split.json").read_text(encoding="utf-8"))
    if base_config["attacks_enabled"] or base_config["defenses_enabled"] or base_config["qsnn_enabled"]:
        raise RuntimeError("SHD extension is validation-only SNN training.")
    set_determinism(SEEDS[0])
    print_runtime_configuration(select_device(base_config["device"]))
    dataset = SHD(save_to=str(ROOT / base_config["data_root"]), train=True)
    frames, labels, _ = preprocess_partition(dataset, base_config, "official_training_resume")
    train_ids = np.asarray(split["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(split["validation_indices"], dtype=np.int64)
    results = [resume_seed(base_config, frames, labels, train_ids, validation_ids, seed) for seed in SEEDS]
    previous = [result["previous_best_validation_accuracy"] for result in results]
    extended = [result["new_best_validation_accuracy"] for result in results]
    previous_mean, new_mean = statistics.mean(previous), statistics.mean(extended)
    summary = {
        "status": "COMPLETED", "seeds": SEEDS,
        "previous_validation_accuracy": {"mean": previous_mean, "sample_sd": statistics.stdev(previous)},
        "extended_validation_accuracy": {"mean": new_mean, "sample_sd": statistics.stdev(extended)},
        "per_seed": results, "parameters": results[0]["parameters"],
        "verdict": "BENEFICIAL" if new_mean > previous_mean else "NOT BENEFICIAL",
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(ROOT / "results" / "shd_snn_resume150_multiseed_summary.json", summary)
    with (ROOT / "results" / "shd_snn_resume150_multiseed_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "checkpoint_loaded", "starting_epoch", "previous_best_validation_accuracy",
                  "new_best_validation_accuracy", "new_best_epoch", "stopping_epoch", "runtime_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result[field] for field in fields if field != "runtime_seconds"}
            row["runtime_seconds"] = result["extension_runtime_seconds"]
            writer.writerow(row)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"SHD RESUMED LONGER TRAINING: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
