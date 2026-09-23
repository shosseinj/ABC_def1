"""Continue SHD SNN best checkpoints to total epoch 350 with plateau scheduling."""
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
from scripts.resume_shd_snn_longer import incumbent_from_history, load_history, write_combined_history

SEEDS = [42, 123, 777, 2026, 6543]
TOTAL_EPOCHS = 350
EARLY_STOPPING_PATIENCE = 40
INITIAL_LR = 0.001


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)


def continue_seed(base_config, frames, labels, train_ids, validation_ids, seed):
    results_dir, checkpoints_dir = ROOT / "results", ROOT / "checkpoints"
    prior_path = results_dir / f"shd_snn_seed{seed}_resume150_validation.json"
    history_path = results_dir / f"shd_snn_seed{seed}_resume150_history.csv"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    history = load_history(history_path)
    _, stale = incumbent_from_history(history)
    starting_epoch = int(prior["stopping_epoch"]) + 1
    source_checkpoint = ROOT / prior["checkpoint"]
    if sha256(source_checkpoint) != prior["checkpoint_sha256"]:
        raise RuntimeError(f"Seed {seed} source checkpoint hash mismatch.")
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if int(checkpoint["config"]["seed"]) != seed:
        raise RuntimeError(f"Seed {seed} source checkpoint belongs to another seed.")

    config = dict(base_config)
    config.pop("seeds")
    config.update({
        "seed": seed, "epochs": TOTAL_EPOCHS, "patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": INITIAL_LR,
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5,
                      "patience": 10, "min_lr": 1e-5, "monitor": "validation_accuracy"},
    })
    config_path = results_dir / f"shd_snn_seed{seed}_resume350_config.json"
    write_json(config_path, config)
    output_checkpoint = checkpoints_dir / f"shd_snn_seed{seed}_resume350_best.pt"
    torch.save(checkpoint, output_checkpoint)

    set_determinism(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = select_device(config["device"])
    model = SHDRecurrentSNN(config["input_channels"], config["hidden_size"],
                            config["n_classes"], config["lif_decay"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=config["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = INITIAL_LR
        group["initial_lr"] = INITIAL_LR
    scheduler = make_scheduler(optimizer)
    best_accuracy = float(prior["new_best_validation_accuracy"])
    best_loss = float(prior["new_best_validation_loss"])
    best_epoch = int(prior["new_best_epoch"])
    previous_best_accuracy = best_accuracy
    scheduler.step(best_accuracy)
    validation_loader = make_tensor_loader(frames, labels, validation_ids,
                                           config["batch_size"], False, seed)
    extension_history, lr_reductions = [], []
    started = time.perf_counter()
    stopping_reason = "max_epochs"

    print(f"seed={seed} checkpoint_loaded={prior['checkpoint']} starting_epoch={starting_epoch} "
          f"previous_best_validation_accuracy={best_accuracy:.6f} optimizer_state=restored "
          f"scheduler=ReduceLROnPlateau", flush=True)
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
        else:
            stale += 1
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation["accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr:
            lr_reductions.append({"epoch": epoch, "from": previous_lr, "to": current_lr})
        if improved:
            torch.save({"model_state": model.state_dict(), "best_epoch": epoch,
                        "validation_accuracy": best_accuracy, "validation_loss": best_loss,
                        "config": config, "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()}, output_checkpoint)
        epoch_runtime = time.perf_counter() - epoch_started
        row = {"seed": seed, "epoch": epoch, "train_loss": total_loss / seen,
               "train_accuracy": correct / seen, "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
               "best_validation_accuracy": best_accuracy, "learning_rate": current_lr,
               "epoch_runtime_seconds": epoch_runtime,
               "gpu_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2)}
        extension_history.append(row)
        print(f"seed={seed} epoch={epoch}/{TOTAL_EPOCHS} train_loss={row['train_loss']:.4f} "
              f"train_accuracy={row['train_accuracy']:.4f} validation_loss={row['validation_loss']:.4f} "
              f"validation_accuracy={row['validation_accuracy']:.4f} best_validation_accuracy={best_accuracy:.4f} "
              f"learning_rate={current_lr:.6f} epoch_runtime={epoch_runtime:.2f}s", flush=True)
        if stale >= EARLY_STOPPING_PATIENCE:
            stopping_reason = "early_stopping"
            break

    frozen = torch.load(output_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(frozen["model_state"])
    selected_validation = evaluate(model, validation_loader, device, config["n_classes"], True)
    write_combined_history(results_dir / f"shd_snn_seed{seed}_resume350_history.csv", history, extension_history)
    result = {
        "status": "VALIDATION_EXTENSION_COMPLETE", "dataset": "SHD", "seed": seed,
        "checkpoint_loaded": str(source_checkpoint.relative_to(ROOT)),
        "checkpoint_loaded_sha256": prior["checkpoint_sha256"], "starting_epoch": starting_epoch,
        "previous_stopping_epoch": prior["stopping_epoch"], "previous_best_validation_accuracy": previous_best_accuracy,
        "new_best_validation_accuracy": selected_validation["accuracy"],
        "new_best_validation_loss": selected_validation["loss"], "new_best_epoch": best_epoch,
        "stopping_epoch": epoch, "stopping_reason": stopping_reason, "total_epoch_budget": TOTAL_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE, "initial_learning_rate": INITIAL_LR,
        "lr_reductions": lr_reductions, "final_learning_rate": optimizer.param_groups[0]["lr"],
        "optimizer_resume": "Adam state restored from same-seed best checkpoint; LR reset to 0.001",
        "scheduler_resume": "new scheduler introduced at epoch 151 and primed with prior best validation accuracy",
        "checkpoint": str(output_checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(output_checkpoint),
        "extension_runtime_seconds": time.perf_counter() - started, "parameters": model.trainable_parameter_count(),
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(results_dir / f"shd_snn_seed{seed}_resume350_validation.json", result)
    return result


def main():
    from tonic.datasets import SHD

    base_config = json.loads((ROOT / "configs" / "shd_snn_multiseed.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "results" / "shd_snn_multiseed_split.json").read_text(encoding="utf-8"))
    if base_config["attacks_enabled"] or base_config["defenses_enabled"] or base_config["qsnn_enabled"]:
        raise RuntimeError("SHD 350-epoch extension is validation-only SNN training.")
    set_determinism(SEEDS[0])
    print_runtime_configuration(select_device(base_config["device"]))
    dataset = SHD(save_to=str(ROOT / base_config["data_root"]), train=True)
    frames, labels, _ = preprocess_partition(dataset, base_config, "official_training_resume350")
    train_ids = np.asarray(split["train_indices"], dtype=np.int64)
    validation_ids = np.asarray(split["validation_indices"], dtype=np.int64)
    results = [continue_seed(base_config, frames, labels, train_ids, validation_ids, seed) for seed in SEEDS]
    previous = [result["previous_best_validation_accuracy"] for result in results]
    extended = [result["new_best_validation_accuracy"] for result in results]
    summary = {
        "status": "COMPLETED", "seeds": SEEDS,
        "previous_validation_accuracy": {"mean": statistics.mean(previous), "sample_sd": statistics.stdev(previous)},
        "extended_validation_accuracy": {"mean": statistics.mean(extended), "sample_sd": statistics.stdev(extended)},
        "per_seed": results, "parameters": results[0]["parameters"],
        "verdict": "BENEFICIAL" if statistics.mean(extended) > statistics.mean(previous) else "NOT BENEFICIAL",
        "official_test_accessed": False, "attacks_run": False, "defenses_run": False, "qsnn_run": False,
    }
    write_json(ROOT / "results" / "shd_snn_resume350_multiseed_summary.json", summary)
    with (ROOT / "results" / "shd_snn_resume350_multiseed_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "previous_best_validation_accuracy", "new_best_validation_accuracy", "new_best_epoch",
                  "stopping_epoch", "lr_reductions", "final_learning_rate", "runtime_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"seed": result["seed"],
                             "previous_best_validation_accuracy": result["previous_best_validation_accuracy"],
                             "new_best_validation_accuracy": result["new_best_validation_accuracy"],
                             "new_best_epoch": result["new_best_epoch"], "stopping_epoch": result["stopping_epoch"],
                             "lr_reductions": len(result["lr_reductions"]),
                             "final_learning_rate": result["final_learning_rate"],
                             "runtime_seconds": result["extension_runtime_seconds"]})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"SHD 350-EPOCH TRAINING: {summary['verdict']}", flush=True)


if __name__ == "__main__":
    main()
