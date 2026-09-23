from pathlib import Path
import csv
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pennylane as qml
import sklearn
import torch

from experiments.iris.training import train_iris_model


def main():
    config_path = ROOT / "configs" / "iris.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    results_dir = ROOT / "results"
    checkpoint_path = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    results_dir.mkdir(exist_ok=True)

    run = train_iris_model(config, checkpoint_path=checkpoint_path)
    metrics = run["metrics"]
    for epoch, loss, val_loss, val_accuracy in run["history"]:
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} loss={loss:.4f} val_acc={val_accuracy:.3f}")

    with (results_dir / "iris_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "val_loss", "val_accuracy"])
        writer.writerows(run["history"])
    (results_dir / "iris_clean_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    manifest = {
        "config": config,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "pennylane": qml.__version__,
        },
        "scaler_data_min": run["scaler"].data_min_.tolist(),
        "scaler_data_max": run["scaler"].data_max_.tolist(),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    (results_dir / "iris_clean_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nTEST METRICS:", metrics)
    threshold = float(config["min_clean_accuracy"])
    if metrics["accuracy"] >= threshold:
        print("PHASE 6 PASSED. Proceed to attack phases.")
        return 0
    print(f"PHASE 6 FAILED. Accuracy {metrics['accuracy']:.3f} < {threshold:.3f}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
