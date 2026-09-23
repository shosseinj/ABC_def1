from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.iris.training import train_iris_model


def main():
    base_config = json.loads((ROOT / "configs" / "iris.json").read_text(encoding="utf-8"))
    results_dir = ROOT / "results"
    checkpoint_dir = ROOT / "checkpoints"
    results_dir.mkdir(exist_ok=True)
    rows = []
    for depth in (2, 4, 6):
        config = {**base_config, "n_layers": depth}
        checkpoint = checkpoint_dir / f"iris_qsnn_depth_{depth}_best.pt"
        run = train_iris_model(config, checkpoint_path=checkpoint)
        metrics = run["metrics"]
        rows.append(metrics)
        print(
            f"depth={depth} params={metrics['trainable_parameters']} "
            f"best_val={metrics['best_val_accuracy']:.3f} "
            f"test_acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
        )

    fields = [
        "n_layers", "quantum_parameters", "trainable_parameters",
        "best_val_accuracy", "best_val_loss", "best_epoch", "accuracy", "macro_precision",
        "macro_recall", "macro_f1", "seed", "epochs", "learning_rate",
        "n_qubits", "elapsed_seconds",
    ]
    with (results_dir / "iris_depth_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)
    (results_dir / "iris_depth_ablation.json").write_text(
        json.dumps({"selection_metric": "best_val_accuracy", "runs": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
