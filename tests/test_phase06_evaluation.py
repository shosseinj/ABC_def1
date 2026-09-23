import json
from pathlib import Path

import numpy as np
import torch

from models.qsnn import IrisQSNN


ROOT = Path(__file__).resolve().parents[1]


def test_clean_training_artifacts_are_valid():
    metrics_path = ROOT / "results" / "iris_clean_metrics.json"
    checkpoint_path = ROOT / "checkpoints" / "iris_qsnn_best.pt"
    assert metrics_path.is_file()
    assert checkpoint_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "best_val_accuracy")
    assert all(np.isfinite(metrics[name]) for name in required)
    assert metrics["trainable_parameters"] == 47
    assert metrics["accuracy"] >= 0.85
    model = IrisQSNN(4, 4, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
