import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_depth_ablation_artifact_is_complete():
    path = ROOT / "results" / "iris_depth_ablation.json"
    assert path.is_file()
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["selection_metric"] == "best_val_accuracy"
    runs = result["runs"]
    assert [run["n_layers"] for run in runs] == [2, 4, 6]
    for run in runs:
        assert run["quantum_parameters"] == run["n_layers"] * 4 * 2
        assert run["trainable_parameters"] == run["quantum_parameters"] + 15
        assert run["seed"] == 42
        assert all(
            np.isfinite(run[name])
            for name in ("best_val_accuracy", "accuracy", "macro_f1")
        )
