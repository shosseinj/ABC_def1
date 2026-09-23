import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_attack_sweep_has_every_configuration():
    result = json.loads((ROOT / "results" / "iris_attack_sweep.json").read_text())
    assert result["optimizer"] == "randomized_reference"
    runs = result["runs"]
    assert len(runs) == 12
    assert {run["epsilon_fraction"] for run in runs} == {0.01, 0.02, 0.05, 0.1}
    assert {run["tau"] for run in runs} == {0.01, 0.05, 0.1}
    required = {"dataset", "encoding", "model_depth", "seed", "epsilon", "tau", "clean_accuracy", "attacked_accuracy", "asr", "trace_distance", "one_minus_fidelity", "delta_cls"}
    assert all(required <= run.keys() for run in runs)
