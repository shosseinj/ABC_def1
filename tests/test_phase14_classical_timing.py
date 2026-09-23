import numpy as np
import pytest
import torch
import json
from pathlib import Path

from attacks.classical_timing import classical_timing_attack, differentiable_angle_encode
from models.qsnn import IrisQSNN

ROOT = Path(__file__).resolve().parents[1]


def frozen_model():
    torch.manual_seed(4)
    model = IrisQSNN(4, 2, 3).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_classical_timing_constraints_gradient_and_determinism():
    model = frozen_model()
    clean = np.array([0.0, 25.0, 75.0, 100.0])
    original = clean.copy()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    adv1, info1 = classical_timing_attack(model, clean, 1, 5, iterations=5, step_size=1, seed=7)
    adv2, info2 = classical_timing_attack(model, clean, 1, 5, iterations=5, step_size=1, seed=7)
    assert adv1.shape == clean.shape
    assert np.all((adv1 >= 0) & (adv1 <= 100))
    assert np.max(np.abs(adv1 - clean)) <= 5 + 1e-9
    assert np.array_equal(clean, original)
    assert np.array_equal(adv1, adv2)
    assert info1 == info2
    assert info1["max_gradient_norm"] > 0
    assert info1["adversarial_loss"] >= info1["clean_loss"]
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())


def test_differentiable_encoding_and_invalid_parameters():
    times = torch.tensor([20.0, 40.0], requires_grad=True)
    differentiable_angle_encode(times).sum().backward()
    assert times.grad is not None and torch.all(times.grad > 0)
    model = frozen_model()
    with pytest.raises(ValueError):
        classical_timing_attack(model, [20, 40, 60, 80], 1, -1)
    with pytest.raises(ValueError):
        classical_timing_attack(model, [20, 40, 60, 80], 1, 1, T=0)


def test_phase14_experiment_artifact():
    result = json.loads((ROOT / "results" / "iris_attack_comparison_phase14.json").read_text())
    runs = result["runs"]
    assert len(runs) == 12
    assert {run["attack"] for run in runs} == {
        "random_jitter", "classical_timing", "temp_drift_reference"
    }
    assert {run["epsilon_fraction"] for run in runs} == {0.01, 0.02, 0.05, 0.1}
    required = {
        "clean_correct_count", "successful_attacks", "mean_trace_distance",
        "mean_one_minus_fidelity", "mean_delta_cls", "mean_isi_tv",
        "iterations", "step_size", "model_depth", "n_qubits", "checkpoint",
    }
    assert all(required <= run.keys() for run in runs)
