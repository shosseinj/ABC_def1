import numpy as np
import pytest
import torch
import json
from pathlib import Path

from attacks.temp_drift import product_fidelity_torch, temp_drift_gradient
from encoding.quantum import angle_encode
from metrics.quantum_drift import fidelity_from_angles, trace_distance_from_bloch
from metrics.spike_statistics import classical_mismatch

ROOT = Path(__file__).resolve().parents[1]


def test_gradient_temp_drift_is_feasible_reproducible_and_consistent():
    clean = np.array([0.0, 30.0, 70.0, 100.0])
    adv1, info1 = temp_drift_gradient(clean, 5, 0.05, iterations=8, restarts=2, seed=9)
    adv2, info2 = temp_drift_gradient(clean, 5, 0.05, iterations=8, restarts=2, seed=9)
    assert np.array_equal(adv1, adv2)
    assert info1 == info2
    assert adv1.shape == clean.shape
    assert np.all(np.isfinite(adv1)) and np.all((adv1 >= 0) & (adv1 <= 100))
    assert np.max(np.abs(adv1 - clean)) <= 5 + 1e-9
    assert classical_mismatch(clean, adv1)["delta_cls"] <= 0.05 + 1e-12
    assert info1["feasible"]
    clean_angles, adv_angles = angle_encode(clean), angle_encode(adv1)
    assert np.isclose(info1["quantum_drift"], trace_distance_from_bloch(clean_angles, adv_angles))
    torch_fidelity = product_fidelity_torch(torch.tensor(clean_angles), torch.tensor(adv_angles))
    assert np.isclose(float(torch_fidelity), fidelity_from_angles(clean_angles, adv_angles))
    assert info1["quantum_drift"] >= 0


def test_gradient_temp_drift_invalid_and_zero_budget_fallback():
    clean = np.array([20.0, 40.0, 60.0, 80.0])
    adv, info = temp_drift_gradient(clean, 0, 0, iterations=2, restarts=1)
    assert np.array_equal(adv, clean)
    assert info["feasible"] and info["quantum_drift"] == 0
    with pytest.raises(ValueError):
        temp_drift_gradient(clean, -1, 0.1)
    with pytest.raises(ValueError):
        temp_drift_gradient(clean, 1, -0.1)


def test_gradient_comparison_artifact_is_complete_and_feasible():
    result = json.loads((ROOT / "results" / "iris_attack_comparison_gradient.json").read_text())
    assert result["gradient_objective"] == "maximize product-state one-minus-fidelity"
    runs = result["runs"]
    assert len(runs) == 48
    assert {run["attack"] for run in runs} == {
        "random_jitter", "classical_timing", "temp_drift_reference", "temp_drift_gradient"
    }
    gradients = [run for run in runs if run["attack"] == "temp_drift_gradient"]
    assert len(gradients) == 12
    assert all(run["feasible_attack_fraction"] == 1.0 for run in gradients)
    assert all(run["mean_delta_cls"] <= run["tau"] + 1e-12 for run in gradients)
