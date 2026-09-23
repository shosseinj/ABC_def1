import numpy as np
import torch

from attacks.temp_drift import (
    temp_drift_adaptive, temp_drift_improved, temp_drift_one_stage,
    temp_drift_gradient_adaptive, temp_drift_quantum_refined, temp_drift_two_stage,
)
from experiments.iris.attack_evaluation import load_frozen_model


def test_improved_temp_drift_is_reproducible_and_exact_feasible():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    kwargs = dict(epsilon=5.0, tau=0.1, T=100.0, steps=4, candidates=8, seed=9)
    first, info = temp_drift_improved(model, clean, 0, **kwargs)
    second, _ = temp_drift_improved(model, clean, 0, **kwargs)
    assert np.allclose(first, second)
    assert np.max(np.abs(first - clean)) <= 5.0 + 1e-12
    assert np.all((first >= 0.0) & (first <= 100.0))
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 32
    assert info["sensitive_coordinate"] == 2


def test_adaptive_temp_drift_uses_fixed_budget_without_gradients():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    adv, info = temp_drift_adaptive(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0,
        steps=4, candidates=8, seed=9,
    )
    assert np.max(np.abs(adv - clean)) <= 5.0 + 1e-12
    assert np.all((adv >= 0.0) & (adv <= 100.0))
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 32
    assert info["generations"] == 4
    assert all(parameter.grad is None for parameter in model.parameters())


def test_two_stage_temp_drift_preserves_stage1_success_and_budget():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    adv, info = temp_drift_two_stage(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    assert np.max(np.abs(adv - clean)) <= 5.0 + 1e-12
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 1600
    assert not info["stage1_success"] or info["stage1_success_preserved"]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_one_stage_temp_drift_is_feasible_and_uses_pairwise_search():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    adv, info = temp_drift_one_stage(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    assert np.max(np.abs(adv - clean)) <= 5.0 + 1e-12
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 1600
    assert info["pairwise_coordinates"] == [[2, 0], [2, 1], [2, 3]]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_quantum_refinement_preserves_success_and_uses_frozen_design():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    first, info = temp_drift_quantum_refined(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    second, _ = temp_drift_quantum_refined(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    assert np.allclose(first, second)
    assert np.max(np.abs(first - clean)) <= 5.0 + 1e-12
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 1600
    assert info["pairwise_coordinates"] == [[2, 0], [2, 1], [2, 3]]
    assert not info["stage1_success"] or info["stage1_success_preserved"]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_adaptive_is_feasible_deterministic_and_freezes_weights():
    config = {"n_qubits": 4, "n_layers": 4, "n_classes": 3}
    model = load_frozen_model(config, "checkpoints/iris_clean_380epoch_seed_42.pt")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    clean = np.array([10.0, 30.0, 50.0, 70.0])
    first, info = temp_drift_gradient_adaptive(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    second, _ = temp_drift_gradient_adaptive(
        model, clean, 0, epsilon=5.0, tau=0.1, T=100.0, seed=9,
    )
    assert np.allclose(first, second)
    assert np.max(np.abs(first - clean)) <= 5.0 + 1e-12
    assert np.all((first >= 0.0) & (first <= 100.0))
    assert info["delta_cls"] <= 0.1 + 1e-12
    assert info["evaluated_candidates"] == 1600
    assert info["initial_candidates"] == 600
    assert info["gradient_updates"] == 1000
    assert info["sensitive_coordinate"] == 2
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))
