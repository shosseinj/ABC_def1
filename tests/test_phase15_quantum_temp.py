import copy
import json
from pathlib import Path

import torch

from defenses.quantum_temp import (
    measured_product_fidelity,
    perturb_spike_times,
    quantum_temp_loss,
    timing_to_angle_torch,
)
from models.qsnn import IrisQSNN

ROOT = Path(__file__).resolve().parents[1]


def test_zero_and_nonzero_timing_perturbations():
    clean = torch.tensor([[0.0, 25.0, 75.0, 100.0]])
    same = perturb_spike_times(clean, 0.0, T=100)
    assert torch.equal(clean, same)
    angles = timing_to_angle_torch(clean)
    assert torch.allclose(1.0 - measured_product_fidelity(angles, angles), torch.zeros(1))

    perturbed = perturb_spike_times(clean, 0.02, T=100, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(clean, perturbed)
    assert torch.all((perturbed >= 0) & (perturbed <= 100))
    assert torch.max(torch.abs(perturbed - clean)) <= 2.0 + 1e-6


def test_stage_a_quantum_loss_has_model_gradients_and_updates_weights():
    torch.manual_seed(2)
    model = IrisQSNN(4, 2, 3)
    clean = torch.rand(6, 4) * (torch.pi / 2)
    perturbed = torch.clamp(clean + 0.08, 0, torch.pi / 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss, parts = quantum_temp_loss(model, clean, perturbed, labels, lambda_q=0.1, lambda_pred=0.0)
    quantum_grad = torch.autograd.grad(parts["quantum"], model.qlayer.weights, retain_graph=True)[0]
    assert torch.isfinite(quantum_grad).all() and torch.any(quantum_grad != 0)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_stage_b_consistency_gradient_and_disabled_loss_compatibility():
    torch.manual_seed(5)
    model = IrisQSNN(4, 2, 3)
    clean = torch.rand(6, 4) * (torch.pi / 2)
    perturbed = torch.clamp(clean + 0.1, 0, torch.pi / 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    total, parts = quantum_temp_loss(model, clean, perturbed, labels, 0.1, 0.1)
    consistency_grad = torch.autograd.grad(parts["consistency"], model.head.weight, retain_graph=True)[0]
    assert torch.isfinite(total)
    assert torch.isfinite(consistency_grad).all() and torch.any(consistency_grad != 0)
    baseline = torch.nn.functional.cross_entropy(model(clean), labels)
    assert torch.equal(parts["classification"], baseline)


def test_phase15_experiment_artifact_is_complete():
    path = ROOT / "results" / "iris_quantum_temp_phase15.json"
    assert path.is_file()
    result = json.loads(path.read_text(encoding="utf-8"))
    assert set(result["training_metrics"]) == {"baseline", "qt_q", "qt_qp"}
    assert result["split_selection_optimizer_epochs_unchanged"]
    runs = result["attack_runs"]
    assert len(runs) == 72
    assert {run["defense"] for run in runs} == {"baseline", "qt_q", "qt_qp"}
    assert {run["epsilon_fraction"] for run in runs} == {0.01, 0.02, 0.05, 0.1}
    assert {run["attack"] for run in runs} == {
        "random_jitter", "classical_timing", "temp_drift_reference", "temp_drift_gradient"
    }
    gradients = [run for run in runs if run["attack"] == "temp_drift_gradient"]
    assert all(run["feasible_attack_fraction"] == 1.0 for run in gradients)
    assert all(run["mean_delta_cls"] <= run["tau"] + 1e-12 for run in gradients)
