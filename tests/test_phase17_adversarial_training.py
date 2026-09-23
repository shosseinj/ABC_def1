from pathlib import Path
import hashlib
import json

import torch
import torch.nn.functional as F

from attacks.classical_timing import classical_timing_attack, differentiable_angle_encode
from attacks.training_timing import training_timing_pgd
from models.qsnn import IrisQSNN


def sample_batch():
    times = torch.tensor([[15.0, 35.0, 55.0, 75.0], [80.0, 60.0, 40.0, 20.0]])
    labels = torch.tensor([0, 1])
    return times, labels


def test_zero_epsilon_returns_clean_timings():
    model = IrisQSNN(4, 1, 3)
    times, labels = sample_batch()
    assert torch.equal(training_timing_pgd(model, times, labels, 0.0, steps=3), times)


def test_training_pgd_respects_budget_and_bounds():
    model = IrisQSNN(4, 1, 3)
    times, labels = sample_batch()
    adversarial = training_timing_pgd(model, times, labels, 2.0, steps=3)
    assert float((adversarial - times).abs().max()) <= 2.0 + 1e-6
    assert torch.all((adversarial >= 0.0) & (adversarial <= 100.0))


def test_training_pgd_increases_local_ce_for_an_example():
    torch.manual_seed(17)
    model = IrisQSNN(4, 1, 3)
    times, labels = sample_batch()
    adversarial = training_timing_pgd(model, times, labels, 0.25, steps=1)
    with torch.no_grad():
        clean_losses = F.cross_entropy(model(differentiable_angle_encode(times)), labels, reduction="none")
        adversarial_losses = F.cross_entropy(
            model(differentiable_angle_encode(adversarial)), labels, reduction="none"
        )
    assert torch.any(adversarial_losses >= clean_losses)


def test_attack_generation_does_not_change_weights_or_parameter_gradients():
    model = IrisQSNN(4, 1, 3)
    times, labels = sample_batch()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert all(parameter.grad is None for parameter in model.parameters())
    training_timing_pgd(model, times, labels, 2.0, steps=1)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_outer_adversarial_training_step_updates_weights():
    model = IrisQSNN(4, 1, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    times, labels = sample_batch()
    adversarial = training_timing_pgd(model, times, labels, 2.0, steps=1)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer.zero_grad()
    loss = 0.5 * F.cross_entropy(model(differentiable_angle_encode(times)), labels)
    loss = loss + 0.5 * F.cross_entropy(model(differentiable_angle_encode(adversarial)), labels)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_training_pgd_does_not_mutate_clean_input():
    model = IrisQSNN(4, 1, 3)
    times, labels = sample_batch()
    before = times.clone()
    training_timing_pgd(model, times, labels, 2.0, steps=3)
    assert torch.equal(times, before)


def test_phase14_classical_pgd_remains_separate_and_frozen_model_only():
    source = (Path(__file__).resolve().parents[1] / "attacks" / "classical_timing.py").read_text()
    assert "training_timing_pgd" not in source
    model = IrisQSNN(4, 1, 3)
    times, _ = sample_batch()
    try:
        classical_timing_attack(model, times[0].numpy(), 0, 2.0, iterations=1)
    except ValueError as error:
        assert "must be frozen" in str(error)
    else:
        raise AssertionError("Phase 14 attack accepted an unfrozen victim")


def test_phase14_attack_source_is_unchanged():
    source_path = Path(__file__).resolve().parents[1] / "attacks" / "classical_timing.py"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        "08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"
    )


def test_phase17_failed_gate_withholds_test_set():
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / "configs" / "iris_phase17_selected.json").read_text())
    gate = json.loads((root / "results" / "iris_phase17_gate.json").read_text())
    assert selected["selection_data"] == "validation_only"
    assert selected["adversarial_training_steps"] == 1
    assert selected["adversarial_training_random_start"] is False
    assert gate["multiseed_replication_gate"] is False
    assert gate["status"] == "not_run_validation_gate_failed"
    assert gate["test_set_accessed"] is False
    assert gate["final_test_runs"] == []
