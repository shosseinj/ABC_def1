import torch
import json
from pathlib import Path

from defenses.quantum_temp import (
    jensen_shannon,
    margin_stability_loss,
    true_class_margin,
)
from models.qsnn import IrisQSNN

ROOT = Path(__file__).resolve().parents[1]


def test_js_is_finite_symmetric_and_zero_for_identical_logits():
    a = torch.tensor([[2.0, -1.0, 0.5]])
    b = torch.tensor([[0.2, 1.1, -0.3]])
    assert torch.isfinite(jensen_shannon(a, b))
    assert torch.allclose(jensen_shannon(a, b), jensen_shannon(b, a))
    assert torch.allclose(jensen_shannon(a, a), torch.tensor(0.0), atol=1e-7)


def test_margin_excludes_true_class_and_loss_obeys_target():
    logits = torch.tensor([[3.0, 2.0, 1.0], [4.0, 5.0, 2.0]])
    labels = torch.tensor([0, 1])
    assert torch.allclose(true_class_margin(logits, labels), torch.tensor([1.0, 1.0]))
    assert margin_stability_loss(logits, labels, 0.1) == 0
    weak = torch.tensor([[0.05, 0.0, -1.0]])
    assert margin_stability_loss(weak, torch.tensor([0]), 0.1) > 0


def test_js_and_margin_reach_classifier_parameters():
    torch.manual_seed(8)
    model = IrisQSNN(4, 2, 3)
    clean = torch.rand(6, 4) * (torch.pi / 2)
    perturbed = torch.clamp(clean + 0.1, 0, torch.pi / 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    clean_logits = model(clean)
    perturbed_logits = model(perturbed)
    for loss in (
        jensen_shannon(clean_logits, perturbed_logits),
        margin_stability_loss(perturbed_logits, labels, 0.1),
    ):
        gradient = torch.autograd.grad(loss, model.head.weight, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and torch.any(gradient != 0)


def test_phase16_validation_gate_prevents_test_access():
    selected = json.loads((ROOT / "configs" / "iris_phase16_selected.json").read_text())
    final = json.loads((ROOT / "results" / "iris_phase16_final_test.json").read_text())
    assert selected["selection_data"] == "validation_only"
    assert selected["lambda_js"] == 0.1
    assert selected["lambda_margin"] == 0.1
    assert selected["margin_target"] == 0.1
    assert final["status"] == "not_run_validation_gate_failed"
    assert final["test_set_accessed"] is False
    assert final["final_test_runs"] == []
    assert len(json.loads((ROOT / "results" / "iris_phase16_js_ablation.json").read_text())["runs"]) == 5
    assert len(json.loads((ROOT / "results" / "iris_phase16_margin_ablation.json").read_text())["runs"]) == 7
