from pathlib import Path
import hashlib
import json

import torch
import torch.nn.functional as F

from defenses.quantum_temp import adversarial_timing_loss
from experiments.iris.data import load_iris_train_validation
from experiments.iris.phase171 import classwise_accuracy, phase171_gate


ROOT = Path(__file__).resolve().parents[1]


def test_lambda_adv_is_applied_numerically_and_clean_ce_is_unchanged():
    clean = torch.tensor([[2.0, 0.0], [0.5, 1.0]])
    adversarial = torch.tensor([[0.1, 1.9], [1.2, 0.3]])
    labels = torch.tensor([0, 1])
    loss_low, low = adversarial_timing_loss(clean, adversarial, labels, 0.25, 0.0)
    loss_high, high = adversarial_timing_loss(clean, adversarial, labels, 0.75, 0.0)
    clean_ce = F.cross_entropy(clean, labels)
    adv_ce = F.cross_entropy(adversarial, labels)
    assert torch.allclose(low["clean"], clean_ce)
    assert torch.allclose(high["clean"], clean_ce)
    assert torch.allclose(loss_high - loss_low, 0.5 * adv_ce)
    disabled, _ = adversarial_timing_loss(clean, adversarial, labels, 0.0, 0.0)
    assert torch.allclose(disabled, clean_ce)


def test_phase171_frozen_attack_and_margin_configuration():
    config = json.loads((ROOT / "configs" / "iris_phase171_candidate.json").read_text())
    for candidate in config["candidates"]:
        assert candidate["adversarial_training_steps"] == 1
        assert candidate["adversarial_training_epsilon"] == 0.02
        assert candidate["adversarial_training_step_size"] == 2.0
        assert candidate["adversarial_training_random_start"] is False
        assert candidate["lambda_margin"] == 0.5
        assert candidate["margin_target"] == 0.1
        assert candidate["lambda_js"] == 0.0
        assert candidate["lambda_q"] == 0.0


def test_classwise_reporting_uses_true_labels():
    samples = [
        {"true_label": 0, "clean_correct": True},
        {"true_label": 1, "clean_correct": False},
        {"true_label": 1, "clean_correct": True},
        {"true_label": 2, "clean_correct": True},
    ]
    assert classwise_accuracy(samples) == {0: 1.0, 1: 0.5, 2: 1.0}


def test_validation_ids_are_stable_and_validation_only_loader_has_no_test_output():
    first = load_iris_train_validation()
    second = load_iris_train_validation()
    assert len(first) == 7
    assert (first[-1] == second[-1]).all()
    assert set(first[-2]).isdisjoint(set(first[-1]))


def test_multiseed_gate_logic():
    passing_seed = {
        "clean_preserved": True,
        "positive_large_epsilon_gain": True,
        "negative_both_small_eps": False,
    }
    neutral_seed = {**passing_seed, "positive_large_epsilon_gain": False}
    candidate = {
        "seeds": [passing_seed, passing_seed, neutral_seed],
        "mean_clean_delta": -0.005,
        "class2_degradation_reduced": True,
    }
    assert phase171_gate([candidate])
    candidate["seeds"][2] = {**neutral_seed, "clean_preserved": False}
    assert not phase171_gate([candidate])


def test_phase171_is_validation_only_and_phase14_is_unchanged():
    gate = json.loads((ROOT / "results" / "iris_phase171_gate.json").read_text())
    assert gate["test_set_accessed"] is False
    assert gate["final_test_runs"] == []
    assert not (ROOT / "results" / "iris_phase171_final_test.json").exists()
    source = ROOT / "attacks" / "classical_timing.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"
    )
