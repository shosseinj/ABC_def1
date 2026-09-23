import json
from pathlib import Path

import numpy as np
import torch

from encoding.quantum import angle_encode
from encoding.ttfs import ttfs_encode
from experiments.mnist.data import downsample_mnist_4x4, make_development_indices
from models.mnist_qsnn import MNIST4x4QSNN


ROOT = Path(__file__).resolve().parents[1]


def test_downsampling_is_deterministic_bounded_and_has_16_features():
    images = torch.arange(3 * 28 * 28, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 28, 28)
    first = downsample_mnist_4x4(images)
    second = downsample_mnist_4x4(images)
    assert first.shape == (3, 16)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_ttfs_and_angle_encoding_preserve_expected_bounds():
    features = np.linspace(0.0, 1.0, 32).reshape(2, 16)
    times = ttfs_encode(features, 100.0)
    angles = angle_encode(times, 100.0)
    assert times.shape == angles.shape == (2, 16)
    assert np.all((times >= 0.0) & (times <= 100.0))
    assert np.all((angles >= 0.0) & (angles <= np.pi / 2.0))


def test_development_indices_are_balanced_disjoint_and_reproducible():
    labels = np.repeat(np.arange(10), 1200)
    first = make_development_indices(labels, 42, 100, 20)
    second = make_development_indices(labels, 42, 100, 20)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    train, validation = first
    assert np.intersect1d(train, validation).size == 0
    assert np.array_equal(np.bincount(labels[train], minlength=10), np.full(10, 100))
    assert np.array_equal(np.bincount(labels[validation], minlength=10), np.full(10, 20))


def test_model_uses_all_inputs_and_has_frozen_size():
    torch.manual_seed(42)
    model = MNIST4x4QSNN()
    theta = torch.rand(2, 16, requires_grad=True)
    logits = model(theta)
    assert logits.shape == (2, 10)
    assert model.quantum_features(theta).shape == (2, 8)
    assert model.trainable_parameter_count() == 122
    assert model.circuit_depth() == 28
    logits.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()


def test_config_is_clean_and_declares_validation_checkpointing():
    config = json.loads((ROOT / "configs" / "mnist_4x4_clean.json").read_text(encoding="utf-8"))
    assert config["input_features"] == 16
    assert config["model_seeds"] == [42, 123, 777, 2026, 6543]
    assert not config["attacks_enabled"] and not config["defenses_enabled"]
    assert "validation" in config["checkpoint_rule"]
