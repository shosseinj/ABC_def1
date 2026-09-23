import os
import pytest

import numpy as np
import torch

from experiments.nmnist.data import events_to_temporal_channels, make_development_indices
from models.nmnist_qsnn import NMNISTCudaQSNN, NMNISTTemporalQSNN
from experiments.nmnist.snn_baseline import (
    events_to_frames, select_device, set_determinism, stratified_train_validation_indices,
)
from models.nmnist_snn import NMNISTConvSNN
from scripts.run_nmnist_snn_multiseed import frozen_config_for_seed, mean_sample_sd


def test_event_reduction_preserves_time_order_and_polarity():
    events = np.array([
        (1, 1, 0, 0), (1, 1, 10, 1), (33, 33, 20, 0), (33, 33, 30, 1),
    ], dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")])
    reduced = events_to_temporal_channels(events, temporal_bins=4)
    assert reduced.shape == (4, 8)
    assert np.count_nonzero(reduced) == 4
    assert reduced[0, 0] > 0 and reduced[1, 1] > 0
    assert reduced[2, 6] > 0 and reduced[3, 7] > 0


def test_development_indices_are_seeded_balanced_and_disjoint():
    targets = np.repeat(np.arange(10), 30)
    train, validation = make_development_indices(targets, 42, 20, 5)
    assert len(train) == 200 and len(validation) == 50
    assert not np.intersect1d(train, validation).size
    assert np.array_equal(np.bincount(targets[train]), np.full(10, 20))
    assert np.array_equal(np.bincount(targets[validation]), np.full(10, 5))


def test_temporal_qsnn_shape_and_capacity():
    model = NMNISTTemporalQSNN()
    assert model(torch.rand(2, 8, 8)).shape == (2, 10)
    assert model.trainable_parameter_count() == 298
    assert model.circuit_depth() == 88


def test_cuda_qsnn_shape_capacity_and_finite_gradient():
    model = NMNISTCudaQSNN()
    inputs = torch.rand(3, 4, 8)
    quantum = model.quantum_features(inputs)
    logits = model(inputs)
    assert quantum.shape == (3, 8) and torch.isfinite(quantum).all()
    assert logits.shape == (3, 10)
    assert model.trainable_parameter_count() == 154
    logits.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_native_event_frames_preserve_time_coordinates_and_polarity():
    events = np.array([
        (1, 2, 0, 0), (1, 2, 10, 1), (33, 32, 20, 0), (33, 32, 30, 1),
    ], dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")])
    frames = events_to_frames(events, temporal_bins=4)
    assert frames.shape == (4, 2, 34, 34)
    assert frames.sum() == len(events)
    assert frames[0, 0, 2, 1] == 1 and frames[1, 1, 2, 1] == 1
    assert frames[2, 0, 32, 33] == 1 and frames[3, 1, 32, 33] == 1


def test_full_training_split_is_seeded_stratified_and_disjoint():
    targets = np.repeat(np.arange(10), 20)
    train, validation = stratified_train_validation_indices(targets, 5, 42)
    assert len(train) == 150 and len(validation) == 50
    assert not np.intersect1d(train, validation).size
    assert np.array_equal(np.bincount(targets[validation]), np.full(10, 5))


def test_convolutional_snn_shape_and_finite_gradient():
    model = NMNISTConvSNN()
    output = model(torch.rand(2, 3, 2, 34, 34))
    assert output.shape == (2, 10)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_snn_determinism_configures_cublas_and_preserves_cuda_selection():
    set_determinism(42)
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    if torch.cuda.is_available():
        assert select_device("cuda").type == "cuda"


def test_multiseed_protocol_changes_only_model_seed():
    base = {"seed": 42, "split_seed": 42, "learning_rate": 0.001, "epochs": 15}
    changed = frozen_config_for_seed(base, 123)
    assert changed == {"seed": 123, "split_seed": 42, "learning_rate": 0.001, "epochs": 15}
    assert base["seed"] == 42


def test_multiseed_summary_uses_sample_standard_deviation():
    summary = mean_sample_sd([0.9, 0.92, 0.94])
    assert summary["mean"] == pytest.approx(0.92)
    assert summary["sample_sd"] == pytest.approx(0.02)
