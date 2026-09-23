import os

import numpy as np
import torch

from experiments.shd.snn_baseline import events_to_spike_bins, stratified_split
from models.shd_snn import SHDRecurrentSNN


def test_shd_binning_preserves_order_and_channel_identity():
    events = np.array([(0, 2, 1), (14000, 699, 1), (1400000, 4, 1)],
                      dtype=[("t", "i8"), ("x", "i4"), ("p", "i1")])
    binned = events_to_spike_bins(events, time_steps=100, duration_us=1400000)
    assert binned.shape == (100, 700)
    assert binned[0, 2] == 1 and binned[1, 699] == 1 and binned[99, 4] == 1


def test_shd_split_is_deterministic_stratified_and_disjoint():
    labels = np.repeat(np.arange(20), 10)
    train_a, validation_a = stratified_split(labels, 0.2, 42)
    train_b, validation_b = stratified_split(labels, 0.2, 42)
    assert np.array_equal(train_a, train_b) and np.array_equal(validation_a, validation_b)
    assert not np.intersect1d(train_a, validation_a).size
    assert np.array_equal(np.bincount(labels[validation_a]), np.full(20, 2))


def test_shd_snn_shape_capacity_and_gradient():
    model = SHDRecurrentSNN()
    output = model(torch.rand(2, 4, 700))
    assert output.shape == (2, 20)
    assert model.trainable_parameter_count() == 108692
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_shd_determinism_environment_is_configured():
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
