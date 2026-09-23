import numpy as np
import torch

from experiments.mnist.data import TrainingOnlyPCAReducer, downsample_mnist_8x8, resize_mnist
from models.mnist_qsnn import MNISTReducedQSNN


def test_8x8_downsampling_and_train_only_reducer_are_bounded(tmp_path):
    images = torch.randint(0, 256, (20, 28, 28), generator=torch.Generator().manual_seed(42), dtype=torch.uint8)
    features = downsample_mnist_8x8(images)
    assert features.shape == (20, 64)
    reducer = TrainingOnlyPCAReducer(16).fit(features[:18])
    transformed = reducer.transform(features[18:])
    assert transformed.shape == (2, 16)
    assert np.all((transformed >= 0) & (transformed <= 1))
    path = tmp_path / "reducer.npz"
    reducer.save(path)
    assert path.exists()
    loaded = TrainingOnlyPCAReducer.load(path)
    assert np.allclose(loaded.transform(features[18:]), transformed)


def test_reduced_qsnn_size_and_output():
    model = MNISTReducedQSNN()
    assert model(torch.rand(2, 16)).shape == (2, 10)
    assert model.trainable_parameter_count() == 202
    assert model.circuit_depth() == 22


def test_capacity_variants_only_add_reupload_blocks():
    expected = {2: (202, 22), 3: (218, 33), 4: (234, 44)}
    for blocks, (parameters, depth) in expected.items():
        model = MNISTReducedQSNN(reupload_blocks=blocks)
        assert model(torch.rand(2, 16)).shape == (2, 10)
        assert model.trainable_parameter_count() == parameters
        assert model.circuit_depth() == depth


def test_pca_dimensions_preserve_reduced_qsnn_capacity():
    for input_features in (16, 24, 32):
        model = MNISTReducedQSNN(reupload_blocks=4, input_features=input_features)
        assert model(torch.rand(2, input_features)).shape == (2, 10)
        assert model.trainable_parameter_count() == 234
        assert model.circuit_depth() == 44


def test_pca24_resolution_model_uses_fixed_capacity():
    model = MNISTReducedQSNN(reupload_blocks=4, input_features=24)
    assert model(torch.rand(2, 24)).shape == (2, 10)
    assert model.trainable_parameter_count() == 234
    assert model.circuit_depth() == 44


def test_resolution_resize_is_deterministic_and_bounded():
    images = torch.randint(0, 256, (3, 28, 28), generator=torch.Generator().manual_seed(7), dtype=torch.uint8)
    for resolution in (8, 12, 14):
        first = resize_mnist(images, resolution)
        second = resize_mnist(images, resolution)
        assert first.shape == (3, resolution * resolution)
        assert np.array_equal(first, second)
        assert np.all((first >= 0) & (first <= 1))
