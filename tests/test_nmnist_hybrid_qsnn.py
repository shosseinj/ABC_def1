import numpy as np
import torch

from experiments.nmnist.hybrid_data import events_to_polarity_frames
from models.nmnist_hybrid_qsnn import (
    ClassicalLatentClassifier, NMNISTHybridQSNN, NMNISTHybridQSNNV2,
    NMNISTSpatialLIFExtractor, SpatialClassicalControl,
)


def test_timestamp_adapter_preserves_event_axes():
    events = np.array([(1, 2, 0, 0), (3, 4, 9, 1)],
                      dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")])
    frames = events_to_polarity_frames(events, temporal_bins=2)
    assert frames.shape == (2, 2, 34, 34)
    assert frames.sum() == 2


def test_classical_and_quantum_models_have_finite_gradients():
    inputs = torch.randint(0, 4, (2, 3, 2, 34, 34), dtype=torch.uint8)
    for model in (ClassicalLatentClassifier(), NMNISTHybridQSNN()):
        logits = model(inputs)
        assert logits.shape == (2, 10)
        logits.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_v2_quantum_head_has_no_classical_bypass_and_normalized_probabilities():
    model = NMNISTHybridQSNNV2(learned_projection=True, two_axis_encoding=True,
                              learned_measurement=True)
    latent = torch.rand(3, 8) * 2 - 1
    probabilities = model.quantum(latent)
    assert probabilities.shape == (3, 256)
    assert torch.allclose(probabilities.sum(1), torch.ones(3), atol=1e-5)
    assert model.head.in_features == probabilities.shape[1]


def test_spatial_frontend_preserves_configured_latent_and_quantum_only_path():
    inputs = torch.randint(0, 4, (2, 3, 2, 34, 34), dtype=torch.uint8)
    classical = SpatialClassicalControl(channels=(16, 32), spatial_size=4, latent_dim=32)
    assert classical(inputs).shape == (2, 10)
    extractor = NMNISTSpatialLIFExtractor(channels=(16, 32), spatial_size=4, latent_dim=32)
    quantum = NMNISTHybridQSNNV2(
        extractor=extractor, latent_dim=32, n_blocks=2, learned_projection=True,
        two_axis_encoding=True, learned_measurement=True)
    assert quantum(inputs).shape == (2, 10)
    assert quantum.quantum_projection.in_features == 32
    assert quantum.head.in_features == 256
