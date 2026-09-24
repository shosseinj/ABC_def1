import numpy as np
import torch

from models.improved_event_snn import ImprovedEventConvSNN
from scripts.train_clean_improved import event_frames


def synthetic_events():
    events = np.zeros(5, dtype=[("x", "i2"), ("y", "i2"),
                                ("p", "?"), ("t", "i8")])
    events["x"] = [3, 3, 4, 5, 6]
    events["y"] = [7, 7, 8, 9, 10]
    events["p"] = [0, 0, 1, 1, 0]
    events["t"] = [0, 0, 5, 9, 10]
    return events


def test_strict_representations_preserve_order_polarity_and_counts():
    events = synthetic_events()
    binary = event_frames(events, "binary")
    integer = event_frames(events, "integer")
    assert binary.shape == integer.shape == (10, 2, 128, 128)
    assert binary.dtype == np.uint8
    assert integer.dtype == np.uint16
    assert set(np.unique(binary)).issubset({0, 1})
    assert integer.sum() == len(events)
    assert integer[0, 0, 7, 3] == 2
    assert binary[0, 0, 7, 3] == 1
    assert integer[:, 1].sum() == 2
    assert integer[0].sum() == 2 and integer[-1].sum() == 1


def test_model_shape_state_reset_and_gradients():
    model = ImprovedEventConvSNN(11, channels=(4, 8, 8, 16, 16, 32))
    frames = torch.zeros(2, 10, 2, 128, 128)
    frames[:, 2, 0, 20, 30] = 1
    first = model(frames)
    second = model(frames)
    assert first.shape == (2, 11)
    assert torch.equal(first, second)
    first.sum().backward()
    assert model.blocks[0].conv.weight.grad is not None
