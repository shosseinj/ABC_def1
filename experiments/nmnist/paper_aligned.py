"""Paper-aligned N-MNIST temporal-grid preprocessing.

This reproduces SpikingJelly 0.0.0.0.14
``NMNIST(..., frames_number=10, split_by='number')`` followed by the official
``DVSSignEncoder``: event-count frames are split by event index and thresholded
to binary occupancy without normalization.
"""
from __future__ import annotations

import numpy as np


def events_to_number_split_frames(events, temporal_bins: int = 10,
                                  sensor_size: tuple[int, int] = (34, 34)) -> np.ndarray:
    """Integrate ordered events into equal-event-count frames, as SpikingJelly does."""
    width, height = map(int, sensor_size)
    temporal_bins = int(temporal_bins)
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    if not (len(x) == len(y) == len(p)):
        raise ValueError("event fields must have equal lengths")
    if len(x) and (x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height):
        raise ValueError("event coordinates outside sensor bounds")
    if len(p) and not np.all((p == 0) | (p == 1)):
        raise ValueError("event polarity must be binary")
    frames = np.zeros((temporal_bins, 2, height, width), dtype=np.uint16)
    events_per_frame = len(x) // temporal_bins
    for frame in range(temporal_bins):
        left = frame * events_per_frame
        right = len(x) if frame == temporal_bins - 1 else left + events_per_frame
        np.add.at(frames[frame], (p[left:right], y[left:right], x[left:right]), 1)
    return frames


def events_to_number_split_binary(events, temporal_bins: int = 10,
                                  sensor_size: tuple[int, int] = (34, 34)) -> np.ndarray:
    """Return `[T, polarity, y, x]` uint8 occupancy with no scaling."""
    return (events_to_number_split_frames(events, temporal_bins, sensor_size) >= 1).astype(np.uint8)
