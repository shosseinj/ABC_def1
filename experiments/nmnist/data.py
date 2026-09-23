from pathlib import Path

import numpy as np


def events_to_temporal_channels(events, temporal_bins=8, spatial_grid=(2, 2), sensor_size=(34, 34)):
    """Pool x/y and polarity while preserving event assignment to ordered time bins."""
    temporal_bins = int(temporal_bins)
    grid_y, grid_x = map(int, spatial_grid)
    width, height = map(int, sensor_size)
    output = np.zeros((temporal_bins, grid_y * grid_x * 2), dtype=np.float32)
    if len(events) == 0:
        return output
    x = np.asarray(events["x"], dtype=np.int64)
    y = np.asarray(events["y"], dtype=np.int64)
    t = np.asarray(events["t"], dtype=np.int64)
    p = np.asarray(events["p"], dtype=np.int64)
    if np.any(np.diff(t) < 0):
        raise ValueError("N-MNIST events must be time ordered.")
    if x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height:
        raise ValueError("N-MNIST event coordinates are outside the sensor bounds.")
    if not np.all((p == 0) | (p == 1)):
        raise ValueError("N-MNIST polarity must be binary.")
    duration = max(int(t[-1]) - int(t[0]) + 1, 1)
    time_bin = np.minimum(((t - t[0]) * temporal_bins) // duration, temporal_bins - 1)
    cell_x = np.minimum((x * grid_x) // width, grid_x - 1)
    cell_y = np.minimum((y * grid_y) // height, grid_y - 1)
    channel = ((cell_y * grid_x + cell_x) * 2 + p).astype(np.int64)
    np.add.at(output, (time_bin, channel), 1.0)
    maximum = float(output.max())
    if maximum > 0:
        output = np.log1p(output) / np.log1p(maximum)
    return output.astype(np.float32)


def make_development_indices(targets, split_seed, train_per_class, validation_per_class):
    targets = np.asarray(targets, dtype=int)
    rng = np.random.default_rng(int(split_seed))
    train_indices = []
    validation_indices = []
    for label in range(10):
        candidates = np.flatnonzero(targets == label)
        rng.shuffle(candidates)
        required = int(train_per_class) + int(validation_per_class)
        if len(candidates) < required:
            raise ValueError(f"Not enough official-training class-{label} samples.")
        train_indices.extend(candidates[:int(train_per_class)])
        validation_indices.extend(candidates[int(train_per_class):required])
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("N-MNIST train and validation indices overlap.")
    return train_indices, validation_indices


def load_nmnist_development(config):
    from tonic.datasets import NMNIST

    root = Path(config["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    dataset = NMNIST(save_to=str(root), train=True)
    targets = np.asarray(dataset.targets, dtype=int)
    train_ids, validation_ids = make_development_indices(
        targets, config["split_seed"], config["train_per_class"], config["validation_per_class"]
    )

    def represent(indices):
        features = []
        labels = []
        event_counts = []
        durations = []
        for index in indices:
            events, label = dataset[int(index)]
            features.append(events_to_temporal_channels(
                events, config["temporal_bins"], config["spatial_grid"], config["sensor_size"][:2]
            ))
            labels.append(int(label))
            event_counts.append(int(len(events)))
            durations.append(int(events["t"][-1] - events["t"][0]) if len(events) else 0)
        return (
            np.stack(features), np.asarray(labels, dtype=int),
            np.asarray(event_counts, dtype=int), np.asarray(durations, dtype=np.int64),
        )

    train = represent(train_ids)
    validation = represent(validation_ids)
    return train, validation, train_ids, validation_ids, len(dataset)
