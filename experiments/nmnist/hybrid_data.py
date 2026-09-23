from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.nmnist.snn_baseline import events_to_frames


def events_to_polarity_frames(events, temporal_bins=10, sensor_size=(34, 34)):
    """Timestamp-facing adapter kept separate for later TEMP-DRIFT insertion."""
    return events_to_frames(events, temporal_bins=temporal_bins, sensor_size=sensor_size)


def build_frame_cache(dataset, path, temporal_bins=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8,
        shape=(len(dataset), temporal_bins, 2, 34, 34),
    )
    labels = np.empty(len(dataset), dtype=np.int64)
    for index in range(len(dataset)):
        events, label = dataset[index]
        frames[index] = events_to_polarity_frames(events, temporal_bins)
        labels[index] = int(label)
        if (index + 1) % 5000 == 0:
            frames.flush()
            print(f"[CACHE] {index + 1}/{len(dataset)}", flush=True)
    frames.flush()
    np.save(path.with_name(path.stem + "_labels.npy"), labels)


class CachedFrameDataset(Dataset):
    def __init__(self, frame_path, indices):
        self.frame_path = str(frame_path)
        self.label_path = str(Path(frame_path).with_name(Path(frame_path).stem + "_labels.npy"))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.frames = None
        self.labels = None

    def _open(self):
        if self.frames is None:
            self.frames = np.load(self.frame_path, mmap_mode="r")
            self.labels = np.load(self.label_path, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        self._open()
        index = self.indices[position]
        return torch.from_numpy(self.frames[index].copy()), int(self.labels[index])
