from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def downsample_mnist_4x4(images):
    images = torch.as_tensor(images)
    if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28):
        raise ValueError("MNIST images must have shape [N,28,28].")
    pixels = images.to(torch.float32) / 255.0 if images.dtype == torch.uint8 else images.to(torch.float32)
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("MNIST pixels must be finite and lie in [0,1].")
    pooled = F.avg_pool2d(pixels.unsqueeze(1), kernel_size=7, stride=7)
    return pooled.reshape(len(images), 16).numpy()


def downsample_mnist_8x8(images):
    return resize_mnist(images, 8)


def resize_mnist(images, resolution):
    images = torch.as_tensor(images)
    if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28):
        raise ValueError("MNIST images must have shape [N,28,28].")
    pixels = images.to(torch.float32) / 255.0 if images.dtype == torch.uint8 else images.to(torch.float32)
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("MNIST pixels must be finite and lie in [0,1].")
    resolution = int(resolution)
    if resolution <= 0 or resolution > 28:
        raise ValueError("MNIST resolution must be between 1 and 28.")
    pooled = F.adaptive_avg_pool2d(pixels.unsqueeze(1), (resolution, resolution))
    return pooled.reshape(len(images), resolution * resolution).numpy()


class TrainingOnlyPCAReducer:
    """PCA and output scaling fitted exclusively on development training data."""

    def __init__(self, n_components):
        self.pca = PCA(n_components=int(n_components), svd_solver="full")
        self.minimum = None
        self.maximum = None

    def fit(self, features):
        reduced = self.pca.fit_transform(np.asarray(features, dtype=np.float64))
        self.minimum = reduced.min(axis=0)
        self.maximum = reduced.max(axis=0)
        return self

    def transform(self, features):
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Reducer must be fitted on training data before transform.")
        reduced = self.pca.transform(np.asarray(features, dtype=np.float64))
        scale = np.maximum(self.maximum - self.minimum, np.finfo(np.float64).eps)
        return np.clip((reduced - self.minimum) / scale, 0.0, 1.0).astype(np.float32)

    def save(self, path):
        path.parent.mkdir(exist_ok=True)
        np.savez(
            path, components=self.pca.components_, mean=self.pca.mean_,
            explained_variance=self.pca.explained_variance_,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
            singular_values=self.pca.singular_values_, minimum=self.minimum,
            maximum=self.maximum, n_samples_seen=np.asarray(self.pca.n_samples_),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as payload:
            reducer = cls(payload["components"].shape[0])
            reducer.pca.components_ = payload["components"].copy()
            reducer.pca.mean_ = payload["mean"].copy()
            reducer.pca.explained_variance_ = payload["explained_variance"].copy()
            reducer.pca.explained_variance_ratio_ = payload["explained_variance_ratio"].copy()
            reducer.pca.singular_values_ = payload["singular_values"].copy()
            reducer.pca.n_samples_ = int(payload["n_samples_seen"])
            reducer.pca.n_features_in_ = reducer.pca.components_.shape[1]
            reducer.pca.n_components_ = reducer.pca.components_.shape[0]
            reducer.minimum = payload["minimum"].copy()
            reducer.maximum = payload["maximum"].copy()
        return reducer


def load_mnist_8x8_development(config):
    return load_mnist_resolution_development(config, 8)


def load_mnist_resolution_development(config, resolution):
    from torchvision.datasets import MNIST

    dataset = MNIST(
        root=str(Path(config["data_root"])), train=True,
        download=bool(config.get("download", False)),
    )
    labels = dataset.targets.numpy()
    train_indices, validation_indices = make_development_indices(
        labels, config["split_seed"], config["train_per_class"],
        config["validation_per_class"],
    )
    features = resize_mnist(dataset.data, resolution)
    return (
        features[train_indices], features[validation_indices],
        labels[train_indices], labels[validation_indices],
        train_indices, validation_indices,
    )


def make_development_indices(labels, split_seed, train_per_class, validation_per_class):
    labels = np.asarray(labels, dtype=int)
    all_indices = np.arange(len(labels), dtype=int)
    train_pool, validation_pool = train_test_split(
        all_indices, test_size=10000, random_state=int(split_seed), stratify=labels
    )
    rng = np.random.default_rng(int(split_seed))

    def balanced_subset(pool, per_class):
        selected = []
        for label in range(10):
            candidates = np.asarray(pool)[labels[np.asarray(pool)] == label].copy()
            rng.shuffle(candidates)
            if len(candidates) < per_class:
                raise ValueError(f"Not enough class-{label} samples.")
            selected.extend(candidates[:per_class])
        return np.asarray(selected, dtype=int)

    train_indices = balanced_subset(train_pool, int(train_per_class))
    validation_indices = balanced_subset(validation_pool, int(validation_per_class))
    if np.intersect1d(train_indices, validation_indices).size:
        raise RuntimeError("MNIST train and validation indices overlap.")
    return train_indices, validation_indices


def load_mnist_development(config):
    from torchvision.datasets import MNIST

    dataset = MNIST(
        root=str(Path(config["data_root"])), train=True,
        download=bool(config.get("download", False)),
    )
    labels = dataset.targets.numpy()
    train_indices, validation_indices = make_development_indices(
        labels, config["split_seed"], config["train_per_class"],
        config["validation_per_class"],
    )
    features = downsample_mnist_4x4(dataset.data)
    return (
        features[train_indices], features[validation_indices],
        labels[train_indices], labels[validation_indices],
        train_indices, validation_indices,
    )


def load_mnist_held_out(config):
    from torchvision.datasets import MNIST

    dataset = MNIST(root=str(Path(config["data_root"])), train=False, download=False)
    return downsample_mnist_4x4(dataset.data), dataset.targets.numpy(), np.arange(len(dataset))
