import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_iris_splits(seed=42, test_size=0.2, val_size=0.2):
    data = load_iris()
    X, y = data.data, data.target
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val,
        random_state=seed, stratify=y_trainval
    )
    scaler = MinMaxScaler(clip=True)
    X_train = np.clip(scaler.fit_transform(X_train), 0.0, 1.0)
    X_val = np.clip(scaler.transform(X_val), 0.0, 1.0)
    X_test = np.clip(scaler.transform(X_test), 0.0, 1.0)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_iris_split_indices(seed=42, test_size=0.2, val_size=0.2):
    data = load_iris()
    indices = np.arange(len(data.target))
    trainval_indices, test_indices, y_trainval, _ = train_test_split(
        indices, data.target, test_size=test_size, random_state=seed,
        stratify=data.target,
    )
    rel_val = val_size / (1.0 - test_size)
    train_indices, val_indices = train_test_split(
        trainval_indices, test_size=rel_val, random_state=seed,
        stratify=y_trainval,
    )
    return train_indices, val_indices, test_indices

def load_iris_split_manifest_labels(seed=42, test_size=0.2, val_size=0.2):
    """Return IDs and labels only; no feature array is returned or indexed here.

    sklearn internally materializes its canonical bundle, but callers receive only
    split membership and targets. This is the manifest-only Phase-20 boundary.
    """
    target = np.asarray(load_iris().target)
    train_ids, validation_ids, hidden_ids = load_iris_split_indices(seed,test_size,val_size)
    return {
        "train_ids": train_ids, "validation_ids": validation_ids, "hidden_ids": hidden_ids,
        "train_labels": target[train_ids], "validation_labels": target[validation_ids],
        "hidden_labels": target[hidden_ids],
    }


def load_iris_train_validation(seed=42, test_size=0.2, val_size=0.2):
    """Return train/validation data without transforming or returning held-out data."""
    data = load_iris()
    indices = np.arange(len(data.target))
    trainval_indices, _, y_trainval, _ = train_test_split(
        indices, data.target, test_size=test_size, random_state=seed,
        stratify=data.target,
    )
    rel_val = val_size / (1.0 - test_size)
    train_indices, val_indices, y_train, y_val = train_test_split(
        trainval_indices, y_trainval, test_size=rel_val, random_state=seed,
        stratify=y_trainval,
    )
    scaler = MinMaxScaler(clip=True)
    X_train = np.clip(scaler.fit_transform(data.data[train_indices]), 0.0, 1.0)
    X_val = np.clip(scaler.transform(data.data[val_indices]), 0.0, 1.0)
    return X_train, X_val, y_train, y_val, scaler, train_indices, val_indices


def load_iris_features_for_allowed_ids(requested_ids, allowed_ids, hidden_ids):
    """Return canonical Iris rows only after an explicit ID-boundary check.

    sklearn necessarily materializes its bundled dataset internally.  The caller,
    however, can receive only IDs in ``allowed_ids`` and never a hidden ID.
    """
    requested = np.asarray(requested_ids, dtype=int)
    allowed = set(map(int, allowed_ids)); hidden = set(map(int, hidden_ids))
    if len(set(map(int, requested)) & hidden):
        raise PermissionError("hidden feature ID requested")
    if not set(map(int, requested)) <= allowed:
        raise PermissionError("feature ID is outside the declared train/validation allowance")
    data = load_iris()
    if np.any(requested < 0) or np.any(requested >= len(data.target)):
        raise IndexError("canonical Iris ID out of range")
    return np.asarray(data.data[requested]).copy(), np.asarray(data.target[requested]).copy()
