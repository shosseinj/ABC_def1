import numpy as np


def class12_gap(logits, labels):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2 or logits.shape[1] < 3 or labels.shape != (len(logits),):
        raise ValueError("Invalid logits or labels.")
    if not np.all(np.isin(labels, (1, 2))):
        raise ValueError("class12_gap accepts only classes 1 and 2.")
    return np.where(labels == 1, logits[:, 1] - logits[:, 2], logits[:, 2] - logits[:, 1])


def centroid_statistics(features, labels, sample_feature):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    sample_feature = np.asarray(sample_feature, dtype=float)
    class1 = features[labels == 1]
    class2 = features[labels == 2]
    centroid1, centroid2 = class1.mean(0), class2.mean(0)
    return {
        "class12_centroid_distance": float(np.linalg.norm(centroid1 - centroid2)),
        "class1_spread": float(np.mean(np.linalg.norm(class1 - centroid1, axis=1))),
        "class2_spread": float(np.mean(np.linalg.norm(class2 - centroid2, axis=1))),
        "sample119_to_class1": float(np.linalg.norm(sample_feature - centroid1)),
        "sample119_to_class2": float(np.linalg.norm(sample_feature - centroid2)),
    }


def align_trajectory_epochs(rows, expected_epochs):
    epochs = [row["epoch"] for row in rows]
    if epochs != list(expected_epochs):
        raise ValueError("Trajectory epochs are missing, duplicated, or out of order.")
    return True
