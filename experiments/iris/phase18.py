"""Deterministic, train/validation-only diagnostics for Phase 18.

The functions in this module are deliberately data-agnostic.  In particular, none
of them imports a dataset loader or knows that a test split exists.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = (1, 2)


def assert_aligned(sample_ids, labels, **representations):
    ids = np.asarray(sample_ids)
    y = np.asarray(labels)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be a unique one-dimensional array")
    if y.shape != ids.shape:
        raise ValueError("labels are not aligned with sample_ids")
    for name, values in representations.items():
        array = np.asarray(values)
        if array.ndim != 2 or len(array) != len(ids):
            raise ValueError(f"{name} is not row-aligned with sample_ids")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    return True


def geometry(features, labels, classes=CLASSES):
    """Centroids, mean radial spreads, distances and finite-safe separation."""
    x, y = np.asarray(features, float), np.asarray(labels, int)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("features and labels are incompatible")
    centroids, spreads = {}, {}
    for cls in classes:
        members = x[y == cls]
        if not len(members):
            raise ValueError(f"class {cls} is absent")
        centroids[cls] = members.mean(axis=0)
        spreads[cls] = float(np.linalg.norm(members - centroids[cls], axis=1).mean())
    distance = float(np.linalg.norm(centroids[classes[0]] - centroids[classes[1]]))
    denominator = spreads[classes[0]] + spreads[classes[1]]
    return {
        "centroid_1": centroids[classes[0]].tolist(),
        "centroid_2": centroids[classes[1]].tolist(),
        "spread_1": spreads[classes[0]], "spread_2": spreads[classes[1]],
        "centroid_distance_1_2": distance,
        "separation_1_2": float(distance / denominator) if denominator > 0 else None,
        "separation_status": "ok" if denominator > 0 else "zero_spread_denominator",
    }


def nearest_centroid(reference_x, reference_y, query_x, classes=CLASSES):
    reference_x, reference_y, query_x = map(np.asarray, (reference_x, reference_y, query_x))
    centroids = np.stack([reference_x[reference_y == c].mean(0) for c in classes])
    distances = np.linalg.norm(query_x[:, None, :] - centroids[None, :, :], axis=2)
    # np.argmin supplies the specified stable class-order tie break.
    return np.asarray(classes)[np.argmin(distances, axis=1)], distances


def local_neighbors(reference_x, reference_y, reference_ids, query_x, query_ids, k=3):
    """Stable Euclidean neighbors ordered by distance then original sample ID."""
    rx, ry, rids = np.asarray(reference_x), np.asarray(reference_y), np.asarray(reference_ids)
    qx, qids = np.asarray(query_x), np.asarray(query_ids)
    if k < 1:
        raise ValueError("k must be positive")
    rows = []
    for vector, sample_id in zip(qx, qids):
        distance = np.linalg.norm(rx - vector, axis=1)
        eligible = rids != sample_id
        order = np.lexsort((ry[eligible], rids[eligible], distance[eligible]))[:k]
        positions = np.flatnonzero(eligible)[order]
        rows.append([{
            "sample_id": int(rids[p]), "label": int(ry[p]), "distance": float(distance[p])
        } for p in positions])
    return rows


def collision_analysis(times, labels, sample_ids, tolerance=0.5):
    """Cross-class exact and prespecified L-infinity near collisions."""
    x, y, ids = np.asarray(times, float), np.asarray(labels, int), np.asarray(sample_ids)
    pairs = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if y[i] == y[j]:
                continue
            distance = float(np.max(np.abs(x[i] - x[j])))
            exact = bool(np.array_equal(x[i], x[j]))
            if exact or distance <= tolerance:
                pairs.append({"sample_id_a": int(ids[i]), "sample_id_b": int(ids[j]),
                              "label_a": int(y[i]), "label_b": int(y[j]),
                              "linf": distance, "kind": "exact" if exact else "near"})
    return {"tolerance": float(tolerance), "exact_count": sum(p["kind"] == "exact" for p in pairs),
            "near_count": sum(p["kind"] == "near" for p in pairs), "pairs": pairs}


def ordering_preservation(normalized, times):
    """Count rows where TTFS fails to preserve the expected reversed ordering."""
    x, t = np.asarray(normalized), np.asarray(times)
    if x.shape != t.shape:
        raise ValueError("normalized and timing arrays differ in shape")
    violations = 0
    for xr, tr in zip(x, t):
        for i in range(len(xr)):
            for j in range(i + 1, len(xr)):
                if np.sign(xr[i] - xr[j]) != -np.sign(tr[i] - tr[j]):
                    violations += 1
    return violations


def fit_linear_diagnostic(train_x, train_y, validation_x, validation_y):
    """Fit only on class-1/2 training rows and evaluate only validation rows."""
    tx, ty = np.asarray(train_x, float), np.asarray(train_y, int)
    vx, vy = np.asarray(validation_x, float), np.asarray(validation_y, int)
    train_mask, validation_mask = np.isin(ty, CLASSES), np.isin(vy, CLASSES)
    # Training-only standardization makes this diagnostic invariant to the known
    # affine rescaling between normalized and continuous TTFS coordinates.
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0),
    )
    estimator.fit(tx[train_mask], ty[train_mask])
    prediction = estimator.predict(vx[validation_mask])
    return {
        "accuracy": float(accuracy_score(vy[validation_mask], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(vy[validation_mask], prediction)),
        "n_fit": int(train_mask.sum()), "n_evaluated": int(validation_mask.sum()),
        "fit_split": "train", "evaluation_split": "validation",
        "prediction": prediction.tolist(), "validation_positions": np.flatnonzero(validation_mask).tolist(),
    }


def true_class_margin(logits, labels):
    z, y = np.asarray(logits, float), np.asarray(labels, int)
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    return z[np.arange(len(y)), y] - other.max(axis=1)


def jensen_shannon_rows(left, right):
    p, q = np.asarray(left, float), np.asarray(right, float)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m), axis=1) + np.sum(q * np.log(q / m), axis=1))
