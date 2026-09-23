import numpy as np


def classwise_accuracy(samples, n_classes=3):
    rows = {}
    for class_id in range(n_classes):
        members = [row for row in samples if row["true_label"] == class_id]
        rows[class_id] = sum(row["clean_correct"] for row in members) / len(members) if members else 0.0
    return rows


def phase171_gate(candidate_rows):
    if not candidate_rows:
        return False
    return any(
        all(row["clean_preserved"] for row in candidate["seeds"])
        and sum(row["positive_large_epsilon_gain"] for row in candidate["seeds"]) >= 2
        and all(not row["negative_both_small_eps"] for row in candidate["seeds"])
        and candidate["mean_clean_delta"] >= -0.01 - 1e-12
        and candidate["class2_degradation_reduced"]
        for candidate in candidate_rows
    )


def summarize_margin(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) else 0.0
