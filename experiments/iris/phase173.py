import numpy as np


RULES = (
    "RULE_A_CURRENT",
    "RULE_B_ACCURACY_STABLE",
    "RULE_C_CLASS_STABILITY",
    "RULE_D_ROBUST_VALIDATION",
)


def select_checkpoint(rows, rule):
    if not rows:
        raise ValueError("At least one epoch row is required.")
    if rule == "RULE_A_CURRENT":
        return min(rows, key=lambda row: (-row["val_accuracy"], row["val_loss"], row["epoch"]))
    if rule == "RULE_B_ACCURACY_STABLE":
        return min(rows, key=lambda row: (
            -row["val_accuracy"], -row["macro_F1"], -row["min_class_acc"],
            -row["mean_margin"], row["epoch"],
        ))
    if rule == "RULE_C_CLASS_STABILITY":
        return min(rows, key=lambda row: (
            -row["val_accuracy"], -row["min_class_acc"], -row["macro_F1"],
            -row["min_class_mean_margin"], row["epoch"],
        ))
    if rule == "RULE_D_ROBUST_VALIDATION":
        best_accuracy = max(row["val_accuracy"] for row in rows)
        eligible = [row for row in rows if row["val_accuracy"] >= best_accuracy - 0.01 - 1e-12]
        return min(eligible, key=lambda row: (
            row["PGD2_ASR"], -row["min_class_acc"], -row["mean_margin"], row["epoch"],
        ))
    raise ValueError(f"Unknown checkpoint rule: {rule}")


def per_class_accuracy(labels, predictions, n_classes=3):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return [float(np.mean(predictions[labels == class_id] == class_id)) for class_id in range(n_classes)]


def margin_counts(margins, threshold=0.10):
    margins = np.asarray(margins, dtype=float)
    return int(np.sum(margins < threshold)), int(np.sum(margins < 0.0))


def pareto_epochs(rows):
    output = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["val_accuracy"] >= candidate["val_accuracy"]
            and other["min_class_acc"] >= candidate["min_class_acc"]
            and other["PGD2_ASR"] <= candidate["PGD2_ASR"]
            and other["mean_margin"] >= candidate["mean_margin"]
            and (
                other["val_accuracy"] > candidate["val_accuracy"]
                or other["min_class_acc"] > candidate["min_class_acc"]
                or other["PGD2_ASR"] < candidate["PGD2_ASR"]
                or other["mean_margin"] > candidate["mean_margin"]
            )
            for other in rows
        )
        if not dominated:
            output.append(candidate)
    return output
