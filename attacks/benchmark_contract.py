"""Frozen event-timestamp budget contract for the TEMP-DRIFT benchmark.

Budgets are measured in native integer timestamp units.  Spatial coordinates,
polarity, and event count are outside this projector and must remain unchanged by
the caller.  The auditor intentionally does not call the projection code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


BENCHMARK_BUDGETS = {
    "N-MNIST": {
        "b_inf": (1, 2, 3),
        "b_1": (500, 750, 1000, 1500),
        "b_0": (200, 300, 400, 600),
    },
    "DVS-Gesture": {
        "b_inf": (1, 2, 3),
        "b_1": (2000, 4000, 8000, 16000),
        "b_0": (1000, 2000, 4000, 8000),
    },
    "CIFAR10-DVS": {
        "b_inf": (1, 2, 3),
        "b_1": (2000, 4000, 8000, 16000),
        "b_0": (1000, 2000, 4000, 8000),
    },
}


@dataclass(frozen=True)
class TimestampBudgets:
    b_inf: int | None = None
    b_1: int | None = None
    b_0: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("b_inf", self.b_inf), ("b_1", self.b_1), ("b_0", self.b_0)):
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if self.b_inf is None and self.b_1 is None and self.b_0 is None:
            raise ValueError("At least one timestamp budget must be specified")


def _timestamps(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.rint(array).astype(np.int64)


def realized_budgets(clean, adversarial) -> dict[str, int]:
    clean_t = _timestamps(clean, "clean")
    adv_t = _timestamps(adversarial, "adversarial")
    if clean_t.shape != adv_t.shape:
        raise ValueError("clean and adversarial timestamps must have equal shape")
    absolute = np.abs(adv_t - clean_t)
    return {
        "b_inf": int(absolute.max(initial=0)),
        "b_1": int(absolute.sum(dtype=np.int64)),
        "b_0": int(np.count_nonzero(absolute)),
    }


def project_timestamp_budgets(
    clean,
    proposed,
    budgets: TimestampBudgets,
    *,
    timestamp_min: int | None = None,
    timestamp_max: int | None = None,
) -> np.ndarray:
    """Project proposed timestamps into the intersection of supplied budgets.

    The deterministic order is B-infinity clipping, B-zero support selection,
    B-one scaling, then timestamp-domain clipping.  Domain clipping can only
    reduce displacement, so it cannot invalidate a budget.
    """
    clean_t = _timestamps(clean, "clean")
    proposed_t = _timestamps(proposed, "proposed")
    if clean_t.shape != proposed_t.shape:
        raise ValueError("clean and proposed timestamps must have equal shape")
    if timestamp_min is not None and timestamp_max is not None and timestamp_min > timestamp_max:
        raise ValueError("timestamp_min cannot exceed timestamp_max")
    if timestamp_min is not None and np.any(clean_t < timestamp_min):
        raise ValueError("clean timestamps fall below timestamp_min")
    if timestamp_max is not None and np.any(clean_t > timestamp_max):
        raise ValueError("clean timestamps exceed timestamp_max")

    delta = proposed_t - clean_t
    if budgets.b_inf is not None:
        delta = np.clip(delta, -budgets.b_inf, budgets.b_inf)

    if budgets.b_0 is not None and np.count_nonzero(delta) > budgets.b_0:
        # Stable sorting makes ties deterministic by original event index.
        keep = np.argsort(-np.abs(delta), kind="stable")[: budgets.b_0]
        mask = np.zeros(delta.size, dtype=bool)
        mask[keep] = True
        delta = np.where(mask, delta, 0)

    magnitude = np.abs(delta)
    total = int(magnitude.sum(dtype=np.int64))
    if budgets.b_1 is not None and total > budgets.b_1:
        scaled = magnitude.astype(np.float64) * (budgets.b_1 / total)
        allocated = np.floor(scaled).astype(np.int64)
        remainder = budgets.b_1 - int(allocated.sum(dtype=np.int64))
        eligible = np.flatnonzero(allocated < magnitude)
        order = eligible[np.argsort(-(scaled[eligible] - allocated[eligible]), kind="stable")]
        allocated[order[:remainder]] += 1
        delta = np.sign(delta).astype(np.int64) * allocated

    projected = clean_t + delta
    if timestamp_min is not None:
        projected = np.maximum(projected, int(timestamp_min))
    if timestamp_max is not None:
        projected = np.minimum(projected, int(timestamp_max))
    return projected.astype(np.int64, copy=False)


def audit_timestamp_budgets(
    clean,
    adversarial,
    budgets: TimestampBudgets,
    *,
    timestamp_min: int | None = None,
    timestamp_max: int | None = None,
) -> dict:
    """Independently recompute feasibility without invoking the projector."""
    clean_t = _timestamps(clean, "clean")
    adv_t = _timestamps(adversarial, "adversarial")
    same_shape = clean_t.shape == adv_t.shape
    if not same_shape:
        return {"passed": False, "checks": {"event_count_preserved": False}, "realized": None}
    displacement = np.abs(adv_t - clean_t)
    observed = {
        "b_inf": int(displacement.max(initial=0)),
        "b_1": int(displacement.sum(dtype=np.int64)),
        "b_0": int(np.count_nonzero(displacement)),
    }
    checks = {
        "event_count_preserved": True,
        "integer_timestamps": bool(np.array_equal(np.asarray(adversarial), adv_t)),
        "timestamp_min": timestamp_min is None or bool(np.all(adv_t >= timestamp_min)),
        "timestamp_max": timestamp_max is None or bool(np.all(adv_t <= timestamp_max)),
        "b_inf": budgets.b_inf is None or observed["b_inf"] <= budgets.b_inf,
        "b_1": budgets.b_1 is None or observed["b_1"] <= budgets.b_1,
        "b_0": budgets.b_0 is None or observed["b_0"] <= budgets.b_0,
    }
    return {"passed": all(checks.values()), "checks": checks, "realized": observed}


REQUIRED_RESULT_FIELDS = frozenset({
    "dataset", "model", "seed", "sample_id", "true_label", "clean_correct",
    "budget_family", "requested_budget", "realized_b_inf", "realized_b_1",
    "realized_b_0", "clean_prediction", "adversarial_prediction",
    "attack_success", "runtime_seconds", "checkpoint_sha256", "split_sha256",
    "audit_passed",
})


def validate_result_record(record: Mapping) -> list[str]:
    """Return schema violations; an empty list means the record is valid."""
    errors = [f"missing field: {name}" for name in sorted(REQUIRED_RESULT_FIELDS - set(record))]
    if errors:
        return errors
    if record["budget_family"] not in {"b_inf", "b_1", "b_0"}:
        errors.append("budget_family must be b_inf, b_1, or b_0")
    for name in ("requested_budget", "realized_b_inf", "realized_b_1", "realized_b_0"):
        if not isinstance(record[name], (int, np.integer)) or record[name] < 0:
            errors.append(f"{name} must be a nonnegative integer")
    for name in ("clean_correct", "attack_success", "audit_passed"):
        if not isinstance(record[name], (bool, np.bool_)):
            errors.append(f"{name} must be boolean")
    if not isinstance(record["runtime_seconds"], (int, float)) or record["runtime_seconds"] < 0:
        errors.append("runtime_seconds must be nonnegative")
    family = record["budget_family"]
    realized_name = {"b_inf": "realized_b_inf", "b_1": "realized_b_1", "b_0": "realized_b_0"}.get(family)
    if realized_name and not errors and record[realized_name] > record["requested_budget"]:
        errors.append(f"{realized_name} exceeds requested_budget")
    return errors
