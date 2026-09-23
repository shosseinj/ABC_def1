import numpy as np
import pytest

from attacks.benchmark_contract import (
    BENCHMARK_BUDGETS,
    TimestampBudgets,
    audit_timestamp_budgets,
    project_timestamp_budgets,
    realized_budgets,
    validate_result_record,
)


def test_each_budget_family_is_enforced_independently():
    clean = np.array([10, 20, 30, 40])
    proposed = np.array([20, 11, 37, 35])
    cases = [
        (TimestampBudgets(b_inf=3), "b_inf", 3),
        (TimestampBudgets(b_1=7), "b_1", 7),
        (TimestampBudgets(b_0=2), "b_0", 2),
    ]
    for contract, metric, limit in cases:
        projected = project_timestamp_budgets(clean, proposed, contract)
        assert realized_budgets(clean, projected)[metric] <= limit
        assert audit_timestamp_budgets(clean, projected, contract)["passed"]


def test_intersection_domain_and_determinism():
    clean = np.array([0, 2, 8, 10])
    proposed = np.array([-20, 20, -20, 30])
    contract = TimestampBudgets(b_inf=4, b_1=6, b_0=2)
    first = project_timestamp_budgets(clean, proposed, contract, timestamp_min=0, timestamp_max=10)
    second = project_timestamp_budgets(clean, proposed, contract, timestamp_min=0, timestamp_max=10)
    assert np.array_equal(first, second)
    assert audit_timestamp_budgets(
        clean, first, contract, timestamp_min=0, timestamp_max=10
    )["passed"]


def test_auditor_rejects_budget_violation_without_projecting_it():
    result = audit_timestamp_budgets([1, 2, 3], [1, 8, 3], TimestampBudgets(b_inf=2))
    assert not result["passed"]
    assert not result["checks"]["b_inf"]
    assert result["realized"]["b_inf"] == 6


def test_invalid_contracts_and_shapes_fail_closed():
    with pytest.raises(ValueError):
        TimestampBudgets()
    with pytest.raises(ValueError):
        TimestampBudgets(b_0=-1)
    with pytest.raises(ValueError):
        project_timestamp_budgets([1], [1, 2], TimestampBudgets(b_1=1))


def test_result_schema_accepts_complete_record_and_rejects_missing_field():
    record = {
        "dataset": "N-MNIST", "model": "SNN", "seed": 42, "sample_id": 1,
        "true_label": 3, "clean_correct": True, "budget_family": "b_inf",
        "requested_budget": 2, "realized_b_inf": 2, "realized_b_1": 10,
        "realized_b_0": 5, "clean_prediction": 3, "adversarial_prediction": 8,
        "attack_success": True, "runtime_seconds": 0.5,
        "checkpoint_sha256": "a" * 64, "split_sha256": "b" * 64,
        "audit_passed": True,
    }
    assert validate_result_record(record) == []
    del record["split_sha256"]
    assert validate_result_record(record) == ["missing field: split_sha256"]


def test_frozen_budget_grid_and_schema_budget_audit():
    assert BENCHMARK_BUDGETS["N-MNIST"]["b_1"] == (500, 750, 1000, 1500)
    assert BENCHMARK_BUDGETS["DVS-Gesture"]["b_0"] == (1000, 2000, 4000, 8000)
    record = {
        "dataset": "N-MNIST", "model": "SNN", "seed": 42, "sample_id": 1,
        "true_label": 3, "clean_correct": True, "budget_family": "b_inf",
        "requested_budget": 2, "realized_b_inf": 3, "realized_b_1": 3,
        "realized_b_0": 1, "clean_prediction": 3, "adversarial_prediction": 8,
        "attack_success": True, "runtime_seconds": 0.1,
        "checkpoint_sha256": "a" * 64, "split_sha256": "b" * 64,
        "audit_passed": False,
    }
    assert "realized_b_inf exceeds requested_budget" in validate_result_record(record)
