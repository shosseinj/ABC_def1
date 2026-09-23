import numpy as np
import pytest

from experiments.nmnist.data import events_to_temporal_channels
from scripts.run_nmnist_attack_comparison_seed42 import (
    comparison_verdict, exact_feasibility, paired_overlap, project_timestamps,
    replace_timestamps, representation_numpy, select_stratified_clean_correct,
)


DTYPE = [("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")]


def test_projection_enforces_all_shared_constraints():
    clean = np.array([10.0, 12.0, 12.0, 20.0])
    projected = project_timestamps([4.0, 18.0, 7.0, 30.0], clean, 3.0)
    assert exact_feasibility(clean, projected, 3.0)
    assert np.allclose(projected, [10.0, 15.0, 15.0, 20.0])


def test_timestamp_replacement_preserves_native_event_fields_and_count():
    events = np.array([(1, 2, 10, 0), (3, 4, 20, 1), (5, 6, 30, 0)], dtype=DTYPE)
    attacked = replace_timestamps(events, [11, 19, 29])
    assert len(attacked) == len(events)
    for field in ("x", "y", "p"):
        assert np.array_equal(attacked[field], events[field])
    assert np.array_equal(attacked["t"], [11, 19, 29])


def test_hard_representation_is_exactly_training_representation():
    events = np.array([(1, 2, 0, 0), (1, 2, 10, 1), (33, 32, 20, 0), (33, 32, 30, 1)], dtype=DTYPE)
    assert np.array_equal(representation_numpy(events), events_to_temporal_channels(events, 4, (2, 2), (34, 34)))


def test_subset_selection_uses_frozen_order_and_is_balanced():
    ids = np.arange(60)
    labels = np.tile(np.arange(10), 6)
    predictions = labels.copy()
    predictions[:10] = (predictions[:10] + 1) % 10
    selected = select_stratified_clean_correct(ids, labels, predictions, per_class=2)
    assert selected == list(range(10, 30))


def test_subset_selection_fails_closed_if_a_class_is_short():
    with pytest.raises(RuntimeError, match="Insufficient"):
        select_stratified_clean_correct([1, 2], [0, 1], [0, 1], per_class=1)


def test_paired_overlap_and_verdict_rules():
    assert paired_overlap([1, 1, 0, 0], [1, 0, 1, 0]) == {
        "rescued": 1, "broken": 1, "both_fail": 1, "both_robust": 1, "both_success": 1,
    }
    rows = []
    for epsilon, pgd, temp in zip((0.02, 0.05, 0.10), (0.1, 0.2, 0.3), (0.1, 0.25, 0.4)):
        rows.extend([{"epsilon_fraction": epsilon, "attack": "PGD", "asr": pgd},
                     {"epsilon_fraction": epsilon, "attack": "TEMP-DRIFT", "asr": temp}])
    assert comparison_verdict(rows) == "TEMP-DRIFT: STRONGER"


def test_required_output_names_are_frozen():
    from scripts.run_nmnist_attack_comparison_seed42 import SAMPLE_CSV, SUMMARY_CSV
    assert SUMMARY_CSV.name == "nmnist_attack_comparison_seed42.csv"
    assert SAMPLE_CSV.name == "nmnist_attack_samples_seed42.csv"
