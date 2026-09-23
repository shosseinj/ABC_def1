from pathlib import Path
import hashlib
import json

import numpy as np

from defenses.quantum_temp import true_class_margin
from experiments.iris.data import load_iris_train_validation
from experiments.iris.phase172 import align_trajectory_epochs, centroid_statistics, class12_gap
from scripts.run_phase172_diagnosis import config_pair


ROOT = Path(__file__).resolve().parents[1]


def test_sample119_has_stable_validation_identity_and_expected_label():
    first = load_iris_train_validation()
    second = load_iris_train_validation()
    yval, ids = first[3], first[-1]
    assert np.array_equal(ids, second[-1])
    index = int(np.where(ids == 119)[0][0])
    assert int(yval[index]) == 2


def test_class12_gap_and_true_margin_exclude_true_class():
    logits = np.array([[0.0, 3.0, 1.0], [0.0, 2.0, 5.0]])
    assert np.array_equal(class12_gap(logits, np.array([1, 2])), np.array([2.0, 3.0]))
    import torch
    margins = true_class_margin(torch.tensor(logits), torch.tensor([1, 2]))
    assert torch.equal(margins, torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_centroids_use_only_supplied_validation_features():
    features = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [14.0, 0.0]])
    labels = np.array([1, 1, 2, 2])
    result = centroid_statistics(features, labels, np.array([14.0, 0.0]))
    assert result["class12_centroid_distance"] == 11.0
    assert result["class1_spread"] == 1.0
    assert result["class2_spread"] == 2.0


def test_trajectory_epoch_alignment_rejects_missing_epochs():
    assert align_trajectory_epochs([{"epoch": 1}, {"epoch": 2}], range(1, 3))
    try:
        align_trajectory_epochs([{"epoch": 1}, {"epoch": 3}], range(1, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("Misaligned trajectory was accepted")


def test_frozen_hyperparameters_are_unchanged():
    base = json.loads((ROOT / "configs" / "iris.json").read_text())
    for seed in (42, 777, 2026):
        _, defense = config_pair(base, seed)
        assert defense["adversarial_training_steps"] == 1
        assert defense["adversarial_training_epsilon"] == 0.02
        assert defense["adversarial_training_step_size"] == 2.0
        assert defense["lambda_adv"] == 0.5
        assert defense["lambda_margin"] == 0.5
        assert defense["margin_target"] == 0.1
        assert defense["lambda_js"] == defense["lambda_q"] == 0.0


def test_diagnosis_is_validation_only_and_phase14_unchanged():
    artifact = json.loads((ROOT / "results" / "iris_phase172_diagnostics.json").read_text())
    assert artifact["test_set_accessed"] is False
    assert artifact["test_loader_invoked"] is False
    source = ROOT / "attacks" / "classical_timing.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"
    )
