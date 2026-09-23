from pathlib import Path
import hashlib
import json

from experiments.iris.phase173 import RULES, margin_counts, per_class_accuracy, select_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def rows():
    base = {"val_loss": 0.5, "macro_F1": 0.9, "min_class_acc": 0.8, "mean_margin": 0.2, "min_class_mean_margin": 0.1, "PGD2_ASR": 0.2}
    return [
        {**base, "epoch": 1, "val_accuracy": 0.9},
        {**base, "epoch": 2, "val_accuracy": 0.9, "val_loss": 0.4, "mean_margin": 0.3},
    ]


def test_current_rule_reproduces_accuracy_then_loss():
    assert select_checkpoint(rows(), "RULE_A_CURRENT")["epoch"] == 2


def test_documented_rules_and_earliest_tie_break():
    data = rows()
    assert select_checkpoint(data, "RULE_B_ACCURACY_STABLE")["epoch"] == 2
    assert select_checkpoint(data, "RULE_C_CLASS_STABILITY")["epoch"] == 1
    data[0]["PGD2_ASR"] = 0.1
    assert select_checkpoint(data, "RULE_D_ROBUST_VALIDATION")["epoch"] == 1
    identical = [{**data[0], "epoch": 3}, {**data[0], "epoch": 4}]
    for rule in RULES:
        assert select_checkpoint(identical, rule)["epoch"] == 3


def test_per_class_metrics_use_true_classes():
    assert per_class_accuracy([0, 1, 1, 2], [0, 1, 2, 2]) == [1.0, 0.5, 1.0]


def test_low_margin_count_includes_all_negative_margins():
    assert margin_counts([-0.4, -0.01, 0.05, 0.20], threshold=0.10) == (3, 2)


def test_rules_are_generic_and_sample_independent():
    source = (ROOT / "experiments" / "iris" / "phase173.py").read_text()
    assert "119" not in source
    assert "class2" not in source.lower()


def test_phase173_validation_only_and_same_rule_across_seeds():
    config = json.loads((ROOT / "configs" / "iris_phase173_checkpoint_rule.json").read_text())
    gate = json.loads((ROOT / "results" / "iris_phase173_gate.json").read_text())
    assert config["selection_data"] == "validation_only"
    assert config["same_rule_across_seeds"] is True
    assert gate["test_set_accessed"] is False
    assert gate["test_loader_invoked"] is False
    assert gate["final_test_runs"] == []


def test_phase173_runner_is_registered():
    source = (ROOT / "phase_runner.py").read_text()
    assert '17.3: "tests/test_phase173_checkpoint_selection.py"' in source


def test_phase173_attack_diagnostics_use_explicit_validation_data_only():
    source = (ROOT / "scripts" / "run_phase173_checkpoint_selection.py").read_text()
    assert "evaluate_attack_data" in source
    assert 'split="validation"' in source
    assert "load_iris_splits" not in source
    assert "evaluate_attack(" not in source


def test_phase14_attack_source_remains_unchanged():
    source = ROOT / "attacks" / "classical_timing.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "08b9b4669fa19a12e826c228b7f2d4712895aab13aed475df3d027fe3369ec65"
