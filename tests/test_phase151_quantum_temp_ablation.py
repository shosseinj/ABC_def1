import json
from pathlib import Path

import numpy as np

from experiments.iris.attack_evaluation import evaluate_attack_data
from experiments.iris.data import load_iris_splits
from experiments.iris.training import train_iris_model


ROOT = Path(__file__).resolve().parents[1]


def minimal_config(**updates):
    config = json.loads((ROOT / "configs" / "iris.json").read_text())
    config.update({"epochs": 1, "quantum_temp_enabled": True, "lambda_q": 2.0, "lambda_pred": 3.0, "consistency_enabled": True})
    config.update(updates)
    return config


def test_arbitrary_lambdas_and_loss_ratios_are_logged(tmp_path):
    run = train_iris_model(minimal_config(), tmp_path / "model.pt", evaluate_test=False)
    summary = run["loss_summary"]
    assert not run["test_evaluated"]
    assert run["predictions"] is None and run["targets"] is None
    assert np.isclose(summary["weighted_quantum_loss"], 2.0 * summary["quantum_loss"])
    assert np.isclose(summary["weighted_consistency_loss"], 3.0 * summary["consistency_loss"])
    assert all(np.isfinite(summary[key]) for key in ("r_q", "r_pred", "r_aux"))


def test_disabled_mode_is_backward_compatible(tmp_path):
    omitted = minimal_config(quantum_temp_enabled=False)
    omitted.pop("lambda_q")
    omitted.pop("lambda_pred")
    omitted.pop("consistency_enabled")
    explicit = {**omitted, "quantum_temp_enabled": False, "lambda_q": 20.0, "lambda_pred": 10.0, "consistency_enabled": True}
    first = train_iris_model(omitted, tmp_path / "first.pt", evaluate_test=False)
    second = train_iris_model(explicit, tmp_path / "second.pt", evaluate_test=False)
    assert first["history"] == second["history"]
    assert all(
        np.array_equal(first["model"].state_dict()[key].numpy(), second["model"].state_dict()[key].numpy())
        for key in first["model"].state_dict()
    )


def test_validation_evaluator_uses_only_explicit_data():
    _, Xval, _, _, yval, _, _ = load_iris_splits()
    result = evaluate_attack_data(
        json.loads((ROOT / "configs" / "iris.json").read_text()),
        ROOT / "checkpoints" / "iris_qsnn_best.pt",
        "random_jitter",
        1.0,
        Xval,
        yval,
        split="validation",
    )
    assert result["split"] == "validation"
    assert result["asr_denominator"] <= len(yval)


def test_phase151_artifacts_record_validation_only_selection():
    final = json.loads((ROOT / "results" / "iris_quantum_temp_phase151_final.json").read_text())
    selected = json.loads((ROOT / "configs" / "iris_quantum_temp_selected.json").read_text())
    assert final["selection_data"] == "validation_only"
    assert final["test_evaluation_started_after_selected_config_was_written"]
    assert selected["selection_data"] == "validation_only"
    assert len(final["test_attack_runs"]) == 48
    assert all(run["split"] == "test" for run in final["test_attack_runs"])
    normalized = json.loads(
        (ROOT / "results" / "iris_quantum_temp_normalized_check.json").read_text()
    )
    assert normalized["test_evaluated"] is False
    assert normalized["result"]["r_q"] > 0.01
