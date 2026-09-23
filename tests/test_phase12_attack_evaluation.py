import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_attack_evaluation_uses_clean_correct_asr_denominator():
    result = json.loads((ROOT / "results" / "iris_attack_comparison.json").read_text())
    assert "clean-correct" in result["asr_definition"]
    for run in result["runs"]:
        assert run["asr_denominator"] > 0
        assert 0 <= run["asr"] <= 1
        assert 0 <= run["attacked_accuracy"] <= 1
        assert 0 <= run["fidelity"] <= 1
        assert run["asr_numerator"] <= run["asr_denominator"]
