import pytest

from scripts.evaluate_shd_resume350_test_once import mean_sample_sd


def test_resume350_test_summary_uses_sample_standard_deviation():
    result = mean_sample_sd([0.7, 0.8, 0.9])
    assert result["mean"] == pytest.approx(0.8)
    assert result["sample_sd"] == pytest.approx(0.1)
