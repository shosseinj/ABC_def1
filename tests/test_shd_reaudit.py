import pytest

from scripts.reaudit_shd_snn_test import mean_sample_sd


def test_reaudit_summary_uses_sample_standard_deviation():
    summary = mean_sample_sd([0.6, 0.7, 0.8])
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["sample_sd"] == pytest.approx(0.1)
