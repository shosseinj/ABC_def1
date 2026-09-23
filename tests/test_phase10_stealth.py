from metrics.spike_statistics import classical_mismatch

def test_timing_only_preserves_count_rate():
    c=[10,30,50,70]
    a=[11,29,52,69]
    m=classical_mismatch(c,a,T=100)
    assert m["spike_count_diff"]==0
    assert m["rate_diff"]==0
    assert 0 <= m["isi_tv"] <= 1
    assert 0 <= m["delta_cls"] <= 1


def test_duplicate_ordered_boundary_and_small_perturbations():
    clean = [100, 0, 50, 50]
    reordered = [50, 100, 50, 0]
    assert classical_mismatch(clean, reordered, T=100)["delta_cls"] == 0
    tiny = [0, 50 - 1e-9, 50 + 1e-9, 100]
    result = classical_mismatch(clean, tiny, T=100)
    assert result["spike_count_diff"] == 0
    assert result["rate_diff"] == 0
    assert 0 <= result["isi_tv"] <= 1
