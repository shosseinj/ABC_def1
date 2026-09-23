import numpy as np
import pytest
from attacks.random_jitter import random_timing_jitter

def test_random_jitter_constraints():
    t=np.array([20,40,60,80.],float)
    adv,delta=random_timing_jitter(t,epsilon=5,T=100,seed=1)
    assert adv.shape==t.shape
    assert np.all(adv>=0) and np.all(adv<=100)
    assert np.max(np.abs(adv-t)) <= 5+1e-9


def test_random_jitter_all_budgets_and_boundaries():
    t = np.array([0.0, 25.0, 75.0, 100.0])
    for epsilon in (1.0, 2.0, 5.0, 10.0):
        adv, _ = random_timing_jitter(t, epsilon=epsilon, T=100, seed=2)
        assert len(adv) == len(t)
        assert np.all((adv >= 0) & (adv <= 100))
        assert np.max(np.abs(adv - t)) <= epsilon + 1e-12
    with pytest.raises(ValueError):
        random_timing_jitter(t, epsilon=-1, T=100)
