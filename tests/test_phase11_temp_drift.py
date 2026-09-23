import numpy as np
from attacks.temp_drift import temp_drift_reference

def test_temp_drift_reference_bounds():
    t=np.array([20,40,60,80.])
    adv,info=temp_drift_reference(t,epsilon=5,tau=0.2,T=100,steps=5,candidates=8,seed=1)
    assert adv.shape==t.shape
    assert np.max(np.abs(adv-t)) <= 5+1e-9
    assert info["quantum_drift"] >= 0
    assert info["delta_cls"] <= 0.2
