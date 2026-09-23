import numpy as np
from metrics.quantum_drift import trace_distance_from_bloch, fidelity_from_angles

def test_identical_states():
    a=np.array([0.1,0.2,0.3])
    assert abs(trace_distance_from_bloch(a,a)) < 1e-12
    assert abs(fidelity_from_angles(a,a)-1.0) < 1e-12


def test_perturbed_product_state_metrics_are_bounded():
    a = np.array([0.0, 0.2, 0.4, 0.6])
    b = np.array([0.1, 0.3, 0.5, 0.7])
    distance = trace_distance_from_bloch(a, b)
    fidelity = fidelity_from_angles(a, b)
    assert 0 < distance <= 1
    assert 0 <= fidelity < 1
    assert np.isclose(distance, np.sqrt(1 - fidelity))
