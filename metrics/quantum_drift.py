import numpy as np

def bloch_from_angle(theta):
    th = np.asarray(theta, dtype=float)
    return np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=-1)

def mean_bloch(theta):
    return bloch_from_angle(theta).mean(axis=0)

def trace_distance_from_bloch(theta_a, theta_b):
    fidelity = fidelity_from_angles(theta_a, theta_b)
    return float(np.sqrt(max(0.0, 1.0 - fidelity)))

def fidelity_qubit_from_bloch(r, s):
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    rr = max(0.0, 1.0 - float(np.dot(r, r)))
    ss = max(0.0, 1.0 - float(np.dot(s, s)))
    val = 0.5 * (1.0 + float(np.dot(r, s)) + np.sqrt(rr * ss))
    return float(np.clip(val, 0.0, 1.0))

def fidelity_from_angles(theta_a, theta_b):
    a = np.asarray(theta_a, dtype=float)
    b = np.asarray(theta_b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Angle arrays must have the same non-empty shape.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Angles must be finite.")
    # Product-state fidelity for the actual RY(theta) input circuit.
    return float(np.clip(np.prod(np.cos((a - b) / 2.0) ** 2), 0.0, 1.0))
