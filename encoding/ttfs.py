import numpy as np

def ttfs_encode(x_norm, T=100.0):
    x = np.asarray(x_norm, dtype=float)
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("TTFS input must be in [0,1].")
    if T <= 0:
        raise ValueError("T must be positive.")
    return T * (1.0 - x)
