import numpy as np

def random_timing_jitter(spike_times, epsilon, T=100.0, seed=None):
    t = np.asarray(spike_times, dtype=float)
    if T <= 0 or epsilon < 0:
        raise ValueError("T must be positive and epsilon must be nonnegative.")
    if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-epsilon, epsilon, size=t.shape)
    adv = np.clip(t + delta, 0.0, T)
    return adv, adv - t
