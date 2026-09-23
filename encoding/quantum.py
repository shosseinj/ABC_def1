import numpy as np

def normalize_time(spike_times, T=100.0):
    t = np.asarray(spike_times, dtype=float)
    if T <= 0:
        raise ValueError("T must be positive.")
    if np.any(t < 0) or np.any(t > T):
        raise ValueError("Spike times must lie in [0,T].")
    return t / T

def angle_encode(spike_times, T=100.0):
    tau = normalize_time(spike_times, T)
    return (np.pi / 2.0) * tau

def amplitude_theta(spike_times, T=100.0, eps=1e-7):
    tau = np.clip(normalize_time(spike_times, T), eps, 1-eps)
    return np.arccos(np.sqrt(1.0 - tau))

def phase_encode(spike_times, T=100.0):
    tau = normalize_time(spike_times, T)
    theta = np.full_like(tau, np.pi/4.0)
    phi = 2.0 * np.pi * tau
    return theta, phi
