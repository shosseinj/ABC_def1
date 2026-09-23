import numpy as np

def isi(spike_times):
    t = np.sort(np.asarray(spike_times, dtype=float))
    return np.diff(t)

def isi_histogram(spike_times, bins=8, T=100.0):
    if T <= 0 or bins < 1:
        raise ValueError("T and bins must be positive.")
    vals = isi(spike_times)
    h, _ = np.histogram(vals, bins=bins, range=(0,T), density=False)
    h = h.astype(float)
    if h.sum() > 0:
        h /= h.sum()
    return h

def total_variation(p, q):
    p, q = np.asarray(p,float), np.asarray(q,float)
    if p.shape != q.shape:
        raise ValueError("Distributions must have the same shape.")
    return 0.5 * float(np.abs(p-q).sum())

def classical_mismatch(clean, adv, T=100.0, bins=8, weights=(1/3,1/3,1/3)):
    clean = np.asarray(clean,float)
    adv = np.asarray(adv,float)
    if T <= 0 or bins < 1:
        raise ValueError("T and bins must be positive.")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(adv)):
        raise ValueError("Spike times must be finite.")
    if np.any(clean < 0) or np.any(clean > T) or np.any(adv < 0) or np.any(adv > T):
        raise ValueError("Spike times must lie in [0,T].")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be three nonnegative values summing to one.")
    n_diff = abs(len(clean)-len(adv)) / max(len(clean),1)
    r_clean, r_adv = len(clean)/T, len(adv)/T
    r_diff = abs(r_clean-r_adv) / max(r_clean,1e-12)
    h1 = isi_histogram(clean,bins=bins,T=T)
    h2 = isi_histogram(adv,bins=bins,T=T)
    tv = total_variation(h1,h2)
    w1,w2,w3 = weights
    delta = w1*n_diff + w2*r_diff + w3*tv
    return {
        "spike_count_diff": float(n_diff),
        "rate_diff": float(r_diff),
        "isi_tv": float(tv),
        "delta_cls": float(delta),
    }
