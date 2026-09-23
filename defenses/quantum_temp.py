import numpy as np
import torch
import torch.nn.functional as F


def perturb_spike_times(spike_times, epsilon_fraction, T=100.0, generator=None):
    """Apply uniform timing noise where epsilon_fraction is a fraction of T."""
    if T <= 0 or epsilon_fraction < 0 or epsilon_fraction > 1:
        raise ValueError("T must be positive and epsilon_fraction must lie in [0,1].")
    if not torch.is_floating_point(spike_times) or not torch.isfinite(spike_times).all():
        raise ValueError("spike_times must be a finite floating-point tensor.")
    if torch.any(spike_times < 0) or torch.any(spike_times > T):
        raise ValueError("spike_times must lie in [0,T].")
    budget = epsilon_fraction * T
    if budget == 0:
        return spike_times.clone()
    noise = torch.empty_like(spike_times).uniform_(-budget, budget, generator=generator)
    return torch.clamp(spike_times + noise, 0.0, T)


def timing_to_angle_torch(spike_times, T=100.0):
    if T <= 0:
        raise ValueError("T must be positive.")
    return (torch.pi / 2.0) * spike_times / T


def measured_product_fidelity(clean_features, perturbed_features, eps=1e-7):
    """Fidelity of per-qubit Z-measurement distributions from expectation values."""
    if clean_features.shape != perturbed_features.shape:
        raise ValueError("Quantum feature tensors must have matching shapes.")
    p = torch.clamp((clean_features + 1.0) / 2.0, eps, 1.0 - eps)
    q = torch.clamp((perturbed_features + 1.0) / 2.0, eps, 1.0 - eps)
    per_qubit = (torch.sqrt(p * q) + torch.sqrt((1.0 - p) * (1.0 - q))) ** 2
    return torch.prod(per_qubit, dim=-1)


def symmetric_kl(clean_logits, perturbed_logits):
    clean_log_prob = F.log_softmax(clean_logits, dim=-1)
    perturbed_log_prob = F.log_softmax(perturbed_logits, dim=-1)
    clean_prob = clean_log_prob.exp()
    perturbed_prob = perturbed_log_prob.exp()
    return 0.5 * (
        F.kl_div(clean_log_prob, perturbed_prob, reduction="batchmean")
        + F.kl_div(perturbed_log_prob, clean_prob, reduction="batchmean")
    )


def jensen_shannon(clean_logits, perturbed_logits):
    clean_log_prob = F.log_softmax(clean_logits, dim=-1)
    perturbed_log_prob = F.log_softmax(perturbed_logits, dim=-1)
    clean_prob = clean_log_prob.exp()
    perturbed_prob = perturbed_log_prob.exp()
    midpoint = 0.5 * (clean_prob + perturbed_prob)
    log_midpoint = torch.log(midpoint)
    return 0.5 * (
        torch.sum(clean_prob * (clean_log_prob - log_midpoint), dim=-1)
        + torch.sum(perturbed_prob * (perturbed_log_prob - log_midpoint), dim=-1)
    ).mean()


def true_class_margin(logits, labels):
    if logits.ndim != 2 or labels.shape != (len(logits),):
        raise ValueError("Logits and labels have incompatible shapes.")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competitors = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), float("-inf")
    )
    return true_logits - competitors.max(dim=1).values


def margin_stability_loss(perturbed_logits, labels, margin_target=0.1):
    if margin_target < 0:
        raise ValueError("margin_target must be nonnegative.")
    return torch.relu(margin_target - true_class_margin(perturbed_logits, labels)).mean()


def adversarial_timing_loss(
    clean_logits, adversarial_logits, labels, lambda_adv=1.0,
    lambda_margin=0.5, margin_target=0.1,
):
    clean = F.cross_entropy(clean_logits, labels)
    adversarial = F.cross_entropy(adversarial_logits, labels)
    margin = margin_stability_loss(adversarial_logits, labels, margin_target)
    weighted_adversarial = lambda_adv * adversarial
    weighted_margin = lambda_margin * margin
    return clean + weighted_adversarial + weighted_margin, {
        "clean": clean,
        "adversarial": adversarial,
        "margin": margin,
        "weighted_adversarial": weighted_adversarial,
        "weighted_margin": weighted_margin,
    }


def quantum_temp_loss(
    model, clean_angles, perturbed_angles, labels, lambda_q, lambda_pred,
    quantum_scale=1.0, lambda_js=0.0, lambda_margin=0.0, margin_target=0.1,
):
    if quantum_scale <= 0:
        raise ValueError("quantum_scale must be positive.")
    clean_features = model.quantum_features(clean_angles)
    perturbed_features = model.quantum_features(perturbed_angles)
    clean_logits = model.head(clean_features)
    perturbed_logits = model.head(perturbed_features)
    classification = F.cross_entropy(clean_logits, labels)
    quantum = (1.0 - measured_product_fidelity(clean_features, perturbed_features)).mean()
    consistency = symmetric_kl(clean_logits, perturbed_logits)
    js = jensen_shannon(clean_logits, perturbed_logits)
    margin = margin_stability_loss(perturbed_logits, labels, margin_target)
    weighted_quantum = lambda_q * quantum / quantum_scale
    weighted_consistency = lambda_pred * consistency
    weighted_js = lambda_js * js
    weighted_margin = lambda_margin * margin
    total = classification + weighted_quantum + weighted_consistency + weighted_js + weighted_margin
    return total, {
        "classification": classification,
        "quantum": quantum,
        "consistency": consistency,
        "weighted_quantum": weighted_quantum,
        "weighted_consistency": weighted_consistency,
        "js": js,
        "margin": margin,
        "weighted_js": weighted_js,
        "weighted_margin": weighted_margin,
        "clean_logits": clean_logits,
    }

def estimate_bloch_from_shots(true_bloch, shots=1000, seed=42):
    rng = np.random.default_rng(seed)
    r = np.clip(np.asarray(true_bloch, dtype=float), -1, 1)
    per_basis = max(1, shots // 3)
    est = []
    for comp in r:
        p_plus = (1.0 + comp) / 2.0
        plus = rng.binomial(per_basis, p_plus)
        minus = per_basis - plus
        est.append((plus - minus) / per_basis)
    return np.asarray(est)

def coherence_score(r_hat, r_ref):
    return float(np.linalg.norm(np.asarray(r_hat)-np.asarray(r_ref)))
