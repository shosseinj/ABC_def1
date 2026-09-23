import numpy as np
import torch
import torch.nn.functional as F


def differentiable_angle_encode(spike_times, T=100.0):
    if T <= 0:
        raise ValueError("T must be positive.")
    return (torch.pi / 2.0) * spike_times / T


def classical_timing_attack(
    model,
    spike_times,
    true_label,
    epsilon,
    T=100.0,
    iterations=20,
    step_size=None,
    random_start=False,
    seed=42,
):
    """Untargeted timing-space PGD maximizing true-label cross-entropy."""
    clean = np.asarray(spike_times, dtype=np.float32)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("spike_times must be a non-empty one-dimensional array.")
    if T <= 0 or epsilon < 0 or iterations < 1:
        raise ValueError("T/iterations must be positive and epsilon nonnegative.")
    if not np.all(np.isfinite(clean)) or np.any(clean < 0) or np.any(clean > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    if not isinstance(true_label, (int, np.integer)) or true_label < 0:
        raise ValueError("true_label must be a nonnegative integer.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")
    alpha = float(step_size if step_size is not None else epsilon / max(iterations // 2, 1))
    if not np.isfinite(alpha) or alpha <= 0:
        if epsilon == 0 and step_size is None:
            alpha = 1.0
        else:
            raise ValueError("step_size must be positive and finite.")

    clean_t = torch.tensor(clean, dtype=torch.float32)
    label = torch.tensor([int(true_label)], dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    if random_start and epsilon > 0:
        delta = torch.empty_like(clean_t).uniform_(-epsilon, epsilon, generator=generator)
    else:
        delta = torch.zeros_like(clean_t)
    delta = torch.clamp(clean_t + delta, 0.0, T) - clean_t

    best_times = clean_t.clone()
    with torch.no_grad():
        clean_loss = float(F.cross_entropy(model(differentiable_angle_encode(clean_t, T).unsqueeze(0)), label))
    best_loss = clean_loss
    gradient_norms = []
    for _ in range(iterations):
        delta = delta.detach().requires_grad_(True)
        adv = torch.clamp(clean_t + delta, 0.0, T)
        loss = F.cross_entropy(model(differentiable_angle_encode(adv, T).unsqueeze(0)), label)
        gradient = torch.autograd.grad(loss, delta)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite timing gradient encountered.")
        gradient_norms.append(float(torch.linalg.vector_norm(gradient)))
        with torch.no_grad():
            delta = delta + alpha * gradient.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_(torch.clamp(clean_t + delta, 0.0, T) - clean_t)
            candidate = torch.clamp(clean_t + delta, 0.0, T)
            candidate_loss = float(
                F.cross_entropy(model(differentiable_angle_encode(candidate, T).unsqueeze(0)), label)
            )
            if candidate_loss > best_loss:
                best_loss = candidate_loss
                best_times = candidate.clone()

    result = best_times.numpy().astype(float)
    return result, {
        "objective": "untargeted_cross_entropy",
        "clean_loss": clean_loss,
        "adversarial_loss": best_loss,
        "iterations": int(iterations),
        "step_size": alpha,
        "random_start": bool(random_start),
        "seed": int(seed),
        "max_gradient_norm": max(gradient_norms),
    }
