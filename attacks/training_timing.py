import torch
import torch.nn.functional as F

from attacks.classical_timing import differentiable_angle_encode


def training_timing_pgd(
    model,
    clean_times,
    labels,
    epsilon,
    T=100.0,
    steps=1,
    step_size=None,
    random_start=False,
    generator=None,
):
    """Generate detached batched timing adversaries for first-order training."""
    if clean_times.ndim != 2 or labels.shape != (len(clean_times),):
        raise ValueError("clean_times and labels have incompatible shapes.")
    if T <= 0 or epsilon < 0 or steps < 1:
        raise ValueError("T/steps must be positive and epsilon nonnegative.")
    if not torch.isfinite(clean_times).all() or torch.any(clean_times < 0) or torch.any(clean_times > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    alpha = float(step_size if step_size is not None else epsilon / steps)
    if epsilon == 0:
        return clean_times.detach().clone()
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
        raise ValueError("step_size must be positive and finite.")

    clean = clean_times.detach()
    if random_start:
        delta = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
    else:
        delta = torch.zeros_like(clean)
    delta = (clean + delta).clamp(0.0, T) - clean
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            adversarial = (clean + delta).clamp(0.0, T)
            logits = model(differentiable_angle_encode(adversarial, T))
            objective = F.cross_entropy(logits, labels)
            gradient = torch.autograd.grad(objective, delta, create_graph=False)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError("Non-finite training timing gradient encountered.")
            with torch.no_grad():
                delta = delta + alpha * gradient.sign()
                delta.clamp_(-epsilon, epsilon)
                delta = (clean + delta).clamp(0.0, T) - clean
    finally:
        for parameter, requires_grad in zip(model.parameters(), parameter_flags):
            parameter.requires_grad_(requires_grad)
    return (clean + delta).clamp(0.0, T).detach()
