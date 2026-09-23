import numpy as np
import torch
from encoding.quantum import angle_encode
from metrics.quantum_drift import trace_distance_from_bloch
from metrics.spike_statistics import classical_mismatch


def product_fidelity_torch(theta_a, theta_b):
    if theta_a.shape != theta_b.shape or theta_a.numel() == 0:
        raise ValueError("Angle tensors must have the same non-empty shape.")
    return torch.prod(torch.cos((theta_a - theta_b) / 2.0) ** 2)


def _stealth_surrogate(clean, adv, T):
    clean_isi = torch.diff(torch.sort(clean).values)
    adv_isi = torch.diff(torch.sort(adv).values)
    if clean_isi.numel() == 0:
        return torch.zeros((), dtype=adv.dtype)
    return torch.mean(torch.abs(adv_isi - clean_isi)) / T


def temp_drift_gradient(
    spike_times,
    epsilon,
    tau,
    T=100.0,
    iterations=40,
    step_size=None,
    restarts=3,
    stealth_weight=1.0,
    seed=42,
):
    """Maximize product-state 1-fidelity and retain only exact-feasible candidates."""
    clean = np.asarray(spike_times, dtype=np.float32)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("spike_times must be a non-empty one-dimensional array.")
    if T <= 0 or epsilon < 0 or tau < 0 or iterations < 1 or restarts < 1:
        raise ValueError("Invalid gradient TEMP-DRIFT parameters.")
    if not np.isfinite(stealth_weight) or stealth_weight < 0:
        raise ValueError("stealth_weight must be finite and nonnegative.")
    if not np.all(np.isfinite(clean)) or np.any(clean < 0) or np.any(clean > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    alpha = float(step_size if step_size is not None else epsilon / max(iterations // 4, 1))
    if not np.isfinite(alpha) or alpha <= 0:
        if epsilon == 0 and step_size is None:
            alpha = 1.0
        else:
            raise ValueError("step_size must be positive and finite.")

    clean_t = torch.tensor(clean, dtype=torch.float32)
    clean_theta = (torch.pi / 2.0) * clean_t / T
    best = clean.copy().astype(float)
    best_drift = 0.0
    best_classical = classical_mismatch(clean, clean, T=T)
    feasible_candidates = 1
    evaluated_candidates = 1
    generator = torch.Generator().manual_seed(int(seed))

    for _ in range(restarts):
        if epsilon > 0:
            delta = torch.empty_like(clean_t).uniform_(-epsilon, epsilon, generator=generator)
        else:
            delta = torch.zeros_like(clean_t)
        delta = torch.clamp(clean_t + delta, 0.0, T) - clean_t
        for _ in range(iterations):
            delta = delta.detach().requires_grad_(True)
            adv = torch.clamp(clean_t + delta, 0.0, T)
            adv_theta = (torch.pi / 2.0) * adv / T
            drift = 1.0 - product_fidelity_torch(clean_theta, adv_theta)
            surrogate = _stealth_surrogate(clean_t, adv, T)
            objective = drift - stealth_weight * torch.relu(surrogate - tau)
            gradient = torch.autograd.grad(objective, delta)[0]
            if not torch.isfinite(gradient).all():
                raise RuntimeError("Non-finite TEMP-DRIFT gradient encountered.")
            with torch.no_grad():
                delta = delta + alpha * gradient.sign()
                delta.clamp_(-epsilon, epsilon)
                delta.copy_(torch.clamp(clean_t + delta, 0.0, T) - clean_t)
                candidate = torch.clamp(clean_t + delta, 0.0, T).numpy().astype(float)
            evaluated_candidates += 1
            exact = classical_mismatch(clean, candidate, T=T)
            if exact["delta_cls"] <= tau + 1e-12:
                feasible_candidates += 1
                candidate_theta = angle_encode(candidate, T)
                candidate_drift = trace_distance_from_bloch(angle_encode(clean, T), candidate_theta)
                if candidate_drift > best_drift:
                    best = candidate.copy()
                    best_drift = candidate_drift
                    best_classical = exact

    fidelity = float(np.prod(np.cos((angle_encode(clean, T) - angle_encode(best, T)) / 2.0) ** 2))
    return best, {
        "objective": "product_state_one_minus_fidelity",
        "quantum_drift": best_drift,
        "fidelity": fidelity,
        "one_minus_fidelity": 1.0 - fidelity,
        **best_classical,
        "feasible": bool(best_classical["delta_cls"] <= tau + 1e-12),
        "attack_found": bool(best_drift > 0.0),
        "feasible_candidate_fraction": feasible_candidates / evaluated_candidates,
        "iterations": int(iterations),
        "step_size": alpha,
        "restarts": int(restarts),
        "stealth_surrogate": "mean_absolute_sorted_isi_displacement_over_T",
        "seed": int(seed),
    }

def temp_drift_reference(spike_times, epsilon, tau, T=100.0,
                         steps=200, candidates=64, seed=42):
    t = np.asarray(spike_times, dtype=float)
    if T <= 0 or epsilon < 0 or tau < 0 or steps < 1 or candidates < 1:
        raise ValueError("Invalid TEMP-DRIFT search parameters.")
    if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    rng = np.random.default_rng(seed)
    clean_theta = angle_encode(t, T)
    best = t.copy()
    best_score = 0.0
    best_info = {"quantum_drift": 0.0, **classical_mismatch(t, t, T=T), "score": 0.0}
    for _ in range(steps):
        for _ in range(candidates):
            delta = rng.uniform(-epsilon, epsilon, size=t.shape)
            cand = np.clip(t + delta, 0.0, T)
            q = trace_distance_from_bloch(clean_theta, angle_encode(cand, T))
            c = classical_mismatch(t, cand, T=T)
            score = q
            if c["delta_cls"] <= tau and score > best_score:
                best_score = score
                best = cand
                best_info = {"quantum_drift": q, **c, "score": score}
    return best, best_info


def temp_drift_improved(
    model,
    spike_times,
    true_label,
    epsilon,
    tau,
    T=100.0,
    steps=200,
    candidates=64,
    seed=42,
    sensitive_coordinate=2,
    retained=8,
):
    """Non-gradient, sensitivity-guided TEMP-DRIFT with local elite refinement."""
    t = np.asarray(spike_times, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("spike_times must be a non-empty one-dimensional array.")
    if T <= 0 or epsilon < 0 or tau < 0 or steps < 1 or candidates < 2:
        raise ValueError("Invalid improved TEMP-DRIFT search parameters.")
    if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    if not 0 <= sensitive_coordinate < t.size or retained < 1:
        raise ValueError("Invalid sensitive coordinate or retained-candidate count.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")

    rng = np.random.default_rng(seed)
    clean_theta = angle_encode(t, T)
    label = int(true_label)
    total_queries = int(steps * candidates)
    initial_queries = total_queries // 2

    with torch.no_grad():
        clean_logits = model(torch.tensor(clean_theta, dtype=torch.float32).unsqueeze(0))[0]
    competing = torch.cat((clean_logits[:label], clean_logits[label + 1:])).max()
    clean_margin = float(clean_logits[label] - competing)
    max_quantum_change = max(
        1.0 - np.cos(np.pi * epsilon / (4.0 * T)) ** (2 * t.size), 1e-12
    )

    def evaluate(batch):
        feasible = []
        mismatch = []
        for cand in batch:
            exact = classical_mismatch(t, cand, T=T)
            mismatch.append(exact)
            feasible.append(exact["delta_cls"] <= tau + 1e-12)
        theta = np.asarray([angle_encode(cand, T) for cand in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            true_logits = logits[:, label]
            other_logits = logits.clone()
            other_logits[:, label] = -torch.inf
            margins = true_logits - other_logits.max(dim=1).values
        margin_reduction = clean_margin - margins.numpy()
        quantum_component = np.clip((1.0 - fidelity) / max_quantum_change, 0.0, 1.0)
        margin_component = np.clip(margin_reduction / (abs(clean_margin) + 1.0), -1.0, 1.0)
        scores = 0.5 * quantum_component + 0.5 * margin_component
        scores[~np.asarray(feasible)] = -np.inf
        return scores, fidelity, margins.numpy(), mismatch, feasible

    # Half of the initial proposals explore all coordinates; half focus on TTFS 2.
    global_count = initial_queries // 2
    delta = rng.uniform(-epsilon, epsilon, size=(initial_queries, t.size))
    delta[global_count:] = 0.0
    delta[global_count:, sensitive_coordinate] = rng.uniform(
        -epsilon, epsilon, size=initial_queries - global_count
    )
    initial = np.clip(t + delta, 0.0, T)
    scores, fidelity, margins, mismatch, feasible = evaluate(initial)
    elite_indices = np.argsort(scores)[-min(retained, initial_queries):]
    elites = initial[elite_indices]

    refinement_queries = total_queries - initial_queries
    refined = []
    coordinate_probabilities = np.full(t.size, 0.5 / (t.size - 1))
    coordinate_probabilities[sensitive_coordinate] = 0.5
    for query in range(refinement_queries):
        parent = elites[query % len(elites)]
        coordinate = rng.choice(t.size, p=coordinate_probabilities)
        scale = epsilon * (0.5 if query < refinement_queries // 2 else 0.2)
        candidate = parent.copy()
        candidate[coordinate] += rng.uniform(-scale, scale)
        candidate = np.clip(candidate, t - epsilon, t + epsilon)
        refined.append(np.clip(candidate, 0.0, T))
    all_candidates = np.vstack((initial, np.asarray(refined)))
    scores, fidelity, margins, mismatch, feasible = evaluate(all_candidates)
    best_index = int(np.argmax(scores))
    if not np.isfinite(scores[best_index]):
        best = t.copy()
        best_fidelity = 1.0
        best_margin = clean_margin
        best_mismatch = classical_mismatch(t, t, T=T)
        best_score = 0.0
    else:
        best = all_candidates[best_index]
        best_fidelity = float(fidelity[best_index])
        best_margin = float(margins[best_index])
        best_mismatch = mismatch[best_index]
        best_score = float(scores[best_index])
    return best, {
        "objective": "joint_normalized_one_minus_fidelity_and_margin_reduction",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": best_fidelity,
        "one_minus_fidelity": 1.0 - best_fidelity,
        "clean_margin": clean_margin,
        "attacked_margin": best_margin,
        "margin_reduction": clean_margin - best_margin,
        **best_mismatch,
        "score": best_score,
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "feasible_candidate_fraction": float(np.mean(feasible)),
        "evaluated_candidates": total_queries,
        "initial_candidates": initial_queries,
        "retained_candidates": int(len(elites)),
        "sensitive_coordinate": int(sensitive_coordinate),
        "seed": int(seed),
    }


def temp_drift_adaptive(
    model,
    spike_times,
    true_label,
    epsilon,
    tau,
    T=100.0,
    steps=50,
    candidates=32,
    seed=42,
    sensitive_coordinate=2,
    retained=12,
):
    """Iteratively refine diverse elites with a rank-balanced joint objective."""
    t = np.asarray(spike_times, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("spike_times must contain at least two timing coordinates.")
    if T <= 0 or epsilon < 0 or tau < 0 or steps < 1 or candidates < 2:
        raise ValueError("Invalid adaptive TEMP-DRIFT search parameters.")
    if not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    if not 0 <= sensitive_coordinate < t.size or retained < 3:
        raise ValueError("Invalid sensitive coordinate or retained-candidate count.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")

    total_queries = int(steps * candidates)
    if total_queries < 20:
        raise ValueError("Adaptive TEMP-DRIFT requires at least 20 candidates.")
    rng = np.random.default_rng(seed)
    clean_theta = angle_encode(t, T)
    label = int(true_label)

    with torch.no_grad():
        clean_logits = model(torch.tensor(clean_theta, dtype=torch.float32).unsqueeze(0))[0]
    competing = torch.cat((clean_logits[:label], clean_logits[label + 1:])).max()
    clean_margin = float(clean_logits[label] - competing)

    def evaluate(batch):
        mismatch = [classical_mismatch(t, cand, T=T) for cand in batch]
        feasible = np.asarray([item["delta_cls"] <= tau + 1e-12 for item in mismatch])
        theta = np.asarray([angle_encode(cand, T) for cand in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            other = logits.clone()
            other[:, label] = -torch.inf
            margins = (logits[:, label] - other.max(dim=1).values).numpy()
        return fidelity, margins, mismatch, feasible

    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        result[order] = np.linspace(0.0, 1.0, len(values))
        return result

    def joint_scores(fidelity, margins, feasible):
        quantum_rank = ranks(1.0 - fidelity)
        margin_rank = ranks(clean_margin - margins)
        boundary_bonus = (margins < 0.0).astype(float)
        score = boundary_bonus + 0.35 * quantum_rank + 0.65 * margin_rank
        score[~feasible] = -np.inf
        return score

    def select_elites(pool, fidelity, margins, feasible):
        score = joint_scores(fidelity, margins, feasible)
        joint_count = retained - 4
        ordered = list(np.argsort(score)[::-1][:joint_count])
        ordered += list(np.argsort(np.where(feasible, margins, np.inf))[:2])
        ordered += list(np.argsort(np.where(feasible, fidelity, np.inf))[:2])
        unique = []
        for index in ordered + list(np.argsort(score)[::-1]):
            if np.isfinite(score[index]) and index not in unique:
                unique.append(int(index))
            if len(unique) == min(retained, int(feasible.sum())):
                break
        return np.asarray(unique, dtype=int), score

    initial_queries = min(600, total_queries)
    global_count = initial_queries // 2
    delta = rng.uniform(-epsilon, epsilon, size=(initial_queries, t.size))
    delta[global_count:] = 0.0
    delta[global_count:, sensitive_coordinate] = rng.uniform(
        -epsilon, epsilon, size=initial_queries - global_count
    )
    pool = np.clip(t + delta, 0.0, T)
    fidelity, margins, mismatch, feasible = evaluate(pool)
    elite_indices, score = select_elites(pool, fidelity, margins, feasible)
    elites = pool[elite_indices]

    remaining = total_queries - initial_queries
    generation_sizes = [remaining // 4] * 4
    for index in range(remaining % 4):
        generation_sizes[index] += 1
    coordinate_probabilities = np.full(t.size, 0.6 / (t.size - 1))
    coordinate_probabilities[sensitive_coordinate] = 0.4
    evaluated_candidates = initial_queries
    for generation, generation_size in enumerate(generation_sizes):
        if generation_size == 0:
            continue
        scale = epsilon * (0.5, 0.3, 0.18, 0.1)[generation]
        offspring = []
        for query in range(generation_size):
            parent = elites[query % len(elites)]
            first, second = elites[rng.integers(0, len(elites), size=2)]
            proposal = parent + 0.5 * (first - second)
            mutate_count = int(rng.integers(1, min(3, t.size) + 1))
            coordinates = rng.choice(
                t.size, size=mutate_count, replace=False, p=coordinate_probabilities
            )
            proposal[coordinates] += rng.uniform(-scale, scale, size=mutate_count)
            proposal = np.clip(proposal, t - epsilon, t + epsilon)
            offspring.append(np.clip(proposal, 0.0, T))
        offspring = np.asarray(offspring)
        child_fidelity, child_margins, child_mismatch, child_feasible = evaluate(offspring)
        pool = np.vstack((pool, offspring))
        fidelity = np.concatenate((fidelity, child_fidelity))
        margins = np.concatenate((margins, child_margins))
        mismatch.extend(child_mismatch)
        feasible = np.concatenate((feasible, child_feasible))
        elite_indices, score = select_elites(pool, fidelity, margins, feasible)
        elites = pool[elite_indices]
        evaluated_candidates += generation_size

    best_index = int(np.argmax(score))
    if not np.isfinite(score[best_index]):
        best = t.copy()
        best_fidelity = 1.0
        best_margin = clean_margin
        best_mismatch = classical_mismatch(t, t, T=T)
        best_score = 0.0
    else:
        best = pool[best_index]
        best_fidelity = float(fidelity[best_index])
        best_margin = float(margins[best_index])
        best_mismatch = mismatch[best_index]
        best_score = float(score[best_index])
    return best, {
        "objective": "boundary_prioritized_rank_balanced_fidelity_and_margin",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": best_fidelity,
        "one_minus_fidelity": 1.0 - best_fidelity,
        "clean_margin": clean_margin,
        "attacked_margin": best_margin,
        "margin_reduction": clean_margin - best_margin,
        **best_mismatch,
        "score": best_score,
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "feasible_candidate_fraction": float(np.mean(feasible)),
        "evaluated_candidates": int(evaluated_candidates),
        "initial_candidates": int(initial_queries),
        "generations": 4,
        "retained_candidates": int(len(elites)),
        "sensitive_coordinate": int(sensitive_coordinate),
        "seed": int(seed),
    }


def temp_drift_gradient_adaptive(
    model, spike_times, true_label, epsilon, tau, T=100.0, steps=50,
    candidates=32, seed=42, sensitive_coordinate=2, retained=12,
    progress_context=None,
):
    """Adaptive TEMP initialization followed by projected spike-time gradients."""
    clean = np.asarray(spike_times, dtype=float)
    total_queries = int(steps * candidates)
    if clean.ndim != 1 or clean.size < 2 or total_queries != 1600:
        raise ValueError("Gradient Adaptive TEMP-DRIFT requires 1600 candidates.")
    if T <= 0 or epsilon < 0 or tau < 0 or retained != 12:
        raise ValueError("Invalid Gradient Adaptive TEMP-DRIFT parameters.")
    if not np.all(np.isfinite(clean)) or np.any(clean < 0) or np.any(clean > T):
        raise ValueError("Spike times must be finite and lie in [0,T].")
    if not 0 <= sensitive_coordinate < clean.size:
        raise ValueError("Invalid sensitive coordinate.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")

    rng = np.random.default_rng(seed)
    label = int(true_label)
    clean_theta = angle_encode(clean, T)
    clean_tensor = torch.tensor(clean, dtype=torch.float32)
    clean_theta_tensor = (torch.pi / 2.0) * clean_tensor / T
    with torch.no_grad():
        clean_logits = model(clean_theta_tensor.unsqueeze(0))[0]
    competing = torch.cat((clean_logits[:label], clean_logits[label + 1:])).max()
    clean_margin = float(clean_logits[label] - competing)
    max_fidelity_loss = max(
        1.0 - np.cos(np.pi * epsilon / (4.0 * T)) ** (2 * clean.size), 1e-12
    )

    def evaluate(batch):
        mismatch = [classical_mismatch(clean, candidate, T=T) for candidate in batch]
        feasible = np.asarray([item["delta_cls"] <= tau + 1e-12 for item in mismatch])
        theta = np.asarray([angle_encode(candidate, T) for candidate in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            other = logits.clone()
            other[:, label] = -torch.inf
            margins = (logits[:, label] - other.max(dim=1).values).numpy()
            predictions = logits.argmax(1).numpy()
        return fidelity, margins, predictions, mismatch, feasible

    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=float)
        result[order] = np.linspace(0.0, 1.0, len(values))
        return result

    def initial_scores(fidelity, margins, feasible):
        quantum_rank = ranks(1.0 - fidelity)
        margin_rank = ranks(clean_margin - margins)
        score = (margins < 0.0).astype(float) + 0.35 * quantum_rank + 0.65 * margin_rank
        score[~feasible] = -np.inf
        return score

    def select_initial(pool, fidelity, margins, feasible):
        score = initial_scores(fidelity, margins, feasible)
        ordered = list(np.argsort(score)[::-1][:retained - 4])
        ordered += list(np.argsort(np.where(feasible, margins, np.inf))[:2])
        ordered += list(np.argsort(np.where(feasible, fidelity, np.inf))[:2])
        unique = []
        for index in ordered + list(np.argsort(score)[::-1]):
            if np.isfinite(score[index]) and index not in unique:
                unique.append(int(index))
            if len(unique) == min(retained, int(feasible.sum())):
                break
        return np.asarray(unique, dtype=int)

    initial_queries = 600
    global_count = initial_queries // 2
    delta = rng.uniform(-epsilon, epsilon, size=(initial_queries, clean.size))
    delta[global_count:] = 0.0
    delta[global_count:, sensitive_coordinate] = rng.uniform(
        -epsilon, epsilon, size=initial_queries - global_count
    )
    initial = np.clip(clean + delta, 0.0, T)
    fidelity, margins, predictions, mismatch, feasible = evaluate(initial)
    elite_indices = select_initial(initial, fidelity, margins, feasible)
    elites = initial[elite_indices].copy()
    elite_fidelity = fidelity[elite_indices].copy()
    elite_margins = margins[elite_indices].copy()
    elite_predictions = predictions[elite_indices].copy()
    elite_mismatch = [mismatch[index] for index in elite_indices]

    all_candidates = [candidate.copy() for candidate in initial]
    all_fidelity = list(fidelity)
    all_margins = list(margins)
    all_predictions = list(predictions)
    all_mismatch = list(mismatch)
    all_feasible = list(feasible)
    gradient_norms = []
    rejected_infeasible = 0
    refinement_queries = total_queries - initial_queries
    update_counts = np.full(retained, refinement_queries // retained, dtype=int)
    update_counts[:refinement_queries % retained] += 1
    max_updates = int(update_counts.max())

    for update in range(max_updates):
        active = np.flatnonzero(update_counts > update)
        current = torch.tensor(elites[active], dtype=torch.float32, requires_grad=True)
        theta = (torch.pi / 2.0) * current / T
        logits = model(theta)
        other = logits.clone()
        other[:, label] = -torch.inf
        margin = logits[:, label] - other.max(dim=1).values
        fidelity_t = torch.prod(torch.cos((theta - clean_theta_tensor) / 2.0) ** 2, dim=1)
        quantum = torch.clamp((1.0 - fidelity_t) / max_fidelity_loss, 0.0, 1.0)
        margin_pressure = torch.clamp(-margin / (abs(clean_margin) + 1.0), -1.0, 1.0)
        successful = (margin.detach() < 0.0).float()
        margin_weight = 0.65 - 0.40 * successful
        objective = torch.sum(margin_weight * margin_pressure + (1.0 - margin_weight) * quantum)
        gradient = torch.autograd.grad(objective, current)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("Non-finite Gradient Adaptive TEMP-DRIFT gradient encountered.")
        gradient_norms.extend(torch.linalg.vector_norm(gradient, dim=1).detach().numpy().tolist())
        progress = update / max(max_updates - 1, 1)
        alpha = epsilon * (0.12 * (1.0 - progress) + 0.02)
        with torch.no_grad():
            direction = gradient.sign()
            direction[:, sensitive_coordinate] *= 1.5
            proposal = current + alpha * direction
            lower = torch.maximum(clean_tensor - epsilon, torch.zeros_like(clean_tensor))
            upper = torch.minimum(clean_tensor + epsilon, torch.full_like(clean_tensor, T))
            proposal = torch.maximum(torch.minimum(proposal, upper), lower).numpy().astype(float)
        proposal_fidelity, proposal_margins, proposal_predictions, proposal_mismatch, proposal_feasible = evaluate(proposal)
        for local_index, elite_index in enumerate(active):
            candidate = proposal[local_index]
            is_feasible = bool(proposal_feasible[local_index])
            if is_feasible:
                elites[elite_index] = candidate
                elite_fidelity[elite_index] = proposal_fidelity[local_index]
                elite_margins[elite_index] = proposal_margins[local_index]
                elite_predictions[elite_index] = proposal_predictions[local_index]
                elite_mismatch[elite_index] = proposal_mismatch[local_index]
            else:
                rejected_infeasible += 1
                candidate = elites[elite_index].copy()
                proposal_fidelity[local_index] = elite_fidelity[elite_index]
                proposal_margins[local_index] = elite_margins[elite_index]
                proposal_predictions[local_index] = elite_predictions[elite_index]
                proposal_mismatch[local_index] = elite_mismatch[elite_index]
            all_candidates.append(candidate.copy())
            all_fidelity.append(float(proposal_fidelity[local_index]))
            all_margins.append(float(proposal_margins[local_index]))
            all_predictions.append(int(proposal_predictions[local_index]))
            all_mismatch.append(proposal_mismatch[local_index])
            all_feasible.append(True)
        if progress_context is not None:
            current_success = bool(np.any(
                proposal_feasible & (proposal_predictions != label)
            ))
            print(
                f"[GRAD] seed={progress_context['model_seed']} "
                f"eps={progress_context['epsilon_fraction']:.0%} "
                f"sample={progress_context['sample_index']}/{progress_context['total_samples']} "
                f"step={update + 1}/{max_updates} "
                f"obj={float(objective.detach()):.4f} "
                f"grad_norm={float(torch.linalg.vector_norm(gradient).detach()):.3f} "
                f"1-F={float(np.max(1.0 - proposal_fidelity)):.4f} "
                f"success={current_success} "
                f"elapsed={progress_context['elapsed']():.1f}s",
                flush=True,
            )

    all_candidates = np.asarray(all_candidates)
    all_fidelity = np.asarray(all_fidelity)
    all_margins = np.asarray(all_margins)
    all_predictions = np.asarray(all_predictions)
    all_feasible = np.asarray(all_feasible)
    successful = all_feasible & (all_predictions != label)
    if successful.any():
        best_index = int(np.argmin(np.where(successful, all_fidelity, np.inf)))
    else:
        score = initial_scores(all_fidelity, all_margins, all_feasible)
        best_index = int(np.argmax(score))
    best = all_candidates[best_index]
    best_mismatch = all_mismatch[best_index]
    return best, {
        "objective": "adaptive_margin_pressure_plus_one_minus_fidelity_gradient",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": float(all_fidelity[best_index]),
        "one_minus_fidelity": float(1.0 - all_fidelity[best_index]),
        "clean_margin": clean_margin,
        "attacked_margin": float(all_margins[best_index]),
        **best_mismatch,
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "feasible_candidate_fraction": float(np.mean(all_feasible)),
        "evaluated_candidates": total_queries,
        "initial_candidates": initial_queries,
        "gradient_updates": refinement_queries,
        "gradient_batches": max_updates,
        "max_gradient_norm": float(max(gradient_norms, default=0.0)),
        "rejected_infeasible_updates": rejected_infeasible,
        "retained_candidates": retained,
        "sensitive_coordinate": int(sensitive_coordinate),
        "final_success": bool(all_predictions[best_index] != label),
        "seed": int(seed),
    }


def temp_drift_two_stage(
    model,
    spike_times,
    true_label,
    epsilon,
    tau,
    T=100.0,
    steps=50,
    candidates=32,
    seed=42,
    sensitive_coordinate=2,
):
    """Cross the boundary first, then increase drift without losing success."""
    total_queries = int(steps * candidates)
    if total_queries != 1600:
        raise ValueError("Two-stage TEMP-DRIFT requires exactly 1600 candidates.")
    stage1_queries = 1200
    clean = np.asarray(spike_times, dtype=float)
    stage1, stage1_info = temp_drift_adaptive(
        model,
        clean,
        true_label,
        epsilon,
        tau,
        T=T,
        steps=40,
        candidates=30,
        seed=seed,
        sensitive_coordinate=sensitive_coordinate,
    )
    label = int(true_label)
    rng = np.random.default_rng(seed + 104729)
    clean_theta = angle_encode(clean, T)

    def evaluate(batch):
        mismatch = [classical_mismatch(clean, cand, T=T) for cand in batch]
        feasible = np.asarray([item["delta_cls"] <= tau + 1e-12 for item in mismatch])
        theta = np.asarray([angle_encode(cand, T) for cand in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            predictions = logits.argmax(dim=1).numpy()
            other = logits.clone()
            other[:, label] = -torch.inf
            margins = (logits[:, label] - other.max(dim=1).values).numpy()
        return fidelity, margins, predictions, mismatch, feasible

    stage1_fidelity, stage1_margin, stage1_prediction, _, _ = evaluate(stage1[None, :])
    stage1_success = bool(stage1_prediction[0] != label)
    coordinate_probabilities = np.full(clean.size, 0.6 / (clean.size - 1))
    coordinate_probabilities[sensitive_coordinate] = 0.4
    population = stage1[None, :]
    population_fidelity = stage1_fidelity.copy()
    population_margins = stage1_margin.copy()
    population_predictions = stage1_prediction.copy()
    population_mismatch = [classical_mismatch(clean, stage1, T=T)]
    population_feasible = np.asarray([stage1_info["feasible"]])

    for generation in range(4):
        scale = epsilon * (0.3, 0.18, 0.1, 0.05)[generation]
        offspring = []
        for query in range(100):
            parent = population[query % len(population)]
            first, second = population[rng.integers(0, len(population), size=2)]
            proposal = parent + 0.5 * (first - second)
            mutate_count = int(rng.integers(1, min(3, clean.size) + 1))
            coordinates = rng.choice(
                clean.size, size=mutate_count, replace=False, p=coordinate_probabilities
            )
            proposal[coordinates] += rng.uniform(-scale, scale, size=mutate_count)
            proposal = np.clip(proposal, clean - epsilon, clean + epsilon)
            offspring.append(np.clip(proposal, 0.0, T))
        offspring = np.asarray(offspring)
        fidelity, margins, predictions, mismatch, feasible = evaluate(offspring)
        population = np.vstack((population, offspring))
        population_fidelity = np.concatenate((population_fidelity, fidelity))
        population_margins = np.concatenate((population_margins, margins))
        population_predictions = np.concatenate((population_predictions, predictions))
        population_mismatch.extend(mismatch)
        population_feasible = np.concatenate((population_feasible, feasible))
        if stage1_success:
            eligible = population_feasible & (population_predictions != label)
            ranking = np.where(eligible, population_fidelity, np.inf)
        else:
            # No Stage 2 is started; the reserve remains a boundary-search budget.
            eligible = population_feasible
            ranking = np.where(eligible, population_margins, np.inf)
        keep = np.argsort(ranking)[:min(12, int(eligible.sum()))]
        population = population[keep]
        population_fidelity = population_fidelity[keep]
        population_margins = population_margins[keep]
        population_predictions = population_predictions[keep]
        population_mismatch = [population_mismatch[index] for index in keep]
        population_feasible = population_feasible[keep]

    if stage1_success:
        best_index = int(np.argmin(population_fidelity))
    else:
        successful = population_feasible & (population_predictions != label)
        best_index = int(np.argmin(np.where(successful, population_fidelity, np.inf))) if successful.any() else int(np.argmin(population_margins))
    best = population[best_index]
    best_fidelity = float(population_fidelity[best_index])
    best_mismatch = population_mismatch[best_index]
    final_success = bool(population_predictions[best_index] != label)
    return best, {
        "objective": "two_stage_boundary_then_one_minus_fidelity",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": best_fidelity,
        "one_minus_fidelity": 1.0 - best_fidelity,
        "attacked_margin": float(population_margins[best_index]),
        **best_mismatch,
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "evaluated_candidates": total_queries,
        "stage1_candidates": stage1_queries,
        "stage2_candidates": total_queries - stage1_queries if stage1_success else 0,
        "stage1_success": stage1_success,
        "stage1_success_preserved": bool(stage1_success and final_success),
        "final_success": final_success,
        "sensitive_coordinate": int(sensitive_coordinate),
        "seed": int(seed),
    }


def temp_drift_one_stage(model, spike_times, true_label, epsilon, tau, T=100.0,
                         steps=50, candidates=32, seed=42,
                         sensitive_coordinate=2, retained=12):
    """One-stage diverse search with success-conditional fidelity priority."""
    clean = np.asarray(spike_times, dtype=float)
    total_queries = int(steps * candidates)
    if clean.ndim != 1 or clean.size < 2 or total_queries != 1600:
        raise ValueError("One-stage TEMP-DRIFT requires four timings and 1600 candidates.")
    if T <= 0 or epsilon < 0 or tau < 0 or not 0 <= sensitive_coordinate < clean.size:
        raise ValueError("Invalid one-stage TEMP-DRIFT parameters.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")
    rng = np.random.default_rng(seed)
    label = int(true_label)
    clean_theta = angle_encode(clean, T)
    with torch.no_grad():
        clean_logits = model(torch.tensor(clean_theta, dtype=torch.float32).unsqueeze(0))[0]
    other = torch.cat((clean_logits[:label], clean_logits[label + 1:])).max()
    clean_margin = float(clean_logits[label] - other)

    def evaluate(batch):
        mismatch = [classical_mismatch(clean, candidate, T=T) for candidate in batch]
        feasible = np.asarray([item["delta_cls"] <= tau + 1e-12 for item in mismatch])
        theta = np.asarray([angle_encode(candidate, T) for candidate in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            alternatives = logits.clone()
            alternatives[:, label] = -torch.inf
            margins = (logits[:, label] - alternatives.max(dim=1).values).numpy()
        return fidelity, margins, mismatch, feasible

    def ranks(values):
        order = np.argsort(values, kind="stable")
        ranked = np.empty(len(values), dtype=float)
        ranked[order] = np.linspace(0.0, 1.0, len(values))
        return ranked

    def scores(fidelity, margins, feasible):
        quantum = ranks(1.0 - fidelity)
        margin = ranks(clean_margin - margins)
        successful = margins < 0.0
        score = np.where(successful, 1.0 + 0.75 * quantum + 0.25 * margin,
                         0.35 * quantum + 0.65 * margin)
        score[~feasible] = -np.inf
        return score

    def select(pool, fidelity, margins, feasible):
        score = scores(fidelity, margins, feasible)
        chosen = list(np.argsort(score)[::-1][:6])
        successful = feasible & (margins < 0.0)
        chosen += list(np.argsort(np.where(successful, fidelity, np.inf))[:3])
        candidates_by_score = list(np.argsort(score)[::-1][:64])
        while len(set(chosen)) < retained:
            current = list(dict.fromkeys(chosen))
            available = [index for index in candidates_by_score
                         if index not in current and np.isfinite(score[index])]
            if not available:
                break
            if not current:
                chosen.append(available[0])
                continue
            distances = [min(np.linalg.norm((pool[index] - pool[kept]) / max(epsilon, 1e-12))
                             for kept in current) for index in available]
            chosen.append(available[int(np.argmax(distances))])
        return np.asarray(list(dict.fromkeys(chosen))[:retained], dtype=int), score

    # Global, coordinate-2, and all coordinate-2 pair interactions.
    delta = np.zeros((600, clean.size), dtype=float)
    delta[:300] = rng.uniform(-epsilon, epsilon, size=(300, clean.size))
    delta[300:450, sensitive_coordinate] = rng.uniform(-epsilon, epsilon, size=150)
    partners = [index for index in range(clean.size) if index != sensitive_coordinate]
    for offset, partner in enumerate(partners):
        block = slice(450 + 50 * offset, 500 + 50 * offset)
        delta[block, sensitive_coordinate] = rng.uniform(-epsilon, epsilon, size=50)
        delta[block, partner] = rng.uniform(-epsilon, epsilon, size=50)
    pool = np.clip(clean + delta, 0.0, T)
    fidelity, margins, mismatch, feasible = evaluate(pool)
    elite_indices, score = select(pool, fidelity, margins, feasible)
    elites = pool[elite_indices]

    probabilities = np.full(clean.size, 0.6 / (clean.size - 1))
    probabilities[sensitive_coordinate] = 0.4
    for generation, scale_fraction in enumerate((0.5, 0.3, 0.18, 0.1)):
        offspring = []
        for query in range(250):
            if query < 40:
                proposal = clean + rng.uniform(-epsilon, epsilon, size=clean.size)
            elif query < 100:
                partner = partners[(query - 40) % len(partners)]
                proposal = elites[query % len(elites)].copy()
                for coordinate in (sensitive_coordinate, partner):
                    proposal[coordinate] += rng.uniform(-epsilon * scale_fraction,
                                                        epsilon * scale_fraction)
            else:
                parent = elites[query % len(elites)]
                first, second = elites[rng.integers(0, len(elites), size=2)]
                proposal = parent + 0.5 * (first - second)
                count = int(rng.integers(1, min(3, clean.size) + 1))
                coordinates = rng.choice(clean.size, count, replace=False, p=probabilities)
                proposal[coordinates] += rng.uniform(-epsilon * scale_fraction,
                                                     epsilon * scale_fraction, size=count)
            proposal = np.clip(proposal, clean - epsilon, clean + epsilon)
            offspring.append(np.clip(proposal, 0.0, T))
        offspring = np.asarray(offspring)
        child_fidelity, child_margins, child_mismatch, child_feasible = evaluate(offspring)
        pool = np.vstack((pool, offspring))
        fidelity = np.concatenate((fidelity, child_fidelity))
        margins = np.concatenate((margins, child_margins))
        mismatch.extend(child_mismatch)
        feasible = np.concatenate((feasible, child_feasible))
        elite_indices, score = select(pool, fidelity, margins, feasible)
        elites = pool[elite_indices]

    successful = feasible & (margins < 0.0)
    best_index = (int(np.argmin(np.where(successful, fidelity, np.inf)))
                  if successful.any() else int(np.argmax(score)))
    best = pool[best_index]
    best_mismatch = mismatch[best_index]
    return best, {
        "objective": "success_conditional_fidelity_priority_with_diverse_elites",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": float(fidelity[best_index]),
        "one_minus_fidelity": float(1.0 - fidelity[best_index]),
        "clean_margin": clean_margin,
        "attacked_margin": float(margins[best_index]),
        **best_mismatch,
        "score": float(score[best_index]),
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "feasible_candidate_fraction": float(np.mean(feasible)),
        "evaluated_candidates": total_queries,
        "global_candidates_per_generation": 40,
        "pairwise_candidates_per_generation": 60,
        "pairwise_coordinates": [[sensitive_coordinate, partner] for partner in partners],
        "retained_candidates": int(len(elites)),
        "seed": int(seed),
    }


def temp_drift_quantum_refined(model, spike_times, true_label, epsilon, tau,
                               T=100.0, steps=50, candidates=32, seed=42,
                               sensitive_coordinate=2, retained=12):
    """Refine a boundary-crossing candidate toward timing-budget corners."""
    clean = np.asarray(spike_times, dtype=float)
    total_queries = int(steps * candidates)
    if clean.ndim != 1 or clean.size < 2 or total_queries != 1600:
        raise ValueError("Quantum-refined TEMP-DRIFT requires 1600 candidates.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The victim model must be frozen before attack generation.")
    stage1, stage1_info = temp_drift_adaptive(
        model, clean, true_label, epsilon, tau, T=T, steps=40, candidates=30,
        seed=seed, sensitive_coordinate=sensitive_coordinate,
    )
    rng = np.random.default_rng(seed + 32452843)
    label = int(true_label)
    clean_theta = angle_encode(clean, T)
    partners = [index for index in range(clean.size) if index != sensitive_coordinate]

    def evaluate(batch):
        mismatch = [classical_mismatch(clean, candidate, T=T) for candidate in batch]
        feasible = np.asarray([item["delta_cls"] <= tau + 1e-12 for item in mismatch])
        theta = np.asarray([angle_encode(candidate, T) for candidate in batch])
        fidelity = np.prod(np.cos((theta - clean_theta) / 2.0) ** 2, axis=1)
        with torch.no_grad():
            logits = model(torch.tensor(theta, dtype=torch.float32))
            predictions = logits.argmax(1).numpy()
            other = logits.clone(); other[:, label] = -torch.inf
            margins = (logits[:, label] - other.max(dim=1).values).numpy()
        return fidelity, margins, predictions, mismatch, feasible

    def select(pool, fidelity, margins, predictions, feasible, stage1_success):
        successful = feasible & (predictions != label)
        if stage1_success or successful.any():
            eligible = successful
            ranking = np.where(eligible, fidelity, np.inf)
        else:
            eligible = feasible
            ranking = np.where(eligible, margins, np.inf)
        ordered = list(np.argsort(ranking)[:min(6, int(eligible.sum()))])
        candidates_by_score = list(np.argsort(ranking)[:min(64, int(eligible.sum()))])
        while len(ordered) < min(retained, int(eligible.sum())):
            available = [index for index in candidates_by_score if index not in ordered]
            if not available:
                break
            distances = [min(np.linalg.norm((pool[index] - pool[kept]) /
                                             max(epsilon, 1e-12)) for kept in ordered)
                         for index in available]
            ordered.append(available[int(np.argmax(distances))])
        return np.asarray(ordered, dtype=int)

    fidelity, margins, predictions, mismatch, feasible = evaluate(stage1[None, :])
    stage1_success = bool(feasible[0] and predictions[0] != label)
    pool = stage1[None, :]
    for generation, local_scale in enumerate((0.30, 0.18, 0.10, 0.05)):
        elite = select(pool, fidelity, margins, predictions, feasible, stage1_success)
        parents = pool[elite]
        offspring = []
        for query in range(100):
            parent = parents[query % len(parents)].copy()
            if query < 15:
                proposal = clean + rng.uniform(-epsilon, epsilon, size=clean.size)
            elif query < 55:
                partner = partners[(query - 15) % len(partners)]
                proposal = parent.copy()
                for coordinate in (sensitive_coordinate, partner):
                    direction = np.sign(proposal[coordinate] - clean[coordinate])
                    if direction == 0 or rng.random() < 0.2:
                        direction = rng.choice((-1.0, 1.0))
                    proposal[coordinate] = clean[coordinate] + direction * epsilon
            elif query < 80:
                proposal = parent.copy()
                direction = np.sign(proposal - clean)
                direction[direction == 0] = rng.choice((-1.0, 1.0), size=int(np.sum(direction == 0)))
                mask = rng.random(clean.size) < 0.75
                proposal[mask] = clean[mask] + direction[mask] * epsilon
            else:
                first, second = parents[rng.integers(0, len(parents), size=2)]
                proposal = parent + 0.5 * (first - second)
                coordinates = rng.choice(clean.size, size=2, replace=False)
                proposal[coordinates] += rng.uniform(-epsilon * local_scale,
                                                     epsilon * local_scale, size=2)
            proposal = np.clip(proposal, clean - epsilon, clean + epsilon)
            offspring.append(np.clip(proposal, 0.0, T))
        child = np.asarray(offspring)
        child_fidelity, child_margins, child_predictions, child_mismatch, child_feasible = evaluate(child)
        pool = np.vstack((pool, child)); fidelity = np.concatenate((fidelity, child_fidelity))
        margins = np.concatenate((margins, child_margins)); predictions = np.concatenate((predictions, child_predictions))
        mismatch.extend(child_mismatch); feasible = np.concatenate((feasible, child_feasible))
        keep = select(pool, fidelity, margins, predictions, feasible, stage1_success)
        pool, fidelity, margins, predictions, feasible = (pool[keep], fidelity[keep], margins[keep],
                                                           predictions[keep], feasible[keep])
        mismatch = [mismatch[index] for index in keep]

    successful = feasible & (predictions != label)
    best_index = (int(np.argmin(np.where(successful, fidelity, np.inf)))
                  if successful.any() else int(np.argmin(np.where(feasible, margins, np.inf))))
    best = pool[best_index]; best_mismatch = mismatch[best_index]
    final_success = bool(predictions[best_index] != label)
    return best, {
        "objective": "boundary_then_success_gated_outward_fidelity_refinement",
        "quantum_drift": trace_distance_from_bloch(clean_theta, angle_encode(best, T)),
        "fidelity": float(fidelity[best_index]),
        "one_minus_fidelity": float(1.0 - fidelity[best_index]),
        "attacked_margin": float(margins[best_index]), **best_mismatch,
        "feasible": bool(best_mismatch["delta_cls"] <= tau + 1e-12),
        "evaluated_candidates": total_queries, "stage1_candidates": 1200,
        "stage2_candidates": 400 if stage1_success else 0,
        "stage1_success": stage1_success,
        "stage1_success_preserved": bool(stage1_success and final_success),
        "final_success": final_success,
        "pairwise_coordinates": [[sensitive_coordinate, partner] for partner in partners],
        "retained_candidates": retained, "seed": int(seed),
    }
