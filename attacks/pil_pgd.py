"""Projected-in-the-loop PGD spike-retiming attack.

Clean-room implementation of Eqs. (8)--(14), Algorithm 1, and Appendix D of
Yu et al., arXiv:2602.03284v1. Input is [B,T,C,H,W]. One nonzero grid cell is
one indivisible amplitude-bearing packet whose complete value is conserved.
Integer counts must never be expanded into unit packets.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

try:
    from numba import cuda
except Exception:  # pragma: no cover - CPU-only environments retain reference path
    cuda = None


@dataclass(frozen=True)
class PILPGDConfig:
    budget_type: str
    beta: int
    iterations: int | None = None
    temperature: float = 1.0
    alpha: float = 1.0
    logit_clip: float = 10.0
    capacity_weight: float = 20.0
    budget_weight: float = 10.0
    projector: str = "cpu"

    def __post_init__(self):
        normalized = self.budget_type.replace("∞", "inf").lower()
        if normalized not in {"b_inf", "binf", "b1", "b0"}:
            raise ValueError("budget_type must be B_inf, B1, or B0")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.projector not in {"cpu", "cuda"}:
            raise ValueError("projector must be 'cpu' or 'cuda'")


def _kind(value: str) -> str:
    value = value.replace("∞", "inf").lower()
    return "B_inf" if value in {"b_inf", "binf"} else value.upper()


def _targets(time_bins: int, kind: str, beta: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.arange(time_bins, device=device)[:, None]
    if kind == "B_inf":
        offsets = torch.arange(-beta, beta + 1, device=device)[None, :]
        target = source + offsets
    else:
        target = torch.arange(time_bins, device=device)[None, :].expand(time_bins, -1)
    valid = (target >= 0) & (target < time_bins)
    return target.clamp(0, time_bins - 1), valid


def soft_retime(clean: torch.Tensor, probabilities: torch.Tensor,
                target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = clean.shape
    lines = c * h * w
    x = clean.reshape(b, t, lines)
    pi = probabilities.reshape(b, t, lines, -1)
    options = pi.shape[-1]
    contribution = x.unsqueeze(-1) * pi * valid.view(1, t, 1, options)
    line = torch.arange(lines, device=clean.device).view(1, 1, lines, 1)
    destination = target.view(1, t, 1, options) * lines + line
    destination = destination.expand(b, -1, -1, -1).reshape(b, -1)
    out = torch.zeros((b, t * lines), device=clean.device, dtype=clean.dtype)
    out.scatter_add_(1, destination, contribution.reshape(b, -1))
    return out.reshape_as(clean)


@torch.no_grad()
def strict_project_grid(clean: torch.Tensor, probabilities: torch.Tensor,
                        budget_type: str, beta: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Score-ordered P* with origin reservations and global per-sample budget.

    Projection is discrete and non-differentiable.  Its bookkeeping is performed
    on CPU after one bulk transfer, avoiding thousands of GPU scalar
    synchronizations while preserving the original candidate tuples, sort order,
    reservation rules, and exact output values.
    """
    kind = _kind(budget_type)
    b, t, c, h, w = clean.shape
    lines = c * h * w
    x = clean.reshape(b, t, lines)
    pi = probabilities.reshape(b, t, lines, -1)
    target, valid = _targets(t, kind, beta, clean.device)
    x_cpu = x.detach().cpu()
    pi_cpu = pi.detach().cpu()
    target_cpu = target.detach().cpu()
    valid_cpu = valid.detach().cpu()
    out_cpu = torch.zeros_like(x_cpu)
    displacement_cpu = torch.zeros_like(x_cpu, dtype=torch.int64)
    for sample in range(b):
        sources = torch.nonzero(x_cpu[sample] != 0, as_tuple=False)
        reserved = {(int(s), int(line)) for s, line in sources.tolist()}
        occupied: set[tuple[int, int]] = set()
        placed: set[tuple[int, int]] = set()
        candidates = []
        for s_tensor, line_tensor in sources:
            s, line = int(s_tensor), int(line_tensor)
            for option in range(pi_cpu.shape[-1]):
                if not bool(valid_cpu[s, option]):
                    continue
                destination = int(target_cpu[s, option])
                if destination == s:
                    continue
                score = float(pi_cpu[sample, s, line, option])
                candidates.append((-score, abs(destination - s), s, line, destination))
        candidates.sort()
        spent = 0
        for _, distance, source, line, destination in candidates:
            key = (source, line)
            destination_key = (destination, line)
            if key in placed or destination_key in occupied or destination_key in reserved:
                continue
            cost = distance if kind == "B1" else 1 if kind == "B0" else 0
            if kind == "B_inf" and distance > beta:
                continue
            if kind in {"B1", "B0"} and spent + cost > beta:
                continue
            out_cpu[sample, destination, line] = x_cpu[sample, source, line]
            displacement_cpu[sample, source, line] = destination - source
            occupied.add(destination_key)
            reserved.discard(key)
            placed.add(key)
            spent += cost
        for source_tensor, line_tensor in sources:
            source, line = int(source_tensor), int(line_tensor)
            if (source, line) not in placed:
                out_cpu[sample, source, line] = x_cpu[sample, source, line]
                occupied.add((source, line))
    out = out_cpu.to(device=clean.device, dtype=clean.dtype)
    displacement = displacement_cpu.to(device=clean.device)
    return out.reshape_as(clean), displacement.reshape_as(clean)


if cuda is not None:
    @cuda.jit
    def _greedy_projection_kernel(x, candidate_source, candidate_line,
                                  candidate_target, candidate_distance,
                                  candidate_count, reserved, occupied, placed,
                                  out, displacement, budget_kind, beta):
        sample = cuda.grid(1)
        if sample >= x.shape[0]:
            return
        spent = 0
        for rank in range(candidate_count[sample]):
            source = candidate_source[sample, rank]
            line = candidate_line[sample, rank]
            destination = candidate_target[sample, rank]
            distance = candidate_distance[sample, rank]
            if placed[sample, source, line] != 0:
                continue
            if occupied[sample, destination, line] != 0:
                continue
            if reserved[sample, destination, line] != 0:
                continue
            cost = 0
            if budget_kind == 1:
                cost = distance
            elif budget_kind == 2:
                cost = 1
            if budget_kind == 0 and distance > beta:
                continue
            if (budget_kind == 1 or budget_kind == 2) and spent + cost > beta:
                continue
            out[sample, destination, line] = x[sample, source, line]
            displacement[sample, source, line] = destination - source
            occupied[sample, destination, line] = 1
            reserved[sample, source, line] = 0
            placed[sample, source, line] = 1
            spent += cost
        for source in range(x.shape[1]):
            for line in range(x.shape[2]):
                if x[sample, source, line] != 0 and placed[sample, source, line] == 0:
                    out[sample, source, line] = x[sample, source, line]
                    occupied[sample, source, line] = 1


def _stable_lexicographic_order(keys_least_to_most: list[torch.Tensor]) -> torch.Tensor:
    batch, count = keys_least_to_most[0].shape
    order = torch.arange(count, device=keys_least_to_most[0].device).expand(batch, -1)
    for key in keys_least_to_most:
        values = torch.gather(key, 1, order)
        local = torch.argsort(values, dim=1, stable=True)
        order = torch.gather(order, 1, local)
    return order


@torch.no_grad()
def strict_project_grid_cuda(clean: torch.Tensor, probabilities: torch.Tensor,
                             budget_type: str, beta: int) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA-vectorized candidate preparation plus exact greedy projection kernel."""
    if clean.device.type != "cuda" or cuda is None:
        raise RuntimeError("CUDA projector requires a CUDA tensor and Numba CUDA")
    kind = _kind(budget_type)
    b, t, c, h, w = clean.shape
    lines, cells = c * h * w, t * c * h * w
    x = clean.reshape(b, t, lines).contiguous()
    pi = probabilities.reshape(b, t, lines, -1).contiguous()
    target, valid = _targets(t, kind, beta, clean.device)

    active = x.reshape(b, cells) != 0
    packet_counts = active.sum(1, dtype=torch.int64)
    max_packets = int(packet_counts.max().item())
    flat_index = torch.arange(cells, device=clean.device).expand(b, -1)
    sentinel = torch.full_like(flat_index, cells)
    active_keys = torch.where(active, flat_index, sentinel)
    source_flat = torch.sort(active_keys, dim=1, stable=True).values[:, :max_packets]
    packet_valid = torch.arange(max_packets, device=clean.device)[None, :] < packet_counts[:, None]
    safe_source = source_flat.clamp_max(cells - 1)
    source_t = safe_source // lines
    source_line = safe_source % lines

    options = pi.shape[-1]
    pi_flat = pi.reshape(b, cells, options)
    source_scores = torch.gather(pi_flat, 1, safe_source[:, :, None].expand(-1, -1, options))
    source_targets = target[source_t]
    source_option_valid = valid[source_t] & packet_valid[:, :, None]
    source_option_valid &= source_targets != source_t[:, :, None]
    source_times = source_t[:, :, None].expand(-1, -1, options)
    source_lines = source_line[:, :, None].expand(-1, -1, options)
    distances = (source_targets - source_times).abs()

    candidate_source = source_times.reshape(b, -1).to(torch.int64)
    candidate_line = source_lines.reshape(b, -1).to(torch.int64)
    candidate_target = source_targets.reshape(b, -1).to(torch.int64)
    candidate_distance = distances.reshape(b, -1).to(torch.int64)
    candidate_valid = source_option_valid.reshape(b, -1)
    negative_score = -source_scores.reshape(b, -1)
    negative_score = torch.where(candidate_valid, negative_score,
                                 torch.full_like(negative_score, torch.inf))
    # Python tuple order: (-score, distance, source, line, destination).
    order = _stable_lexicographic_order([
        candidate_target, candidate_line, candidate_source,
        candidate_distance, negative_score,
    ])
    candidate_source = torch.gather(candidate_source, 1, order).contiguous()
    candidate_line = torch.gather(candidate_line, 1, order).contiguous()
    candidate_target = torch.gather(candidate_target, 1, order).contiguous()
    candidate_distance = torch.gather(candidate_distance, 1, order).contiguous()
    candidate_count = candidate_valid.sum(1, dtype=torch.int64).contiguous()

    reserved = (x != 0).to(torch.uint8).contiguous()
    occupied = torch.zeros_like(reserved)
    placed = torch.zeros_like(reserved)
    out = torch.zeros_like(x)
    displacement = torch.zeros_like(x, dtype=torch.int64)
    budget_kind = {"B_inf": 0, "B1": 1, "B0": 2}[kind]
    stream = cuda.external_stream(torch.cuda.current_stream(clean.device).cuda_stream)
    threads = 32
    _greedy_projection_kernel[(b + threads - 1) // threads, threads, stream](
        cuda.as_cuda_array(x), cuda.as_cuda_array(candidate_source),
        cuda.as_cuda_array(candidate_line), cuda.as_cuda_array(candidate_target),
        cuda.as_cuda_array(candidate_distance), cuda.as_cuda_array(candidate_count),
        cuda.as_cuda_array(reserved), cuda.as_cuda_array(occupied), cuda.as_cuda_array(placed),
        cuda.as_cuda_array(out), cuda.as_cuda_array(displacement), budget_kind, int(beta))
    return out.reshape_as(clean), displacement.reshape_as(clean)


class PILPGDAttack:
    def __init__(self, model, config: PILPGDConfig):
        self.model = model
        self.config = config

    def __call__(self, clean: torch.Tensor, labels: torch.Tensor):
        kind = _kind(self.config.budget_type)
        b, t, c, h, w = clean.shape
        target, valid = _targets(t, kind, self.config.beta, clean.device)
        options = target.shape[1]
        logits = torch.zeros((b, t, c, h, w, options), device=clean.device,
                             dtype=clean.dtype, requires_grad=True)
        source_mask = (clean != 0).unsqueeze(-1)
        iterations = self.config.iterations or (20 if kind == "B_inf" else 40)
        source_times = torch.arange(t, device=clean.device).view(1, t, 1, 1, 1, 1)
        target_times = target.view(1, t, 1, 1, 1, options)
        distance = (target_times - source_times).abs().to(clean.dtype)
        projector = strict_project_grid_cuda if self.config.projector == "cuda" else strict_project_grid

        def exact_model_forward(value: torch.Tensor) -> torch.Tensor:
            # The validated victim uses batch-1 cuDNN arithmetic. A mathematically
            # batched convolution can change near-zero attack-gradient signs, so
            # retain independent forwards while batching every projection tensor.
            if self.config.projector == "cuda" and value.shape[0] > 1:
                return torch.cat([self.model(value[index:index + 1])
                                  for index in range(value.shape[0])], dim=0)
            return self.model(value)

        for _ in range(iterations):
            masked = logits.masked_fill(~(source_mask & valid.view(1, t, 1, 1, 1, options)), -torch.inf)
            probabilities = torch.softmax(masked / self.config.temperature, dim=-1)
            probabilities = torch.nan_to_num(probabilities)
            soft = soft_retime(clean, probabilities, target, valid)
            hard, _ = projector(clean, probabilities, kind, self.config.beta)
            pil = hard + soft - soft.detach()
            # Sum independent per-sample objectives. For B=1 this is identical to
            # the scalar runner; for B>1 no sample's packet count or loss scales
            # another sample's gradient.
            task = F.cross_entropy(exact_model_forward(pil), labels, reduction="none")
            occupancy = soft_retime((clean != 0).to(clean.dtype), probabilities, target, valid)
            capacity = F.relu(occupancy - 1).square().flatten(1).sum(1)
            capacity = capacity / source_mask.flatten(1).sum(1).clamp_min(1)
            if kind == "B1":
                soft_cost = (probabilities * distance * source_mask).sum(dim=(1, 2, 3, 4, 5))
            elif kind == "B0":
                stay = target_times == source_times
                soft_cost = ((probabilities * (~stay).to(clean.dtype)) * source_mask).sum(dim=(1, 2, 3, 4, 5))
            else:
                soft_cost = torch.zeros(b, device=clean.device)
            budget_penalty = F.relu(soft_cost / max(self.config.beta, 1) - 1)
            objective = (task - self.config.capacity_weight * capacity
                         - self.config.budget_weight * budget_penalty).sum()
            gradient, = torch.autograd.grad(objective, logits)
            with torch.no_grad():
                logits.add_(self.config.alpha * gradient.sign()).clamp_(-self.config.logit_clip,
                                                                        self.config.logit_clip)
            logits.requires_grad_(True)
        masked = logits.masked_fill(~(source_mask & valid.view(1, t, 1, 1, 1, options)), -torch.inf)
        probabilities = torch.nan_to_num(torch.softmax(masked / self.config.temperature, dim=-1))
        final, displacement = projector(clean, probabilities, kind, self.config.beta)
        # Freeze evidence that the exact strict tensor returned to serialization
        # is also the tensor evaluated by the victim model.
        self.last_evaluated_input = final.detach().clone()
        with torch.no_grad():
            self.last_logits = exact_model_forward(final).detach().clone()
        return final, displacement


def run_independent_parallel(model, clean: torch.Tensor, labels: torch.Tensor,
                             config: PILPGDConfig, max_workers: int):
    """Run independent batch-size-1 attacks concurrently on CUDA streams.

    This is deliberately not mathematical sample batching: every sample gets a
    separate attack object, logits, scalar objective, projection state, and CUDA
    stream.  Consequently the victim model is always called with batch size one,
    preserving the batch-size-1 convolution/LIF arithmetic.
    """
    if clean.shape[0] != labels.shape[0]:
        raise ValueError("clean batch and labels must have equal leading dimension")
    count = int(clean.shape[0])
    workers = max(1, min(int(max_workers), count))

    def attack_one(index: int):
        attack = PILPGDAttack(model, config)
        if clean.device.type == "cuda":
            stream = torch.cuda.Stream(device=clean.device)
            with torch.cuda.stream(stream):
                adversarial, displacement = attack(clean[index:index + 1], labels[index:index + 1])
            stream.synchronize()
        else:
            adversarial, displacement = attack(clean[index:index + 1], labels[index:index + 1])
        return (index, adversarial.detach(), displacement.detach(),
                attack.last_logits.detach(), attack.last_evaluated_input.detach())

    if workers == 1:
        results = [attack_one(index) for index in range(count)]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pil-independent") as pool:
            results = list(pool.map(attack_one, range(count)))
    results.sort(key=lambda row: row[0])
    adversarial = torch.cat([row[1] for row in results], dim=0)
    displacement = torch.cat([row[2] for row in results], dim=0)
    logits = torch.cat([row[3] for row in results], dim=0)
    evaluated = torch.cat([row[4] for row in results], dim=0)
    return adversarial, displacement, logits, evaluated
