"""Equivalence-tested batched adapter for the exact upstream PIL-PGD source."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from numba import cuda

from experiments.nmnist.official_pil_adapter import (
    DEFAULT_REPOSITORY,
    MATERIALIZED_SOURCE,
    UPSTREAM_COMMIT,
    UPSTREAM_SOURCE,
    UPSTREAM_SOURCE_SHA256,
    TimeMajorVictim,
    _cat_projection,
    load_official_attack_module,
)


@cuda.jit
def _official_global_budget_projection_kernel(
        x, candidate_source, candidate_line, candidate_target,
        candidate_distance, candidate_count, reserved, occupied, moved,
        output, displacement, budget, is_l0):
    """One independent upstream greedy scan per sample."""
    sample = cuda.grid(1)
    if sample >= x.shape[0]:
        return
    remaining = budget
    for rank in range(candidate_count[sample]):
        if remaining <= 0:
            break
        source = candidate_source[sample, rank]
        line = candidate_line[sample, rank]
        target = candidate_target[sample, rank]
        distance = candidate_distance[sample, rank]
        if moved[sample, source, line] != 0:
            continue
        if occupied[sample, target, line] != 0:
            continue
        if reserved[sample, target, line] != 0:
            continue
        cost = 1 if is_l0 != 0 else distance
        # This break, rather than continue, deliberately reproduces upstream L1.
        if cost > remaining:
            break
        output[sample, target, line] = x[sample, source, line]
        occupied[sample, target, line] = 1
        reserved[sample, source, line] = 0
        moved[sample, source, line] = 1
        displacement[sample, source, line] = target - source
        remaining -= cost
    for source in range(x.shape[1]):
        for line in range(x.shape[2]):
            if x[sample, source, line] != 0 and moved[sample, source, line] == 0:
                output[sample, source, line] = x[sample, source, line]


@cuda.jit
def _official_binf_line_projection_kernel(x, probabilities, output, displacement, radius):
    """Exact greedy order within each independent event line."""
    index = cuda.grid(1)
    lines = x.shape[2]
    if index >= x.shape[0] * lines:
        return
    sample = index // lines
    line = index % lines
    time_bins = x.shape[1]
    source_mask = 0
    for source in range(time_bins):
        if x[sample, source, line] > 0:
            source_mask |= 1 << source
    reserved = source_mask
    unplaced = source_mask
    occupied = 0
    while unplaced != 0:
        best_score = -3.402823466e38
        best_source = -1
        best_target = -1
        for source in range(time_bins):
            if (unplaced & (1 << source)) == 0:
                continue
            for k in range(2 * radius + 1):
                target = source + k - radius
                if target < 0 or target >= time_bins:
                    continue
                if (occupied & (1 << target)) != 0:
                    continue
                if target != source and (reserved & (1 << target)) != 0:
                    continue
                score = probabilities[sample, source, line, k]
                if score > best_score:
                    best_score = score
                    best_source = source
                    best_target = target
        if best_source < 0:
            for source in range(time_bins):
                if (unplaced & (1 << source)) != 0:
                    output[sample, source, line] = x[sample, source, line]
            break
        output[sample, best_target, line] = x[sample, best_source, line]
        displacement[sample, best_source, line] = best_target - best_source
        occupied |= 1 << best_target
        reserved &= ~(1 << best_source)
        unplaced &= ~(1 << best_source)


@torch.no_grad()
def _batched_official_binf_projection(value: torch.Tensor, probabilities: torch.Tensor,
                                      radius: int, return_disp: bool = False):
    time_bins, batch, channels, height, width = value.shape
    lines = channels * height * width
    x = value.permute(1, 0, 2, 3, 4).reshape(batch, time_bins, lines).contiguous()
    pi = probabilities.permute(1, 0, 2, 3, 4, 5).reshape(
        batch, time_bins, lines, 2 * radius + 1).contiguous()
    output = torch.zeros_like(x)
    displacement = torch.zeros_like(x, dtype=torch.int16)
    stream = cuda.external_stream(torch.cuda.current_stream(value.device).cuda_stream)
    threads = 128
    count = batch * lines
    _official_binf_line_projection_kernel[(count + threads - 1) // threads, threads, stream](
        cuda.as_cuda_array(x), cuda.as_cuda_array(pi), cuda.as_cuda_array(output),
        cuda.as_cuda_array(displacement), radius,
    )
    output = output.reshape(batch, time_bins, channels, height, width).permute(1, 0, 2, 3, 4).contiguous()
    displacement = displacement.reshape(batch, time_bins, channels, height, width).permute(1, 0, 2, 3, 4).contiguous()
    if return_disp:
        return output, displacement
    return output


@torch.no_grad()
def _batched_official_global_projection(value: torch.Tensor, probabilities: torch.Tensor,
                                        budget: int, is_l0: bool,
                                        return_disp: bool = False):
    """CUDA batching of the unchanged upstream L1/L0 candidate order and scan."""
    time_bins, batch, channels, height, width = value.shape
    lines = channels * height * width
    x = value.permute(1, 0, 2, 3, 4).reshape(batch, time_bins, lines).contiguous()
    pi = probabilities.permute(1, 0, 2, 3, 4, 5).reshape(
        batch, time_bins, lines, time_bins).contiguous()
    active = x.reshape(batch, -1) != 0
    packet_counts = active.sum(1, dtype=torch.int64)
    max_packets = int(packet_counts.max().item())
    cell_count = time_bins * lines
    flat = torch.arange(cell_count, device=value.device).expand(batch, -1)
    source_flat = torch.sort(
        torch.where(active, flat, torch.full_like(flat, cell_count)), dim=1
    ).values[:, :max_packets]
    packet_valid = torch.arange(max_packets, device=value.device)[None, :] < packet_counts[:, None]
    safe_source = source_flat.clamp_max(cell_count - 1)
    source_t = safe_source // lines
    source_line = safe_source % lines
    source_scores = torch.gather(
        pi.reshape(batch, cell_count, time_bins), 1,
        safe_source[:, :, None].expand(-1, -1, time_bins),
    )
    # Recreate each upstream candidate vector *without padding* before argsort.
    # This matters for exact ties because torch's default argsort is unstable and
    # its tie permutation depends on vector length. The CUDA scan remains batched.
    packet_counts_cpu = packet_counts.cpu().tolist()
    max_candidates = max_packets * (time_bins - 1)
    all_source, all_line, all_target, all_distance = [], [], [], []
    target_axis = torch.arange(time_bins, device=value.device)
    tie_epsilon = torch.tensor(1e-6, device=value.device, dtype=value.dtype)
    for sample, packet_count in enumerate(packet_counts_cpu):
        count = int(packet_count)
        local_source = source_t[sample, :count]
        local_line = source_line[sample, :count]
        target = target_axis.view(1, time_bins).expand(count, -1)
        source_expanded = local_source[:, None].expand_as(target)
        valid = target != source_expanded
        distance = (target - source_expanded).abs()
        key = source_scores[sample, :count] + tie_epsilon * (
            (time_bins - 1) - distance).to(value.dtype)
        key = key[valid]
        source_vector = source_expanded[valid]
        line_vector = local_line[:, None].expand_as(target)[valid]
        target_vector = target[valid]
        distance_vector = distance[valid]
        order = torch.argsort(key, descending=True)
        padding = max_candidates - order.numel()
        all_source.append(F.pad(source_vector[order], (0, padding)))
        all_line.append(F.pad(line_vector[order], (0, padding)))
        all_target.append(F.pad(target_vector[order], (0, padding)))
        all_distance.append(F.pad(distance_vector[order], (0, padding)))
    source_candidates = torch.stack(all_source).to(torch.int64).contiguous()
    line_candidates = torch.stack(all_line).to(torch.int64).contiguous()
    target_candidates = torch.stack(all_target).to(torch.int64).contiguous()
    distance_candidates = torch.stack(all_distance).to(torch.int64).contiguous()
    candidate_count = (packet_counts * (time_bins - 1)).contiguous()
    reserved = (x != 0).to(torch.uint8).contiguous()
    occupied = torch.zeros_like(reserved)
    moved = torch.zeros_like(reserved)
    output = torch.zeros_like(x)
    displacement = torch.zeros_like(x, dtype=torch.int16)
    stream = cuda.external_stream(torch.cuda.current_stream(value.device).cuda_stream)
    threads = 32
    _official_global_budget_projection_kernel[(batch + threads - 1) // threads, threads, stream](
        cuda.as_cuda_array(x), cuda.as_cuda_array(source_candidates),
        cuda.as_cuda_array(line_candidates), cuda.as_cuda_array(target_candidates),
        cuda.as_cuda_array(distance_candidates), cuda.as_cuda_array(candidate_count),
        cuda.as_cuda_array(reserved), cuda.as_cuda_array(occupied), cuda.as_cuda_array(moved),
        cuda.as_cuda_array(output), cuda.as_cuda_array(displacement), int(budget), int(is_l0),
    )
    output = output.reshape(batch, time_bins, channels, height, width).permute(1, 0, 2, 3, 4).contiguous()
    displacement = displacement.reshape(batch, time_bins, channels, height, width).permute(1, 0, 2, 3, 4).contiguous()
    if return_disp:
        return output, displacement
    return output


def build_attack(module, victim: nn.Module, device: torch.device,
                 budget_type: str, budget: int):
    adapted = TimeMajorVictim(victim)
    if budget_type == "B_inf":
        # Upstream B_inf has no batch-shared budget counter. Both relaxed and
        # strict projectors are event-line separable. Execute the strict greedy
        # order independently on CUDA for each event line.
        base = module.PGDTimeShiftAfterEncoder_Lowgpu

        class HybridProjection(base):
            def _final_projection_packets_greedy_active(
                    self, value, probabilities, return_disp=False):
                return _batched_official_binf_projection(
                    value, probabilities, self.D, return_disp=return_disp)

        return HybridProjection(
            device=device, model_without_encoder=adapted, D=budget, steps=20,
            alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
            random_start=False, cap_limit=1.0,
        )
    if budget_type == "B1":
        base = module.PGDTimeShiftAfterEncoder_L1

        class IndependentProjection(base):
            def _strict_projection_allT_L1_active_global(
                    self, value, probabilities, step_budget, return_disp=False):
                return _batched_official_global_projection(
                    value, probabilities, step_budget, False, return_disp)

        return IndependentProjection(
            device=device, model_without_encoder=adapted, steps=40,
            alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
            random_start=False, cap_limit=1.0, l1_steps_budget=budget,
        )
    if budget_type == "B0":
        base = module.PGDTimeShiftAfterEncoder_L0

        class IndependentProjection(base):
            def _strict_projection_allT_L0_active_global(
                    self, value, probabilities, move_budget, return_disp=False):
                return _batched_official_global_projection(
                    value, probabilities, move_budget, True, return_disp)

        return IndependentProjection(
            device=device, model_without_encoder=adapted, steps=40,
            alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
            random_start=False, cap_limit=1.0, l0_moves_budget=budget,
        )
    raise ValueError(f"unsupported budget type: {budget_type}")
