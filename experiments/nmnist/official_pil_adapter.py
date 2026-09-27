"""Thin adapters around the exact attack source from the paper repository.

The attack implementation is loaded from the immutable upstream Git object.
Only tensor layout and per-example projection adapters are supplied here:
the custom victim consumes [B,T,C,H,W], whereas upstream consumes
[T,B,C,H,W], and upstream's B1/B0 projectors use one counter per invocation.
Splitting projection by example preserves an independent budget while allowing
the required attack/model batch size of 64.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn


UPSTREAM_COMMIT = "f16738f80be94067223f4968c1c318a125b9a321"
UPSTREAM_SOURCE = "utils/attack.py"
UPSTREAM_SOURCE_SHA256 = "e2f76fc6b4796d9cece0dbf498ced5f62c4f9581c17476da34ab41a1087e11ea"
DEFAULT_REPOSITORY = Path(
    r"C:\Users\jafari.h.SPADANACO\AppData\Local\Temp\opencode\Spike-Retiming-Attacks"
)
MATERIALIZED_SOURCE = Path(
    r"C:\Users\jafari.h.SPADANACO\AppData\Local\Temp\opencode\official_attack_f16738_exact.py"
)


def load_official_attack_module(repository: Path = DEFAULT_REPOSITORY):
    """Load the exact upstream Git blob after commit and SHA-256 verification."""
    source = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_COMMIT}:{UPSTREAM_SOURCE}"], cwd=repository
    )
    digest = hashlib.sha256(source).hexdigest()
    if digest != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(f"official attack source hash mismatch: {digest}")
    MATERIALIZED_SOURCE.write_bytes(source)
    name = "spike_retiming_official_f16738"
    spec = importlib.util.spec_from_file_location(name, MATERIALIZED_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct official attack module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TimeMajorVictim(nn.Module):
    """Expose the repository victim through upstream's time-major interface."""

    def __init__(self, victim: nn.Module):
        super().__init__()
        self.victim = victim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("official attack input must be [T,B,C,H,W]")
        return self.victim(value.permute(1, 0, 2, 3, 4).contiguous())


def _cat_projection(base_method, instance, value, probabilities, *args, **kwargs):
    """Apply an unchanged upstream projector independently to each batch item."""
    return_disp = bool(kwargs.get("return_disp", False))
    outputs, displacements = [], []
    for index in range(value.shape[1]):
        result = base_method(
            instance,
            value[:, index:index + 1],
            probabilities[:, index:index + 1],
            *args,
            **kwargs,
        )
        if return_disp:
            output, displacement = result
            outputs.append(output)
            displacements.append(displacement)
        else:
            outputs.append(result)
    output = torch.cat(outputs, dim=1)
    if return_disp:
        return output, torch.cat(displacements, dim=1)
    return output


def build_attack(module, victim: nn.Module, device: torch.device,
                 budget_type: str, budget: int):
    """Instantiate upstream PIL-PGD with paper/official N-MNIST settings."""
    adapted = TimeMajorVictim(victim)
    if budget_type == "B_inf":
        base = module.PGDTimeShiftAfterEncoder_Lowgpu

        class IndependentProjection(base):
            def _final_projection_relaxed_packets_active(self, value, probabilities):
                return _cat_projection(
                    base._final_projection_relaxed_packets_active,
                    self, value, probabilities,
                )

            def _final_projection_packets_greedy_active(
                    self, value, probabilities, return_disp=False):
                return _cat_projection(
                    base._final_projection_packets_greedy_active,
                    self, value, probabilities, return_disp=return_disp,
                )

        return IndependentProjection(
            device=device, model_without_encoder=adapted, D=budget, steps=20,
            alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
            random_start=False, cap_limit=1.0,
        )
    if budget_type == "B1":
        base = module.PGDTimeShiftAfterEncoder_L1

        class IndependentProjection(base):
            def _strict_projection_allT_L1_active_global(
                    self, value, probabilities, step_budget, return_disp=False):
                return _cat_projection(
                    base._strict_projection_allT_L1_active_global,
                    self, value, probabilities, step_budget,
                    return_disp=return_disp,
                )

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
                return _cat_projection(
                    base._strict_projection_allT_L0_active_global,
                    self, value, probabilities, move_budget,
                    return_disp=return_disp,
                )

        return IndependentProjection(
            device=device, model_without_encoder=adapted, steps=40,
            alpha_phi=1.0, lambda_cap=20.0, temperature=1.0,
            random_start=False, cap_limit=1.0, l0_moves_budget=budget,
        )
    raise ValueError(f"unsupported budget type: {budget_type}")
