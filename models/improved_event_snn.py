"""Repository-native convolutional SNN for 128x128 event-frame benchmarks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FastSigmoidSpike(torch.autograd.Function):
    """Hard spike with a fast-sigmoid surrogate derivative."""

    @staticmethod
    def forward(ctx, voltage: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (voltage,) = ctx.saved_tensors
        return grad_output / (1.0 + 5.0 * voltage.abs()).square()


def lif_sequence(current: torch.Tensor, decay: float, threshold: float) -> torch.Tensor:
    """Apply a reset-by-subtraction LIF neuron to [B,T,C,H,W] currents."""
    membrane = torch.zeros_like(current[:, 0])
    spikes = []
    for step in range(current.shape[1]):
        membrane = decay * membrane + current[:, step]
        spike = FastSigmoidSpike.apply(membrane - threshold)
        membrane = membrane - spike.detach() * threshold
        spikes.append(spike)
    return torch.stack(spikes, dim=1)


class SpikingConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, *, pool: bool,
                 decay: float, threshold: float, pool_after_spike: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outgoing)
        self.pool = (nn.MaxPool2d(2) if pool_after_spike else nn.AvgPool2d(2)) \
            if pool else nn.Identity()
        self.pool_after_spike = bool(pool_after_spike and pool)
        self.decay = float(decay)
        self.threshold = float(threshold)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        batch, steps = spikes.shape[:2]
        current = self.bn(self.conv(spikes.flatten(0, 1)))
        if not self.pool_after_spike:
            current = self.pool(current)
        current = current.unflatten(0, (batch, steps))
        spikes = lif_sequence(current, self.decay, self.threshold)
        if self.pool_after_spike:
            spikes = self.pool(spikes.flatten(0, 1)).unflatten(0, (batch, steps))
        return spikes


class ImprovedEventConvSNN(nn.Module):
    """Seven-layer Conv/BN/LIF SNN with gradual pooling and global readout.

    This remains the project's direct convolutional LIF model family. All ten
    temporal steps are processed, neuron state is local to one forward call,
    and classification logits are averaged over time.
    """

    def __init__(self, n_classes: int, channels=(32, 64, 64, 128, 128, 256),
                 decay: float = 0.5, threshold: float = 1.0,
                 readout_size: int = 4, pool_after_spike: bool = False,
                 pools=(True, False, True, False, True, True),
                 feature_dropout: float = 0.0) -> None:
        super().__init__()
        if len(channels) != 6:
            raise ValueError("channels must contain six widths")
        if len(pools) != len(channels):
            raise ValueError("pools and channels must have equal length")
        blocks = []
        incoming = 2
        for outgoing, pool in zip(channels, pools):
            blocks.append(SpikingConvBlock(incoming, outgoing, pool=pool,
                                           decay=decay, threshold=threshold,
                                           pool_after_spike=pool_after_spike))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.readout_size = int(readout_size)
        self.feature_dropout = nn.Dropout(float(feature_dropout))
        self.readout = nn.Linear(incoming * self.readout_size ** 2, int(n_classes))
        self.n_classes = int(n_classes)
        self.channels = tuple(int(c) for c in channels)
        self.decay = float(decay)
        self.threshold = float(threshold)

    def forward(self, frames: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2:] != (2, 128, 128):
            raise ValueError("Expected [batch,time,2,128,128]")
        spikes = frames
        for block in self.blocks:
            spikes = block(spikes)
        features = F.adaptive_avg_pool2d(
            spikes.flatten(0, 1), self.readout_size
        ).flatten(1).unflatten(0, frames.shape[:2])
        logits = self.readout(self.feature_dropout(features))
        return logits if return_sequence else logits.mean(dim=1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)
