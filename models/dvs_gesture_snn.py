import torch
from torch import nn

from models.cifar10_dvs_snn import lif_step


class DVSGestureConvSNN(nn.Module):
    """Repository-native VGGSNN-style LIF network for T=10 DVS-Gesture frames."""
    def __init__(self, decay=0.5, n_classes=11, spatial_size=64):
        super().__init__()
        self.decay, self.spatial_size = float(decay), int(spatial_size)
        channels = (32, 64, 128, 128)
        self.blocks = nn.ModuleList()
        incoming = 2
        for outgoing in channels:
            self.blocks.append(nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                                             nn.BatchNorm2d(outgoing), nn.AvgPool2d(2)))
            incoming = outgoing
        self.readout = nn.Linear(channels[-1], n_classes)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, self.spatial_size, self.spatial_size):
            raise ValueError("Expected [batch,time,2,spatial_size,spatial_size]")
        membranes = [None] * len(self.blocks)
        output = frames.new_zeros((frames.shape[0], self.readout.out_features))
        for time in range(frames.shape[1]):
            spikes = frames[:, time]
            for index, block in enumerate(self.blocks):
                current = block(spikes)
                if membranes[index] is None:
                    membranes[index] = torch.zeros_like(current)
                spikes, membranes[index] = lif_step(current, membranes[index], self.decay)
            output += self.readout(spikes.mean(dim=(-2, -1)))
        return output / frames.shape[1]

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
