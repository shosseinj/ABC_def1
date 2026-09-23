import torch
from torch import nn

from models.nmnist_snn import lif_step


class SHDRecurrentSNN(nn.Module):
    """Compact recurrent LIF classifier for binned SHD spike trains."""

    def __init__(self, input_channels=700, hidden_size=128, n_classes=20, decay=0.9):
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_size = int(hidden_size)
        self.n_classes = int(n_classes)
        self.decay = float(decay)
        self.input_layer = nn.Linear(self.input_channels, self.hidden_size, bias=True)
        self.recurrent_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.readout = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, spikes):
        if spikes.ndim != 3 or spikes.shape[2] != self.input_channels:
            raise ValueError("Expected SHD input shaped [batch, time, 700].")
        batch_size, steps = spikes.shape[:2]
        input_current = self.input_layer(spikes)
        membrane = spikes.new_zeros((batch_size, self.hidden_size))
        hidden_spikes = spikes.new_zeros((batch_size, self.hidden_size))
        logits = spikes.new_zeros((batch_size, self.n_classes))
        for step in range(steps):
            current = input_current[:, step] + self.recurrent_layer(hidden_spikes)
            hidden_spikes, membrane = lif_step(current, membrane, self.decay)
            logits = logits + self.readout(hidden_spikes)
        return logits / steps

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
