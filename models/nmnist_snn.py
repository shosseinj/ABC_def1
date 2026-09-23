import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + 5.0 * x.abs()).pow(2)


def lif_step(current, membrane, decay, threshold=1.0):
    membrane = membrane * decay + current
    spike = SurrogateSpike.apply(membrane - threshold)
    return spike, membrane - spike.detach() * threshold


class NMNISTConvSNN(nn.Module):
    """Compact convolutional LIF network for ordered N-MNIST event frames."""

    def __init__(self, decay=0.5, n_classes=10):
        super().__init__()
        self.decay = float(decay)
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(32 * 8 * 8, n_classes)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 34, 34):
            raise ValueError("Expected N-MNIST input shaped [batch, time, 2, 34, 34].")
        batch_size, steps = frames.shape[:2]
        encoded1 = self.pool(self.bn1(self.conv1(frames.flatten(0, 1))))
        encoded1 = encoded1.unflatten(0, (batch_size, steps))
        membrane1 = torch.zeros_like(encoded1[:, 0])
        membrane2 = None
        output_membrane = frames.new_zeros((batch_size, self.classifier.out_features))
        for step in range(steps):
            spikes1, membrane1 = lif_step(encoded1[:, step], membrane1, self.decay)
            current2 = self.pool(self.bn2(self.conv2(spikes1)))
            if membrane2 is None:
                membrane2 = torch.zeros_like(current2)
            spikes2, membrane2 = lif_step(current2, membrane2, self.decay)
            output_membrane = output_membrane + self.classifier(spikes2.flatten(1))
        return output_membrane / steps

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
