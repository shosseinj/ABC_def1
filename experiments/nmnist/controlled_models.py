"""Identical batch-major prediction interface for the four controlled N-MNIST models."""
from __future__ import annotations

import torch
from torch import nn

from experiments.nmnist.reference_models.resnet import ResNet18
from experiments.nmnist.reference_models.simplenet import SimpleNet_v2
from experiments.nmnist.reference_models.vgg import VGGSNN
from models.nmnist_snn import NMNISTConvSNN


class BatchMajorReference(nn.Module):
    def __init__(self, reference: nn.Module):
        super().__init__()
        self.reference = reference

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        logits = self.reference(frames.permute(1, 0, 2, 3, 4).contiguous())
        if logits.shape != (frames.shape[1], frames.shape[0], 10):
            raise RuntimeError(f"reference output shape mismatch: {tuple(logits.shape)}")
        return logits.mean(0)


def build_model(name: str) -> nn.Module:
    if name == "custom":
        return NMNISTConvSNN(0.5, 10)
    classes = {"convnet": SimpleNet_v2, "resnet18": ResNet18, "vggsnn": VGGSNN}
    return BatchMajorReference(classes[name](num_classes=10, img_size=(2, 34, 34)))
