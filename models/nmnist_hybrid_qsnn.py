import math

import torch
from torch import nn
from torch.nn import functional as F

from models.nmnist_snn import lif_step


class NMNISTLIFExtractor(nn.Module):
    """Small shared Conv/LIF encoder producing one bounded feature per qubit."""

    def __init__(self, latent_dim=8, decay=0.5):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.decay = float(decay)
        self.conv1 = nn.Conv2d(2, 12, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool = nn.AvgPool2d(2)
        self.projection = nn.Linear(24 * 2 * 2 * 2, self.latent_dim)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 34, 34):
            raise ValueError("Expected [batch,time,2,34,34] N-MNIST frames.")
        # Do not mutate the caller-owned canonical event frame.  The same
        # frame tensor is retained for serialization/audit after inference.
        frames = frames.float().div(8.0).clamp(max=1.0)
        batch, steps = frames.shape[:2]
        first = self.pool(self.bn1(self.conv1(frames.flatten(0, 1))))
        first = first.unflatten(0, (batch, steps))
        membrane1 = torch.zeros_like(first[:, 0])
        membrane2 = None
        spike_sum = None
        for step in range(steps):
            spikes1, membrane1 = lif_step(first[:, step], membrane1, self.decay)
            current2 = self.pool(self.bn2(self.conv2(spikes1)))
            if membrane2 is None:
                membrane2 = torch.zeros_like(current2)
                spike_sum = torch.zeros_like(current2)
            spikes2, membrane2 = lif_step(current2, membrane2, self.decay)
            spike_sum = spike_sum + spikes2
        spike_rate = F.adaptive_avg_pool2d(spike_sum.div(steps), 2).flatten(1)
        membrane_summary = F.adaptive_avg_pool2d(membrane2, 2).flatten(1)
        return torch.tanh(self.projection(torch.cat((spike_rate, membrane_summary), dim=1)))


class ClassicalLatentClassifier(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.extractor = NMNISTLIFExtractor(latent_dim)
        self.head = nn.Linear(latent_dim, 10)

    def forward(self, frames):
        return self.head(self.extractor(frames))


class NMNISTSpatialLIFExtractor(nn.Module):
    """Conv/LIF extractor retaining configurable second-stage spatial structure."""

    def __init__(self, channels=(12, 24), spatial_size=4, latent_dim=16, decay=0.5):
        super().__init__()
        first_channels, second_channels = channels
        self.decay = float(decay)
        self.spatial_size = int(spatial_size)
        self.latent_dim = int(latent_dim)
        self.conv1 = nn.Conv2d(2, first_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_channels)
        self.conv2 = nn.Conv2d(first_channels, second_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(second_channels)
        self.pool = nn.AvgPool2d(2)
        summary_features = second_channels * self.spatial_size * self.spatial_size * 2
        self.projection = nn.Linear(summary_features, self.latent_dim)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 34, 34):
            raise ValueError("Expected [batch,time,2,34,34] N-MNIST frames.")
        # Preserve the caller-owned canonical frame for audit and serialization.
        frames = frames.float().div(8.0).clamp(max=1.0)
        batch, steps = frames.shape[:2]
        first = self.pool(self.bn1(self.conv1(frames.flatten(0, 1))))
        first = first.unflatten(0, (batch, steps))
        membrane1 = torch.zeros_like(first[:, 0])
        membrane2 = None
        spike_sum = None
        for step in range(steps):
            spikes1, membrane1 = lif_step(first[:, step], membrane1, self.decay)
            current2 = self.pool(self.bn2(self.conv2(spikes1)))
            if membrane2 is None:
                membrane2 = torch.zeros_like(current2)
                spike_sum = torch.zeros_like(current2)
            spikes2, membrane2 = lif_step(current2, membrane2, self.decay)
            spike_sum = spike_sum + spikes2
        spike_rate = F.adaptive_avg_pool2d(spike_sum.div(steps), self.spatial_size).flatten(1)
        final_membrane = F.adaptive_avg_pool2d(membrane2, self.spatial_size).flatten(1)
        return torch.tanh(self.projection(torch.cat((spike_rate, final_membrane), dim=1)))


class SpatialClassicalControl(nn.Module):
    def __init__(self, **extractor_kwargs):
        super().__init__()
        self.extractor = NMNISTSpatialLIFExtractor(**extractor_kwargs)
        self.head = nn.Linear(self.extractor.latent_dim, 10)

    def forward(self, frames):
        return self.head(self.extractor(frames))


class MeaningfulQuantumLayer(nn.Module):
    """Eight-qubit latent re-upload circuit with joint probability readout."""

    def __init__(self, n_qubits=8, n_blocks=3):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_blocks = int(n_blocks)
        if self.n_qubits != 8:
            raise ValueError("The compact latent maps one-to-one onto eight qubits.")
        self.weights = nn.Parameter(torch.empty(n_blocks, n_qubits, 2))
        basis = torch.arange(2 ** n_qubits)
        for wire in range(n_qubits):
            low = basis[(basis & (1 << wire)) == 0]
            self.register_buffer(f"low_{wire}", low, persistent=False)
            control = (basis & (1 << wire)) != 0
            permutation = basis.clone()
            permutation[control] ^= 1 << ((wire + 1) % n_qubits)
            self.register_buffer(f"cnot_{wire}", permutation, persistent=False)
        nn.init.uniform_(self.weights, -0.15, 0.15)

    def _ry(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        low_amplitude, high_amplitude = state[:, low], state[:, high]
        cosine = torch.cos(theta / 2).unsqueeze(1)
        sine = torch.sin(theta / 2).unsqueeze(1)
        return state.index_copy(1, low, cosine * low_amplitude - sine * high_amplitude).index_copy(
            1, high, sine * low_amplitude + cosine * high_amplitude)

    def _rz(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        phase = torch.polar(torch.ones_like(theta), theta / 2).unsqueeze(1)
        return state.index_copy(1, low, state[:, low] * phase.conj()).index_copy(
            1, high, state[:, high] * phase)

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.n_qubits:
            raise ValueError("Expected one compact latent feature per qubit.")
        state = torch.zeros(latent.shape[0], 2 ** self.n_qubits,
                            dtype=torch.complex64, device=latent.device)
        state[:, 0] = 1.0
        for block in range(self.n_blocks):
            for wire in range(self.n_qubits):
                # Every block re-encodes every latent coordinate before variational gates.
                state = self._ry(state, math.pi * latent[:, wire], wire)
                state = self._ry(state, self.weights[block, wire, 0].expand(len(latent)), wire)
                state = self._rz(state, self.weights[block, wire, 1].expand(len(latent)), wire)
            for wire in range(self.n_qubits):
                state = state[:, getattr(self, f"cnot_{wire}")]
        return state.abs().square()


class NMNISTHybridQSNN(nn.Module):
    def __init__(self, latent_dim=8, quantum_blocks=2):
        super().__init__()
        self.extractor = NMNISTLIFExtractor(latent_dim)
        self.quantum = MeaningfulQuantumLayer(latent_dim, quantum_blocks)
        self.head = nn.Linear(2 ** latent_dim, 10)

    def forward(self, frames):
        latent = self.extractor(frames)
        with torch.autocast(device_type=frames.device.type, enabled=False):
            observables = self.quantum(latent.float())
        return self.head(observables)


class QuantumHeadV2(nn.Module):
    """Configurable quantum head for targeted encoding and measurement screens."""

    def __init__(self, n_qubits=8, n_blocks=2, learned_projection=False,
                 two_axis_encoding=False, learned_measurement=False,
                 topology="ring"):
        super().__init__()
        if n_qubits != 8 or n_blocks not in (2, 3, 4):
            raise ValueError("V2 supports eight qubits and two to four blocks.")
        if topology not in ("ring", "alternating"):
            raise ValueError("Topology must be ring or alternating.")
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.learned_projection = learned_projection
        self.two_axis_encoding = two_axis_encoding
        self.learned_measurement = learned_measurement
        self.topology = topology
        self.weights = nn.Parameter(torch.empty(n_blocks, n_qubits, 2))
        if learned_projection:
            self.angle_projection = nn.Linear(n_qubits, n_qubits)
            with torch.no_grad():
                self.angle_projection.weight.copy_(torch.eye(n_qubits))
                self.angle_projection.bias.zero_()
        if two_axis_encoding:
            self.phase_scale = nn.Parameter(torch.zeros(n_blocks, n_qubits))
        if learned_measurement:
            self.measurement_angles = nn.Parameter(torch.zeros(n_qubits, 2))

        basis = torch.arange(2 ** n_qubits)
        for wire in range(n_qubits):
            low = basis[(basis & (1 << wire)) == 0]
            self.register_buffer(f"low_{wire}", low, persistent=False)
        for control in range(n_qubits):
            for target in range(n_qubits):
                if control == target:
                    continue
                active = (basis & (1 << control)) != 0
                permutation = basis.clone()
                permutation[active] ^= 1 << target
                self.register_buffer(f"cnot_{control}_{target}", permutation, persistent=False)
        nn.init.uniform_(self.weights, -0.15, 0.15)

    def _ry(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        low_amplitude, high_amplitude = state[:, low], state[:, high]
        cosine = torch.cos(theta / 2).unsqueeze(1)
        sine = torch.sin(theta / 2).unsqueeze(1)
        return state.index_copy(1, low, cosine * low_amplitude - sine * high_amplitude).index_copy(
            1, high, sine * low_amplitude + cosine * high_amplitude)

    def _rz(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        phase = torch.polar(torch.ones_like(theta), theta / 2).unsqueeze(1)
        return state.index_copy(1, low, state[:, low] * phase.conj()).index_copy(
            1, high, state[:, high] * phase)

    def _entangle(self, state, block):
        if self.topology == "ring":
            pairs = [(wire, (wire + 1) % self.n_qubits) for wire in range(self.n_qubits)]
        else:
            offset = block % 2
            pairs = [(wire, wire + 1) for wire in range(offset, self.n_qubits - 1, 2)]
            pairs += [(wire + 1, wire) for wire in range(1 - offset, self.n_qubits - 1, 2)]
        for control, target in pairs:
            state = state[:, getattr(self, f"cnot_{control}_{target}")]
        return state

    def forward(self, latent):
        angles = self.angle_projection(latent) if self.learned_projection else latent
        angles = math.pi * angles.clamp(-1.0, 1.0)
        state = torch.zeros(latent.shape[0], 2 ** self.n_qubits,
                            dtype=torch.complex64, device=latent.device)
        state[:, 0] = 1.0
        for block in range(self.n_blocks):
            for wire in range(self.n_qubits):
                state = self._ry(state, angles[:, wire], wire)
                if self.two_axis_encoding:
                    state = self._rz(state, self.phase_scale[block, wire] * angles[:, wire], wire)
                state = self._ry(state, self.weights[block, wire, 0].expand(len(latent)), wire)
                state = self._rz(state, self.weights[block, wire, 1].expand(len(latent)), wire)
            state = self._entangle(state, block)
        if self.learned_measurement:
            for wire in range(self.n_qubits):
                state = self._rz(state, self.measurement_angles[wire, 0].expand(len(latent)), wire)
                state = self._ry(state, self.measurement_angles[wire, 1].expand(len(latent)), wire)
        return state.abs().square()


class NMNISTHybridQSNNV2(nn.Module):
    def __init__(self, extractor=None, latent_dim=8, **quantum_kwargs):
        super().__init__()
        self.extractor = extractor if extractor is not None else NMNISTLIFExtractor(latent_dim)
        self.quantum_projection = (nn.Identity() if latent_dim == 8 else nn.Linear(latent_dim, 8))
        self.quantum = QuantumHeadV2(**quantum_kwargs)
        self.head = nn.Linear(256, 10)

    def forward(self, frames):
        latent = torch.tanh(self.quantum_projection(self.extractor(frames)))
        with torch.autocast(device_type=frames.device.type, enabled=False):
            probabilities = self.quantum(latent.float())
        return self.head(probabilities)
