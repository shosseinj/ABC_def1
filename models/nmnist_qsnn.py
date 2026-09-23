import math

import torch
import torch.nn as nn

try:
    import pennylane as qml
except Exception:
    qml = None


class NMNISTTemporalQSNN(nn.Module):
    """Eight-qubit QSNN that sequentially reuploads ordered event-bin channels."""

    def __init__(self, n_qubits=8, temporal_steps=8, n_classes=10):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required. Install requirements.txt")
        if n_qubits != 8 or temporal_steps != 8:
            raise ValueError("The frozen N-MNIST baseline requires 8 qubits and 8 temporal steps.")
        self.n_qubits = n_qubits
        self.temporal_steps = temporal_steps
        device = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for step in range(temporal_steps):
                for wire in range(n_qubits):
                    qml.RY(
                        3.141592653589793 * inputs[..., step * n_qubits + wire], wires=wire
                    )
                    qml.RY(weights[step, wire, 0], wires=wire)
                    qml.RZ(weights[step, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            local = [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
            correlations = [
                qml.expval(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % n_qubits))
                for wire in range(n_qubits)
            ]
            return local + correlations

        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": (temporal_steps, n_qubits, 2)})
        self.head = nn.Linear(2 * n_qubits, n_classes)

    def forward(self, event_channels):
        flattened = event_channels.reshape(*event_channels.shape[:-2], -1)
        return self.head(self.qlayer(flattened))

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def circuit_depth(self):
        return self.temporal_steps * (3 + self.n_qubits)


class NMNISTCudaQSNN(nn.Module):
    """Batched eight-qubit state-vector QSNN implemented with CUDA Torch tensors."""

    def __init__(self, n_qubits=8, n_blocks=4, n_classes=10, temporal_bins=4,
                 spatial_polarity_channels=8):
        super().__init__()
        if n_qubits not in (8, 12) or n_blocks not in (4, 6):
            raise ValueError("Supported controlled-ablation capacities are 8/12 qubits and 4/6 blocks.")
        if temporal_bins < 1 or spatial_polarity_channels not in (8, 16):
            raise ValueError("Invalid controlled-ablation input dimensions.")
        self.n_qubits = n_qubits
        self.n_blocks = n_blocks
        self.temporal_bins = temporal_bins
        self.spatial_polarity_channels = spatial_polarity_channels
        self.weights = nn.Parameter(torch.empty(n_blocks, n_qubits, 2))
        self.head = nn.Linear(n_qubits, n_classes)

        basis = torch.arange(2 ** n_qubits)
        for wire in range(n_qubits):
            low = basis[(basis & (1 << wire)) == 0]
            self.register_buffer(f"low_{wire}", low, persistent=False)
            control = (basis & (1 << wire)) != 0
            permutation = basis.clone()
            permutation[control] ^= 1 << ((wire + 1) % n_qubits)
            self.register_buffer(f"cnot_{wire}", permutation, persistent=False)
        z_signs = torch.stack([
            1.0 - 2.0 * ((basis >> wire) & 1).float() for wire in range(n_qubits)
        ])
        self.register_buffer("z_signs", z_signs, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weights, -0.1, 0.1)
        self.head.reset_parameters()

    def _ry(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        low_amplitude = state[:, low]
        high_amplitude = state[:, high]
        cosine = torch.cos(theta / 2).unsqueeze(1)
        sine = torch.sin(theta / 2).unsqueeze(1)
        updated = state.clone()
        updated[:, low] = cosine * low_amplitude - sine * high_amplitude
        updated[:, high] = sine * low_amplitude + cosine * high_amplitude
        return updated

    def _rz(self, state, theta, wire):
        low = getattr(self, f"low_{wire}")
        high = low | (1 << wire)
        phase = torch.polar(torch.ones_like(theta), theta / 2).unsqueeze(1)
        updated = state.clone()
        updated[:, low] = state[:, low] * phase.conj()
        updated[:, high] = state[:, high] * phase
        return updated

    def quantum_features(self, event_channels):
        expected = (self.temporal_bins, self.spatial_polarity_channels)
        if event_channels.ndim != 3 or event_channels.shape[1:] != expected:
            raise ValueError(f"Expected input shape [batch, {expected[0]}, {expected[1]}].")
        state = torch.zeros(
            event_channels.shape[0], 2 ** self.n_qubits,
            dtype=torch.complex64, device=event_channels.device,
        )
        state[:, 0] = 1.0
        for block in range(self.n_blocks):
            if self.temporal_bins >= self.n_blocks:
                temporal_indices = range(
                    math.ceil(block * self.temporal_bins / self.n_blocks),
                    math.ceil((block + 1) * self.temporal_bins / self.n_blocks),
                )
            else:
                temporal_indices = (block,) if block < self.temporal_bins else ()
            for temporal_bin in temporal_indices:
                for channel in range(self.spatial_polarity_channels):
                    state = self._ry(
                        state, math.pi * event_channels[:, temporal_bin, channel],
                        channel % self.n_qubits,
                    )
            for wire in range(self.n_qubits):
                state = self._ry(state, self.weights[block, wire, 0].expand(event_channels.shape[0]), wire)
                state = self._rz(state, self.weights[block, wire, 1].expand(event_channels.shape[0]), wire)
            for wire in range(self.n_qubits):
                state = state[:, getattr(self, f"cnot_{wire}")]
        probabilities = state.abs().square()
        return probabilities @ self.z_signs.T

    def forward(self, event_channels):
        return self.head(self.quantum_features(event_channels))

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
