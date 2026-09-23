import torch.nn as nn

try:
    import pennylane as qml
except Exception:
    qml = None


class MNIST4x4QSNN(nn.Module):
    """Four-qubit data-reuploading QSNN consuming all 16 image latencies."""

    def __init__(self, n_qubits=4, reupload_blocks=4, n_classes=10):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required. Install requirements.txt")
        if n_qubits != 4 or reupload_blocks != 4:
            raise ValueError("The frozen 4x4 architecture requires 4 qubits and 4 blocks.")
        self.n_qubits = n_qubits
        self.reupload_blocks = reupload_blocks
        device = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for block in range(reupload_blocks):
                offset = block * n_qubits
                for wire in range(n_qubits):
                    qml.RY(inputs[..., offset + wire], wires=wire)
                    qml.RY(weights[block, wire, 0], wires=wire)
                    qml.RZ(weights[block, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            local = [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
            correlations = [
                qml.expval(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % n_qubits))
                for wire in range(n_qubits)
            ]
            return local + correlations

        self.qlayer = qml.qnn.TorchLayer(
            circuit, {"weights": (reupload_blocks, n_qubits, 2)}
        )
        self.head = nn.Linear(2 * n_qubits, n_classes)

    def quantum_features(self, theta):
        return self.qlayer(theta)

    def forward(self, theta):
        return self.head(self.quantum_features(theta))

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def circuit_depth(self):
        return self.reupload_blocks * (3 + self.n_qubits)


class MNISTReducedQSNN(nn.Module):
    """Eight-qubit QSNN consuming cyclic slices of reduced image features."""

    def __init__(self, n_qubits=8, reupload_blocks=2, n_classes=10, input_features=16):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required. Install requirements.txt")
        if n_qubits != 8 or reupload_blocks < 2:
            raise ValueError("The reduced architecture requires 8 qubits and at least 2 blocks.")
        if input_features < n_qubits or input_features % n_qubits:
            raise ValueError("Reduced input features must be a positive multiple of 8.")
        self.n_qubits = n_qubits
        self.reupload_blocks = reupload_blocks
        self.input_features = input_features
        device = qml.device("default.qubit", wires=n_qubits, shots=None)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for block in range(reupload_blocks):
                offset = (block * n_qubits) % input_features
                for wire in range(n_qubits):
                    qml.RY(inputs[..., offset + wire], wires=wire)
                    qml.RY(weights[block, wire, 0], wires=wire)
                    qml.RZ(weights[block, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            local = [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
            correlations = [
                qml.expval(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % n_qubits))
                for wire in range(n_qubits)
            ]
            return local + correlations

        self.qlayer = qml.qnn.TorchLayer(
            circuit, {"weights": (reupload_blocks, n_qubits, 2)}
        )
        self.head = nn.Linear(2 * n_qubits, n_classes)

    def forward(self, theta):
        return self.head(self.qlayer(theta))

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def circuit_depth(self):
        return self.reupload_blocks * (3 + self.n_qubits)
