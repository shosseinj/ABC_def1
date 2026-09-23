import torch
import torch.nn as nn

try:
    import pennylane as qml
except Exception:
    qml = None

class IrisQSNN(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4, n_classes=3):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required. Install requirements.txt")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Linear(n_qubits, n_classes)

    def forward(self, theta):
        return self.head(self.quantum_features(theta))

    def quantum_features(self, theta):
        return self.qlayer(theta)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
