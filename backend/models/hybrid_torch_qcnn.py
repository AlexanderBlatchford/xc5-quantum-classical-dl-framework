from typing import Dict, List, Tuple
import time, math, numpy as np

def run_hybrid_torch_qcnn(Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    try:
        import torch
        from torch import nn
        import pennylane as qml
    except Exception as e:
        raise RuntimeError(f"hybrid_torch requires torch and pennylane. Install: pip install torch pennylane. Original error: {e}")

    torch.manual_seed(42)

    n_qubits = int(params.get("n_qubits", max(2, min(6, Xtr.shape[1]))))
    n_layers = int(params.get("layers", 2))
    epochs = int(params.get("epochs", 15))
    lr = float(params.get("lr", 1e-3))
    batch_size = int(params.get("batch_size", 32))
    n_outputs = len(classes)

    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
    QLayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    class HybridQCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Sequential(nn.Linear(Xtr.shape[1], n_qubits), nn.Tanh())
            self.scale = nn.Parameter(torch.tensor(math.pi), requires_grad=False)
            self.q = QLayer
            self.head = nn.Sequential(nn.Linear(n_qubits, 16), nn.ReLU(), nn.Linear(16, n_outputs))
        def forward(self, x):
            z = self.pre(x) * self.scale
            qexp = self.q(z)
            return self.head(qexp)

    # tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = HybridQCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
    train_ms = (time.perf_counter() - t0) * 1000.0

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t).numpy()
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = proba / proba.sum(axis=1, keepdims=True)

    return proba, {"train_ms": train_ms, "infer_ms": 0.0}, {"n_qubits": n_qubits, "n_layers": n_layers}
