from typing import Dict, List, Tuple, Optional
import time, numpy as np
import pennylane as qml
import pennylane.numpy as pnp

def _build_qnn(n_qubits: int, n_features: int, layers: int, noise_p: float, shots: Optional[int]):
    dev = qml.device("default.mixed", wires=n_qubits, shots=shots)
    rng = np.random.default_rng(7)
    W = rng.normal(size=(n_qubits, n_features))

    def embed_block(x):
        a = W @ x
        a = np.clip(a, -5, 5) * (np.pi / 5)
        for q in range(n_qubits):
            qml.RY(a[q], wires=q)
    def var_block(theta):
        for q in range(n_qubits-1):
            qml.CNOT(wires=[q, q+1])
        for q in range(n_qubits):
            qml.Rot(theta[q,0], theta[q,1], theta[q,2], wires=q)
    def noise_block(p=noise_p):
        if p and p>0: 
            for q in range(n_qubits): qml.DepolarizingChannel(p, wires=q)

    @qml.qnode(dev, interface="autograd")
    def qnn_margin(x, thetas, p_noise=noise_p):
        for _ in range(layers):
            embed_block(x)
            var_block(thetas[_])
            noise_block(p_noise)
        return qml.expval(qml.PauliZ(0))

    def init_weights(rng=np.random.default_rng(7)):
        return pnp.array(rng.normal(scale=0.15, size=(layers, n_qubits, 3)), requires_grad=True)
    return qnn_margin, init_weights

def _train_ovr(Xtr, ytr, n_classes, epochs, lr, n_qubits, layers, noise_p, shots):
    qnn_margin, init_weights = _build_qnn(n_qubits, Xtr.shape[1], layers, noise_p, shots)
    rng = np.random.default_rng(7)

    def to_margins(weights, X): return pnp.array([qnn_margin(x, weights) for x in X])
    def loss_mse(weights, X, y_pm): return pnp.mean((to_margins(weights, X) - y_pm)**2)

    heads = {}
    for c in range(n_classes):
        y_pm = pnp.array(np.where(ytr == c, +1, -1))
        weights = init_weights()
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        n = len(Xtr)
        for _ in range(epochs):
            idx = rng.choice(n, size=min(32, n), replace=False)
            weights, _ = opt.step_and_cost(lambda w: loss_mse(w, Xtr[idx], y_pm[idx]), weights)
        heads[c] = weights

    def predict(Xte):
        scores = []
        for c in range(n_classes):
            f = np.array([qnn_margin(x, heads[c]) for x in Xte], dtype=float).reshape(-1,1)
            scores.append(f)
        S = np.hstack(scores)
        eS = np.exp(S)  # softmax temperature=1
        return eS / eS.sum(axis=1, keepdims=True)
    return predict

def run_vqc_ovr(Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    shots = params.get("shots", 0); shots = None if shots in (0, None) else int(shots)
    noise_p = float(params.get("noise_prob", 0.01))
    layers = int(params.get("layers", 4))
    epochs = int(params.get("epochs", 50))
    lr = float(params.get("lr", 0.08))
    n_qubits = int(params.get("n_qubits", 2))

    t0 = time.perf_counter()
    predict = _train_ovr(Xtr, ytr, n_classes=len(set(ytr)), epochs=epochs, lr=lr,
                         n_qubits=n_qubits, layers=layers, noise_p=noise_p, shots=shots)
    proba = predict(Xte)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return proba, {"train_ms": total_ms, "infer_ms": 0.0}, {}
