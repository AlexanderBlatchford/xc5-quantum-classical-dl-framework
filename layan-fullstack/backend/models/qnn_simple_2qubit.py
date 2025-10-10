from typing import Dict, List, Tuple
import time, numpy as np
import pennylane as qml
from pennylane import numpy as pnp

def _dev(): return qml.device("default.qubit", wires=2)

def _encode(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)

def _variational_circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RX(params[3], wires=1)

def _make_qnode():
    dev = _dev()
    @qml.qnode(dev, interface="autograd")
    def qnode(x, params):
        _encode(x)
        _variational_circuit(params)
        return qml.expval(qml.PauliZ(0))
    return qnode

def run_qnn_simple(Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    # Use only first 2 features (model is 2-qubit)
    Xtr2 = Xtr[:, :2].copy()
    Xte2 = Xte[:, :2].copy()

    qnode = _make_qnode()
    rng = np.random.default_rng(7)

    def to_margin(w, X): return pnp.array([qnode(x, w) for x in X])
    def loss_mse(w, X, ypm): return pnp.mean((to_margin(w, X) - ypm)**2)

    t0 = time.perf_counter()
    heads = {}
    for c in sorted(set(ytr)):
        ypm = pnp.array(np.where(ytr == c, +1, -1))
        w = pnp.array(rng.random(4), requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=float(params.get("lr", 0.1)))
        for _ in range(int(params.get("epochs", 25))):
            w, _ = opt.step_and_cost(lambda v: loss_mse(v, Xtr2, ypm), w)
        heads[c] = w

    scores = []
    for c in sorted(heads.keys()):
        f = np.array([qnode(x, heads[c]) for x in Xte2], dtype=float).reshape(-1,1)
        scores.append(f)
    S = np.hstack(scores)
    eS = np.exp(S)
    proba = eS / eS.sum(axis=1, keepdims=True)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return proba, {"train_ms": total_ms, "infer_ms": 0.0}, {"used_features": 2}
