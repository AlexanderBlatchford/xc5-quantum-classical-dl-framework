# ---- Code Cell ----
# qnn_iter_qubits.py
# Train a Variational Quantum Classifier (QNN) on Iris with 2, 4, and 6 qubits
# Uses data re-uploading and one-vs-rest training.

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Config
# =========================
SEED = 7
rng = np.random.default_rng(SEED)

EPOCHS = 200
BATCH_SIZE = 32
LR = 0.08
K_BLOCKS = 6
NOISE_P = 0.02
SHOTS = None   # set 4096 for hardware-like behaviour

# =========================
# Load dataset (Iris)
# =========================
iris = load_iris(as_frame=True)
X = iris.data.to_numpy(dtype=float)
y = iris.target.to_numpy()
class_names = iris.target_names

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

# =========================
# QNN definition (parametric)
# =========================
def make_qnn(N_QUBITS, K_BLOCKS, noise_p=NOISE_P, shots=SHOTS):
    dev = qml.device("default.mixed", wires=N_QUBITS, shots=shots)
    W = rng.normal(size=(N_QUBITS, X_tr.shape[1]))

    def embed_block(x):
        a = W @ x
        a = np.clip(a, -5, 5) * (np.pi / 5)
        for q in range(N_QUBITS):
            qml.RY(a[q], wires=q)

    def var_block(theta):
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q+1])
        for q in range(N_QUBITS):
            qml.Rot(theta[q,0], theta[q,1], theta[q,2], wires=q)

    def noise_block(p=noise_p):
        if p and p > 0.0:
            for q in range(N_QUBITS):
                qml.DepolarizingChannel(p, wires=q)

    @qml.qnode(dev, interface="autograd")
    def qnn_margin(x, thetas, p_noise=noise_p):
        for k in range(K_BLOCKS):
            embed_block(x)
            var_block(thetas[k])
            noise_block(p_noise)
        return qml.expval(qml.PauliZ(0))

    def init_weights():
        return pnp.array(
            rng.normal(scale=0.15, size=(K_BLOCKS, N_QUBITS, 3)),
            requires_grad=True
        )

    return qnn_margin, init_weights, W


# =========================
# Training helpers
# =========================
def to_margins(qnn_margin, weights, X):
    return pnp.array([qnn_margin(x, weights) for x in X])

def loss_mse(qnn_margin, weights, X, y_pm):
    return pnp.mean((to_margins(qnn_margin, weights, X) - y_pm)**2)

def binary_accuracy(qnn_margin, weights, X, y_pm):
    yhat = pnp.where(to_margins(qnn_margin, weights, X) >= 0, 1, -1)
    return float(pnp.mean(yhat == y_pm))

def train_one_vs_rest(qnn_margin, init_weights, X_tr, y_tr, pos_class, lr=LR, epochs=EPOCHS, batch=BATCH_SIZE):
    y_pm = pnp.array(np.where(y_tr == pos_class, +1, -1))
    weights = init_weights()
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    n = len(X_tr)
    for ep in range(1, epochs+1):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        weights, cur_loss = opt.step_and_cost(lambda w: loss_mse(qnn_margin, w, X_tr[idx], y_pm[idx]), weights)
        if ep % 50 == 0:
            acc = binary_accuracy(qnn_margin, weights, X_tr, y_pm)
            print(f"[class {pos_class}] epoch {ep} | loss {cur_loss:.3f} | acc {acc:.3f}")
    return weights

for N_QUBITS in [2]:
    print("\n=============================")
    print(f" Training QNN with {N_QUBITS} qubits ")
    print("=============================")

    # build QNN circuit
    qnn_margin, init_weights = make_qnn(N_QUBITS, K_BLOCKS)

    # train OVR heads
    heads = {}
    for c in np.unique(y_tr):
        print(f"\n--- Training head {c} vs rest ---")
        heads[int(c)] = train_one_vs_rest(qnn_margin, init_weights, X_tr, y_tr, pos_class=int(c))

    # OVR prediction function
    def ovr_predict(X):
        scores = []
        for c in sorted(heads.keys()):
            f = np.array([qnn_margin(x, heads[c]) for x in X], dtype=float)
            scores.append(f.reshape(-1,1))
        S = np.hstack(scores)  # shape = (n_samples, n_classes)
        return np.argmax(S, axis=1), S

    # evaluate
    yhat_tr, _ = ovr_predict(X_tr)
    yhat_te, _ = ovr_predict(X_te)

    print("\nResults:")
    print("Train acc:", accuracy_score(y_tr, yhat_tr))
    print("Test  acc:", accuracy_score(y_te, yhat_te))
    print("\nConfusion matrix (test):\n", confusion_matrix(y_te, yhat_te))
    print("\nReport (test):\n", classification_report(y_te, yhat_te, target_names=class_names))




