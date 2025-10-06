import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse

# ---- Quantum Device ----
dev = qml.device("default.qubit", wires=2)

# ---- Feature Encoding ----
def encode(x):
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)

# ---- Variational Circuit ----
def variational_circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RX(params[3], wires=1)

# ---- QNode Definition ----
@qml.qnode(dev)
def quantum_model(x, params):
    encode(x)
    variational_circuit(params)
    return qml.expval(qml.PauliZ(0))

# ---- Prediction Wrapper ----
def predict(x, params):
    return np.array([quantum_model(sample, params) for sample in x])

# ---- Cost Function ----
def cost(params, x, y):
    preds = predict(x, params)
    return np.mean((preds - y) ** 2)

# ---- Training Function ----
def train_qnn(data_path, target_col, epochs=25):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array([1 if val > 0.5 else 0 for val in y])  # ensure binary

    # Reduce features to 2 (for 2 qubits)
    X = X[:, :2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = np.random.random(4)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    for epoch in range(epochs):
        params, loss = opt.step_and_cost(lambda v: cost(v, X_train, y_train), params)
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    preds = predict(X_test, params)
    acc = np.mean((preds > 0) == y_test)
    print(f"\n✅ Training Complete — Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    train_qnn(args.data, args.target)
