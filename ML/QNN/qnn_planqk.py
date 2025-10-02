import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.circuit.library import RawFeatureVector

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler

from planqk.quantum.sdk import PlanqkQuantumProvider
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit.primitives import BackendSamplerV2, Sampler

import math

ITERATIONS = 100

def load_dataset(args):
    if args.data is None:
        # Use Iris dataset (2-class subset for simplicity)
        iris = load_iris(as_frame=True)
        X = iris.data[iris.target < 2]  # only classes 0 and 1
        y = iris.target[iris.target < 2]
        print(f"Using default Iris dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    else:
        df = pd.read_csv(args.data)
        if args.target.isdigit():
            target_col = int(args.target)
            y = df.iloc[:, target_col]
            X = df.drop(df.columns[target_col], axis=1)
        else:
            y = df[args.target]
            X = df.drop(columns=[args.target])

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale features
    X = StandardScaler().fit_transform(X)

    return X, y_enc


def build_feature_map(mode, n_qubits):
    if mode == "angle":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=2)
    elif mode == "amplitude":
        # Dimension must equal 2**n_qubits
        dim = 2 ** n_qubits
        return RawFeatureVector(feature_dimension=dim)
    else:
        raise ValueError("Encoding mode must be 'angle' or 'amplitude'.")



def main(args):
    algorithm_globals.random_seed = 42

    # Load dataset
    X, y_enc = load_dataset(args)

    n_features = X.shape[1]

    if args.encoding == "amplitude":
        # Auto-choose n_qubits if not set
        if args.n_qubits is None:
            n_qubits = math.ceil(math.log2(n_features))
        else:
            n_qubits = args.n_qubits

        max_features = 2**n_qubits

        # Pad or reduce to exactly 2^n_qubits features
        if n_features > max_features:
            pca = PCA(n_components=max_features)
            X = pca.fit_transform(X)
        elif n_features < max_features:
            pad = np.zeros((X.shape[0], max_features - n_features))
            X = np.hstack((X, pad))

    else:  # angle encoding
        n_qubits = args.n_qubits or n_features
        if n_features > n_qubits:
            pca = PCA(n_components=n_qubits)
            X = pca.fit_transform(X)
        elif n_features < n_qubits:
            pad = np.zeros((X.shape[0], n_qubits - n_features))
            X = np.hstack((X, pad))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Encoding
    feature_map = build_feature_map(args.encoding, n_qubits)

    # Ansatz
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=2)

    # Optimizer
    optimizer = COBYLA(maxiter=ITERATIONS)

    # Sampler backend

    # Select a quantum backend suitable for the task. All PLANQK supported quantum backends are
    # listed at https://app.planqk.de/quantum-backends.

    # sampler = Sampler(backend=backend)

    if args.backend is not None:
        print(f"Using PLANQK backend: {args.backend}")
        provider = PlanqkQuantumProvider(access_token="plqk_ktTFQDaIrRcHFLRIuG05zQ6BjDJsNHBmA8k7F1AXm8")
        backend = provider.get_backend("azure.ionq.simulator")

        # Wrap into a Sampler primitive
        sampler = BackendSamplerV2(backend=backend)
    else:
        sampler = Sampler()  # local statevector simulator
        print("Using local Sampler backend.")

    # build the feature_map for both modes
    feature_map = build_feature_map(args.encoding, n_qubits)

    # instantiate the variational classifier
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )


    print("Training VQC...")
    vqc.fit(X_train, y_train)

    # Evaluate
    y_pred = vqc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQC with Qiskit")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV file (default: Iris dataset)")
    parser.add_argument("--target", type=str, default=None, help="Target column (name or index)")
    parser.add_argument("--n_qubits", type=int, default=None, help="Number of qubits (default: features or log2(features))")
    # parser.add_argument("--encoding", type=str, choices=["angle", "amplitude"], default="angle", help="Encoding mode")
    parser.add_argument("--encoding", type=str, choices=["angle", "amplitude"], default="amplitude", help="Encoding mode")
    # parser.add_argument("--backend", type=str, default="Planqk Default", help="Quantum backend name (e.g. azure.ionq.simulator; default: local Sampler)")
    parser.add_argument("--backend", type=str, default=None, help="Quantum backend name (e.g. azure.ionq.simulator; default: local Sampler)")
    args = parser.parse_args()

    main(args)
