from typing import Dict, List, Tuple
import time, numpy as np

# optional deps
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    _TF_OK = True
except Exception as e:
    _TF_OK = False
    _TF_ERR = e

from .vqc_ovr import _train_ovr as train_vqc_ovr  # reuse our QNN after encoding

def _ensure():
    if not _TF_OK:
        raise RuntimeError(f"aec_qnn requires tensorflow. Install: pip install tensorflow-cpu ({_TF_ERR})")

def _build_autoencoder(input_dim: int, encoding_dim: int = 4):
    inp = Input(shape=(input_dim,))
    enc = Dense(encoding_dim, activation="relu")(inp)
    dec = Dense(input_dim, activation="linear")(enc)
    auto = Model(inputs=inp, outputs=dec)
    enc_model = Model(inputs=inp, outputs=enc)
    auto.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return auto, enc_model

def run_aec_qnn_tf(Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    _ensure()
    enc_dim = int(params.get("encoding_dim", min(4, Xtr.shape[1])))
    ae_epochs = int(params.get("ae_epochs", 20))
    batch = int(params.get("batch_size", 32))

    # 1) train autoencoder on Xtr
    auto, encoder = _build_autoencoder(Xtr.shape[1], enc_dim)
    t0 = time.perf_counter()
    auto.fit(Xtr, Xtr, epochs=ae_epochs, batch_size=batch, verbose=0)
    ae_ms = (time.perf_counter() - t0) * 1000.0

    # 2) encode features
    Xtr_z = encoder.predict(Xtr, verbose=0)
    Xte_z = encoder.predict(Xte, verbose=0)

    # 3) train quantum OvR on encoded features
    q_params = {
        "epochs": int(params.get("q_epochs", 50)),
        "lr": float(params.get("q_lr", 0.08)),
        "n_qubits": int(params.get("n_qubits", max(2, min(6, enc_dim)))),
        "layers": int(params.get("layers", 4)),
        "noise_prob": float(params.get("noise_prob", 0.01)),
        "shots": params.get("shots", 0),
    }
    shots = q_params["shots"]; shots = None if shots in (0, None) else int(shots)

    t1 = time.perf_counter()
    predict = train_vqc_ovr(Xtr_z, ytr, n_classes=len(set(ytr)),
                            epochs=q_params["epochs"], lr=q_params["lr"],
                            n_qubits=q_params["n_qubits"], layers=q_params["layers"],
                            noise_p=q_params["noise_prob"], shots=shots)
    proba = predict(Xte_z)
    q_ms = (time.perf_counter() - t1) * 1000.0

    return proba, {"train_ms": ae_ms + q_ms, "infer_ms": 0.0}, {"encoding_dim": enc_dim}
