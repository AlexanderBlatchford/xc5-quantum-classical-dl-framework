from typing import Callable, Dict, List, Tuple, Any
import numpy as np

Runner = Callable[[np.ndarray, np.ndarray, np.ndarray, dict, List[str]], Tuple[np.ndarray, dict, dict]]

# Always-available classical models (scikit-learn)
from models.classical_sklearn import run_classical

# Always-available quantum models (PennyLane)
from models.vqc_ovr import run_vqc_ovr
from models.qnn_simple_2qubit import run_qnn_simple

def _lazy_mlp_torch() -> Runner:
    def runner(Xtr, ytr, Xte, params, classes):
        from models.mlp_torch import run_mlp_torch
        return run_mlp_torch(Xtr, ytr, Xte, params, classes)
    return runner

def _lazy_hybrid_torch() -> Runner:
    def runner(Xtr, ytr, Xte, params, classes):
        from models.hybrid_torch_qcnn import run_hybrid_torch_qcnn
        return run_hybrid_torch_qcnn(Xtr, ytr, Xte, params, classes)
    return runner

def _lazy_aec_qnn_tf() -> Runner:
    def runner(Xtr, ytr, Xte, params, classes):
        from models.aec_qnn_tf import run_aec_qnn_tf
        return run_aec_qnn_tf(Xtr, ytr, Xte, params, classes)
    return runner

_CLASSICAL: Dict[str, Runner] = {
    "mlp":      lambda Xtr, ytr, Xte, p, classes: run_classical("mlp", Xtr, ytr, Xte, p, classes),
    "svm":      lambda Xtr, ytr, Xte, p, classes: run_classical("svm", Xtr, ytr, Xte, p, classes),
    "rf":       lambda Xtr, ytr, Xte, p, classes: run_classical("rf",  Xtr, ytr, Xte, p, classes),
    "logreg":   lambda Xtr, ytr, Xte, p, classes: run_classical("logreg", Xtr, ytr, Xte, p, classes),
    "mlp_torch": _lazy_mlp_torch(),
}

_QUANTUM: Dict[str, Runner] = {
    "qnn":        run_vqc_ovr,
    "vqc":        run_vqc_ovr,       # alias
    "qnn_simple": run_qnn_simple,
    "hybrid_torch": _lazy_hybrid_torch(),
    "aec_qnn":      _lazy_aec_qnn_tf(),
}

def get_classical_runner(key: str) -> Runner:
    if key not in _CLASSICAL:
        raise ValueError(f"Unknown classical model '{key}'. Available: {list(_CLASSICAL)}")
    return _CLASSICAL[key]

def get_quantum_runner(key: str) -> Runner:
    if key not in _QUANTUM:
        raise ValueError(f"Unknown quantum model '{key}'. Available: {list(_QUANTUM)}")
    return _QUANTUM[key]
