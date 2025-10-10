from typing import Dict, List, Tuple
import time, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def _fit_predict(clf, Xtr, ytr, Xte):
    t0 = time.perf_counter(); clf.fit(Xtr, ytr); train_ms = (time.perf_counter()-t0)*1000.0
    t1 = time.perf_counter(); proba = clf.predict_proba(Xte); infer_ms = (time.perf_counter()-t1)*1000.0
    return proba, {"train_ms": train_ms, "infer_ms": infer_ms}, {}

def run_classical(key: str, Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    if key == "mlp":
        epochs = int(params.get("epochs", 50)); lr = float(params.get("lr", 0.003)); batch = int(params.get("batch_size", 32))
        clf = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                            learning_rate_init=lr, batch_size=batch, max_iter=epochs,
                            random_state=7, n_iter_no_change=epochs+5, verbose=False)
    elif key == "svm":
        C = float(params.get("C", 1.0)); gamma = params.get("gamma", "scale")
        clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=7)
    elif key == "rf":
        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", None); max_depth = None if max_depth in (None, "", "null") else int(max_depth)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=7)
    elif key == "logreg":
        C = float(params.get("C", 1.0)); clf = LogisticRegression(max_iter=200, C=C, n_jobs=None)
    else:
        raise ValueError(f"unknown classical key {key}")
    return _fit_predict(clf, Xtr, ytr, Xte)
