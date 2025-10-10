from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    confusion_matrix, precision_recall_fscore_support
)

def metrics_from_probs(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    try:
        if proba.shape[1] == 2:
            auc = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    try:
        ll = float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))
    except Exception:
        ll = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc, "loss": ll}

def details_from_preds(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: List[str],
    timings: Optional[Dict[str, float]] = None,
    extras: Optional[Dict] = None,
):
    y_pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), zero_division=0
    )
    per_class = []
    for i, cls in enumerate(classes):
        per_class.append({
            "class": str(cls),
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(supp[i]),
        })
    out = {"confusion": cm.tolist(), "per_class": per_class}
    if timings: out["timings"] = timings
    if extras:  out["extras"] = extras
    return out
