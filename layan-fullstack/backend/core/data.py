from typing import Tuple, Optional, Dict, Any
import io, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _infer_target_column(df: pd.DataFrame, requested: Optional[str]):
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}
    if requested:
        if requested in cols:
            return requested, f"Using target '{requested}'."
        rq = _norm(requested)
        if rq in norm_map:
            return norm_map[rq], f"Target '{requested}' not found; using '{norm_map[rq]}' (normalized match)."
    for syn in ["label", "target", "class", "species", "y"]:
        if syn in norm_map:
            return norm_map[syn], f"Target not provided/found; using '{norm_map[syn]}' (synonym)."
    non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        return non_num[-1], f"Target not provided/found; using last non-numeric column '{non_num[-1]}'."
    n = len(df)
    thresh = max(50, int(0.2 * n))
    for c in cols[::-1]:
        if df[c].nunique() <= thresh:
            return c, f"Target not provided/found; using low-cardinality column '{c}'."
    raise ValueError(f"Could not infer target column. Available columns: {cols}.")

def prepare_data_from_csv(csv_bytes: bytes, target_col_requested: Optional[str]):
    df = pd.read_csv(io.BytesIO(csv_bytes))
    target_used, target_note = _infer_target_column(df, target_col_requested)

    y_raw = df[target_used]
    X_df = df.drop(columns=[target_used])

    combined = pd.concat([X_df, y_raw], axis=1).dropna()
    X_df = combined.drop(columns=[target_used])
    y_raw = combined[target_used]

    drop_names = {c for c in X_df.columns if _norm(c) in {"id", "index"}}
    for c in X_df.select_dtypes(include=["number"]).columns:
        if X_df[c].nunique() == len(X_df):
            drop_names.add(c)
    if drop_names:
        X_df = X_df.drop(columns=list(drop_names))

    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)

    X_df = X_df.select_dtypes(include=["number"]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after cleaning.")

    X = X_df.values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)

    dataset_info = {
        "target": target_used,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "features": list(X_df.columns),
        "classes": [str(c) for c in le.classes_],
        "class_counts": {str(c): int((y_raw == c).sum()) for c in le.classes_},
    }
    return X_tr, X_te, y_tr, y_te, le, scaler, target_note, dataset_info
