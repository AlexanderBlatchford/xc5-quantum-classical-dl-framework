# ---- Code Cell ----
# 🔧 Auto-install required libraries (fixed: use scikit-learn, not sklearn)
import sys, importlib

def need(pkg):
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} already installed")
        return False
    except ImportError:
        return True

def pip_install(*pkgs):
    print("📦 Installing:", " ".join(pkgs))
    !{sys.executable} -m pip install -U { " ".join(pkgs) }

# Core stack
if need("numpy"): pip_install("numpy>=1.26")
if need("pandas"): pip_install("pandas>=2.1")
if need("matplotlib"): pip_install("matplotlib")
if need("seaborn"): pip_install("seaborn")
# ✅ correct package name:
if need("sklearn"): pip_install("scikit-learn")   # not "sklearn"
# PyTorch CPU build (safe everywhere). Comment out if you already installed CUDA build via conda.
if need("torch"): pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

print("🎉 All required libraries are ready!")


# ---- Code Cell ----
# 📦 0) Setup & Imports
import os, json, random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

# ---- Code Cell ----
# 🧹 1) Normalizer class
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        return Normalizer(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    @staticmethod
    def load(path: str) -> "Normalizer":
        data = np.load(path)
        return Normalizer(mean=data["mean"], std=data["std"]) 


# ---- Code Cell ----
# 📥 2) Data helpers (CSV + synthetic) & split

def load_from_csv(path: str, target_col, drop_na: bool = True):
    """
    Load CSV and return (X, y, class_names).
    
    - target_col: column NAME (str) or index (int; supports negatives).
    - Features: all other columns.
    - If features include non-numeric (like 'Species'), they are one-hot encoded.
    - class_names is only used when the target is categorical (strings).
    """
    df = pd.read_csv(path)
    if drop_na:
        df = df.dropna()

    # --- Separate target from features ---
    if isinstance(target_col, int):
        if target_col < 0:
            target_col = df.shape[1] + target_col
        y = df.iloc[:, target_col].to_numpy()
        feat_df = df.drop(df.columns[target_col], axis=1)
    else:
        y = df[target_col].to_numpy()
        feat_df = df.drop(columns=[target_col])

    # --- One-hot encode any non-numeric feature columns ---
    feat_df = pd.get_dummies(feat_df, drop_first=False, dtype=np.float32)

    # Convert features to numpy
    X = feat_df.to_numpy(dtype=np.float32)

    # --- Class names only relevant if target is categorical (string) ---
    class_names = None
    if pd.Series(y).dtype == object:
        class_names = sorted(list(pd.unique(y)))

    return X, y, class_names


def make_synthetic(kind: str, n: int = 800, d: int = 8, num_classes: int = 3, seed: int = SEED):
    """
    Simple synthetic datasets:
      - 'binary'      -> 2-class linear-ish
      - 'multiclass'  -> k-means-like clusters
      - 'regression'  -> linear with noise
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)

    if kind == "binary":
        w = rng.randn(d)
        logits = X @ w
        y = (logits + 0.5 * rng.randn(n) > 0).astype(int)
    elif kind == "multiclass":
        centers = rng.randn(num_classes, d)
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centers], axis=1)
        y = dists.argmin(axis=1)
    elif kind == "regression":
        w = rng.randn(d)
        y = X @ w + 0.5 * rng.randn(n)
    else:
        raise ValueError("kind must be 'binary' | 'multiclass' | 'regression'")
    return X, y


def train_val_test_split(x: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = SEED):
    """Random split into train/val/test."""
    n = len(x)
    idx = np.random.RandomState(seed).permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx]), (x[test_idx], y[test_idx])


# ✅ Robust task/type inference (handles string or numeric targets cleanly)
def infer_task_type(y: np.ndarray, max_classes: int = 100):
    """
    Decide if task is classification or regression and return:
      (task, y_processed, class_names_or_None)

    Rules:
      - String/object labels -> classification (factorized to ints)
      - Small integer set (2..max_classes) -> classification
      - Otherwise -> regression
    """
    y_flat = np.asarray(y).reshape(-1)

    # String/object labels => classification
    if y_flat.dtype.kind in {"U", "S", "O"}:
        codes, uniques = pd.factorize(y_flat, sort=True)
        class_names = [str(u) for u in uniques]
        return "classification", codes.astype(np.int64), class_names

    # Numeric: check if labels look like small set of integers
    uniques = np.unique(y_flat)
    try:
        int_uniques = uniques.astype(int)
        is_int_like = np.allclose(uniques, int_uniques)
    except Exception:
        is_int_like = False

    if is_int_like and 2 <= len(uniques) <= max_classes:
        mapping = {int(v): i for i, v in enumerate(sorted(int_uniques))}
        y_mapped = np.vectorize(lambda v: mapping[int(round(v))])(y_flat.astype(float))
        class_names = [str(k) for k in sorted(mapping.keys())]
        return "classification", y_mapped.astype(np.int64), class_names

    # Fallback: regression
    return "regression", y_flat.astype(np.float32), None


# ---- Code Cell ----
# 🧠 3) Model: MLP
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], num_outputs: int, dropout: float, task: str):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))
        self.net = nn.Sequential(*layers)
        self.task = task

    def forward(self, x):
        return self.net(x)


# ---- Code Cell ----
# 🏃 4) Train/Eval loops
def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    return x if std <= 0 else x + torch.randn_like(x) * std


def compute_metrics(task: str, logits: torch.Tensor, y: torch.Tensor) -> dict:
    if task == "classification":
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        return {"accuracy": acc}
    else:
        mse = nn.functional.mse_loss(logits.squeeze(), y.float()).item()
        mae = nn.functional.l1_loss(logits.squeeze(), y.float()).item()
        return {"mse": mse, "mae": mae}


def train_one_epoch(model, loader, optimizer, device, task, noise_std=0.0):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = add_noise(xb, noise_std)
        optimizer.zero_grad()
        out = model(xb)
        loss = nn.CrossEntropyLoss()(out, yb.long()) if task == "classification" else nn.MSELoss()(out.squeeze(), yb.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, task):
    model.eval()
    total_loss = 0.0
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = nn.CrossEntropyLoss()(out, yb.long()) if task == "classification" else nn.MSELoss()(out.squeeze(), yb.float())
            total_loss += loss.item() * xb.size(0)
            all_logits.append(out.cpu())
            all_y.append(yb.cpu())
    logits = torch.cat(all_logits)
    ys = torch.cat(all_y)
    metrics = compute_metrics(task, logits, ys)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---- Code Cell ----
# ⚙️ 5) Experiment Config
CONFIG = {
    "data_source": "csv",    # "synthetic" | "csv"
    "synthetic_kind": "binary",    # "binary" | "multiclass" | "regression"
    "csv_path": "./iris.csv",
    "target_col": "Species",
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "hidden": [64, 64],
    "dropout": 0.0,
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    "noise_std": 0.0,
    "patience": 5,
    "output_dir": "runs/mlp_experiment_notebook",
}
CONFIG


# ---- Code Cell ----
# ▶️ 6) Build Dataset, Dataloaders, Model
os.makedirs(CONFIG["output_dir"], exist_ok=True)

if CONFIG["data_source"] == "synthetic":
    X, y = make_synthetic(CONFIG["synthetic_kind"])
    class_names = None
elif CONFIG["data_source"] == "csv":
    X, y, class_names = load_from_csv(CONFIG["csv_path"], CONFIG["target_col"])
else:
    raise ValueError("data_source must be 'synthetic' or 'csv'")

TASK, y_proc, _auto_names = infer_task_type(y)
if TASK == "classification":
    y_np = y_proc.astype(np.int64)
    num_outputs = int(np.max(y_np) + 1)
    if class_names is None:
        class_names = [str(i) for i in range(num_outputs)]
else:
    y_np = y_proc.astype(np.float32)
    num_outputs = 1

(Xtr, ytr), (Xval, yval), (Xte, yte) = train_val_test_split(
    X, y_np, CONFIG["val_ratio"], CONFIG["test_ratio"], SEED
)

normalizer = Normalizer.fit(Xtr)
Xtr = normalizer.transform(Xtr)
Xval = normalizer.transform(Xval)
Xte  = normalizer.transform(Xte)

Xtr_t = torch.from_numpy(Xtr).float()
Xval_t = torch.from_numpy(Xval).float()
Xte_t  = torch.from_numpy(Xte ).float()
ytr_t  = torch.from_numpy(ytr)
yval_t = torch.from_numpy(yval)
yte_t  = torch.from_numpy(yte )

train_ds = TensorDataset(Xtr_t, ytr_t)
val_ds   = TensorDataset(Xval_t, yval_t)
test_ds  = TensorDataset(Xte_t, yte_t)

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
test_loader  = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

model = MLP(
    input_dim=Xtr.shape[1],
    hidden=CONFIG["hidden"],
    num_outputs=num_outputs,
    dropout=CONFIG["dropout"],
    task=TASK,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

# ---- Code Cell ----
# 📈 Plot Training History (loss + optional accuracy)

import os, json
import matplotlib.pyplot as plt

out_dir = CONFIG["output_dir"]
metrics_path = os.path.join(out_dir, "metrics.json")

# Load history from saved metrics.json
with open(metrics_path, "r") as f:
    m = json.load(f)

hist = m.get("history", {})

# --- Plot Loss ---
plt.figure(figsize=(6,4))
plt.plot(hist.get("train_loss", []), label="train_loss")
plt.plot(hist.get("val_loss", []), label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Accuracy (if classification & recorded) ---
if "val_acc" in hist:
    plt.figure(figsize=(6,4))
    plt.plot(hist["val_acc"], label="val_acc", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("⚠️ No accuracy recorded (regression task or training cell not logging accuracy).")


# ---- Code Cell ----


# ---- Code Cell ----


# ---- Code Cell ----


