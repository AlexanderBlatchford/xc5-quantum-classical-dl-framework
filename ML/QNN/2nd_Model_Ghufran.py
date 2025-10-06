# ---- Code Cell ----
# If you already have these, you can skip this cell.
!pip install -q torch pennylane
# Optional backends if you want to try Qiskit later:
# !pip install -q pennylane-qiskit qiskit-aer


# ---- Code Cell ----
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import pennylane as qml
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== experiment knobs ====
N_QUBITS  = 4          # try 2, 4, 6
N_LAYERS  = 2          # quantum circuit depth
N_OUTPUTS = 1          # 1 for binary classification; set >1 for multi-class
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 15
ENCODING = "angles"    # feature -> rotation angles
USE_QISKIT = False     # set True only if you installed qiskit backends

# ==== device selection ====
if USE_QISKIT:
    # requires: pennylane-qiskit & qiskit-aer
    dev = qml.device("qiskit.aer", wires=N_QUBITS, shots=1024)
else:
    # fast analytic simulator (no shots)
    dev = qml.device("default.qubit", wires=N_QUBITS, shots=None)

print(dev)


# ---- Code Cell ----
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.utils.data as td

# load iris
iris = load_iris()
X = iris.data.astype("float32")    # shape (150,4)
y = iris.target                    # shape (150,)

# config for iris
N_QUBITS = 4        # one per feature
N_OUTPUTS = 3       # 3 classes
print("X:", X.shape, "y:", y.shape)

# train/test split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

class IrisDataset(td.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)       # float32
        self.y = torch.from_numpy(y).long()# long for CrossEntropyLoss
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = td.DataLoader(IrisDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = td.DataLoader(IrisDataset(X_te, y_te), batch_size=32, shuffle=False)


# ---- Code Cell ----
criterion = nn.CrossEntropyLoss()

def accuracy_from_logits_mc(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds.eq(y_true).float().mean()).item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_acc, n = 0.0, 0
    for xb, yb in loader:
        logits = model(xb)
        total_acc += accuracy_from_logits_mc(logits, yb) * len(xb)
        n += len(xb)
    return total_acc / max(n, 1)

def train_model(model, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    train_losses, val_accs = [], []
    for ep in range(1, epochs+1):
        model.train()
        run = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        tr_loss = run / len(train_loader.dataset)
        va_acc = evaluate(model, test_loader)
        sched.step(va_acc)
        train_losses.append(tr_loss); val_accs.append(va_acc)
        print(f"Epoch {ep:02d} | loss={tr_loss:.4f} | val_acc={va_acc:.3f}")
    return train_losses, val_accs


# ---- Code Cell ----
class MLPBaseline(nn.Module):
    def __init__(self, d_in, d_hidden=32, d_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, d_out)   # 3-class logits
        )
    def forward(self, x): return self.net(x)

mlp = MLPBaseline(d_in=N_QUBITS, d_out=N_OUTPUTS)
print(mlp)


# ---- Code Cell ----
@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)}
QLayer = qml.qnn.TorchLayer(qnode, weight_shapes)

class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(N_QUBITS, N_QUBITS),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(math.pi), requires_grad=False)
        self.q = QLayer
        self.head = nn.Sequential(
            nn.Linear(N_QUBITS, 16),
            nn.ReLU(),
            nn.Linear(16, N_OUTPUTS)   # 3-class logits
        )
    def forward(self, x):
        z = self.pre(x) * self.scale
        qexp = self.q(z)
        return self.head(qexp)

hybrid = HybridQCNN()
print(hybrid)


# ---- Code Cell ----
print("=== Training MLP Baseline ===")
mlp_train_losses, mlp_val_accs = train_model(mlp)

print("\n=== Training Hybrid Model ===")
hyb_train_losses, hyb_val_accs = train_model(hybrid)


# ---- Code Cell ----
plt.figure()
plt.plot(mlp_train_losses, label="MLP Loss")
plt.plot(hyb_train_losses, label="Hybrid Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

plt.figure()
plt.plot(mlp_val_accs, label="MLP Val Acc")
plt.plot(hyb_val_accs, label="Hybrid Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()

print(f"Final MLP acc:    {mlp_val_accs[-1]:.3f}")
print(f"Final Hybrid acc: {hyb_val_accs[-1]:.3f}")


# ---- Code Cell ----
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.append(preds)
        y_true.append(yb.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return y_true, y_pred

# predictions
y_true_mlp, y_pred_mlp = get_preds(mlp, test_loader)
y_true_hyb, y_pred_hyb = get_preds(hybrid, test_loader)

labels = np.unique(np.concatenate([y_true_mlp, y_true_hyb]))

# confusion matrices (side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cm_mlp = confusion_matrix(y_true_mlp, y_pred_mlp, labels=labels)
cm_hyb = confusion_matrix(y_true_hyb, y_pred_hyb, labels=labels)

disp1 = ConfusionMatrixDisplay(cm_mlp, display_labels=[iris.target_names[i] for i in labels])
disp1.plot(ax=axes[0], colorbar=False)
axes[0].set_title("MLP — Confusion Matrix")

disp2 = ConfusionMatrixDisplay(cm_hyb, display_labels=[iris.target_names[i] for i in labels])
disp2.plot(ax=axes[1], colorbar=False)
axes[1].set_title("Hybrid — Confusion Matrix")

plt.tight_layout()
plt.show()

# precision/recall/F1 per class
print("MLP classification report:\n",
      classification_report(y_true_mlp, y_pred_mlp, target_names=iris.target_names))
print("Hybrid classification report:\n",
      classification_report(y_true_hyb, y_pred_hyb, target_names=iris.target_names))


# ---- Code Cell ----


# ---- Code Cell ----


# ---- Code Cell ----


# ---- Code Cell ----


