from typing import Dict, List, Tuple
import time, numpy as np

def run_mlp_torch(Xtr, ytr, Xte, params: Dict, classes: List[str]) -> Tuple[np.ndarray, Dict, Dict]:
    try:
        import torch
        from torch import nn
    except Exception as e:
        raise RuntimeError(f"mlp_torch requires torch. Install: pip install torch. Original error: {e}")

    torch.manual_seed(42)

    hidden_cfg = params.get("hidden", [64, 64])
    if isinstance(hidden_cfg, (list, tuple)):
        hidden = tuple(int(h) for h in hidden_cfg)
    else:
        hidden = (int(hidden_cfg),)

    epochs = int(params.get("epochs", 20))
    lr = float(params.get("lr", 1e-3))
    batch = int(params.get("batch_size", 64))
    dropout = float(params.get("dropout", 0.0))
    n_out = len(classes)

    class MLP(nn.Module):
        def __init__(self, d_in: int, hidden: tuple, d_out: int, dropout: float = 0.0):
            super().__init__()
            layers = []
            nprev = d_in
            for h in hidden:
                layers += [nn.Linear(nprev, h), nn.ReLU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
                nprev = h
            layers += [nn.Linear(nprev, d_out)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    model = MLP(d_in=Xtr.shape[1], hidden=hidden, d_out=n_out, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    train_ms = (time.perf_counter() - t0) * 1000.0

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t).numpy()
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = proba / proba.sum(axis=1, keepdims=True)

    return proba, {"train_ms": train_ms, "infer_ms": 0.0}, {"hidden": list(hidden)}
