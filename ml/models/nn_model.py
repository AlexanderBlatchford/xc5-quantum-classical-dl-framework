import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def train_nn(X_train, y_train, X_test, y_test, hidden_dim=64, lr=0.001,
             epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=True)

    model = FeedforwardNN(X_train.shape[1], hidden_dim, len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[NN] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Predictions
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device))
        preds = torch.argmax(preds, dim=1).cpu().numpy()

    return preds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def train_nn(X_train, y_train, X_test, y_test, hidden_dim=64, lr=0.001,
             epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=True)

    model = FeedforwardNN(X_train.shape[1], hidden_dim, len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[NN] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Predictions
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device))
        preds = torch.argmax(preds, dim=1).cpu().numpy()

    return preds
