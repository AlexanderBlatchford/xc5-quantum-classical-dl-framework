# NN.py
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# Define Neural Network
# -------------------------
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(FeedforwardNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# -------------------------
# Training function
# -------------------------
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()


# -------------------------
# Evaluation function
# -------------------------
def evaluate_model(model, test_loader, device, task):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            if task == "classification":
                if outputs.shape[1] == 1:  # binary
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).int()
                else:  # multi-class
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                predictions.extend(preds.cpu().numpy())
            else:  # regression
                predictions.extend(outputs.squeeze().cpu().numpy())

            actuals.extend(y_batch.cpu().numpy())
    return predictions, actuals


# -------------------------
# Main script
# -------------------------
def main(args):
    # Load CSV
    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset")

    # Features & target
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Encode target if classification
    if args.task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    if args.task == "classification":
        num_classes = len(set(y))
        if num_classes == 2:  # binary
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            output_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        else:  # multi-class
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            output_dim = num_classes
            criterion = nn.CrossEntropyLoss()
    else:  # regression
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        output_dim = 1
        criterion = nn.MSELoss()

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=args.batch_size,
    )

    # Model, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedforwardNN(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        train_model(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] completed")

    # Evaluation
    preds, actuals = evaluate_model(model, test_loader, device, args.task)
    print("\nSample predictions vs actual:")
    for p, a in list(zip(preds, actuals))[:10]:
        print(f"Pred: {p}, Actual: {a}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple NN on CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], default="classification")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    main(args)

# python NN.py --csv ../data/WineQT.csv --target quality --task classification --epochs 100 --lr 0.0005 --hidden_dim 128
