import sys, os
sys.path.append(os.path.dirname(__file__))

import argparse
from utils import load_data, evaluate
from models import train_nn, train_logistic, train_random_forest, train_svm

def main(args):
    (X_train, X_test, y_train, y_test), labels = load_data(args.csv, args.target, args.test_size)

    if args.model == "nn":
        preds = train_nn(X_train, y_train, X_test, y_test,
                         hidden_dim=args.hidden_dim,
                         lr=args.lr,
                         epochs=args.epochs,
                         batch_size=args.batch_size)
    elif args.model == "logistic":
        preds = train_logistic(X_train, y_train, X_test, y_test)
    elif args.model == "randomforest":
        preds = train_random_forest(X_train, y_train, X_test, y_test)
    elif args.model == "svm":
        preds = train_svm(X_train, y_train, X_test, y_test)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    evaluate(y_test, preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification models")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--model", type=str, choices=["nn", "logistic", "randomforest", "svm"], default="nn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    main(args)
