"""
C3: Logistic Regression (One-vs-Rest) on the UCI Iris dataset.

Expects data/iris.data to exist -- the raw UCI file, comma-separated, no
header, columns:
    sepal_length, sepal_width, petal_length, petal_width, class
where class is one of: Iris-setosa, Iris-versicolor, Iris-virginica.
(This is exactly the file distributed at
 https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

Run:
    python -m src.c3_logistic_regression
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = "outputs/c3"
DATA_PATH = "data/iris.data"


def load_iris_data(path=DATA_PATH):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))
    rows = np.array(rows)
    X = rows[:, :4].astype(float)
    labels = rows[:, 4]
    classes = sorted(str(c) for c in set(labels))  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    y = np.array([classes.index(lbl) for lbl in labels])
    return X, y, classes


# ---------------------------------------------------------------------------
# (a) From-scratch binary cross-entropy logistic regression
# ---------------------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y_true, y_pred, eps=1e-12):
    """
    Binary cross-entropy loss, written from scratch:
        L = -mean( y*log(p) + (1-y)*log(1-p) )
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def bce_gradient(X, y_true, y_pred):
    """
    Gradient of mean BCE loss w.r.t. weights w (X already includes bias column):
        dL/dw = (1/n) * X^T (p - y)
    """
    n = X.shape[0]
    return (X.T @ (y_pred - y_true)) / n


def train_logreg_scratch(X, y_binary, lr=0.1, n_iters=3000):
    """
    Full-batch gradient descent logistic regression from scratch.
    X: (n, d) features WITHOUT bias column (bias is added internally).
    y_binary: (n,) 0/1 labels for one-vs-rest.
    Returns learned weights w (d+1,) including bias, and loss history.
    """
    n, d = X.shape
    Xb = np.column_stack([np.ones(n), X])  # add bias column
    w = np.zeros(d + 1)
    history = []
    for _ in range(n_iters):
        z = Xb @ w
        p = sigmoid(z)
        loss = bce_loss(y_binary, p)
        history.append(loss)
        grad = bce_gradient(Xb, y_binary, p)
        w = w - lr * grad
    return w, history


def predict_proba_scratch(X, w):
    n = X.shape[0]
    Xb = np.column_stack([np.ones(n), X])
    return sigmoid(Xb @ w)


def one_vs_rest_scratch(X_train, y_train, X_test, y_test, n_classes, log):
    log("\n--- (a) One-vs-rest logistic regression, from scratch ---")
    weights = {}
    for c in range(n_classes):
        y_bin_train = (y_train == c).astype(float)
        w, history = train_logreg_scratch(X_train, y_bin_train, lr=0.1, n_iters=3000)
        weights[c] = w
        log(f"class {c} vs rest: final BCE loss = {history[-1]:.4f}")

    # Predict: pick the class whose one-vs-rest classifier gives highest probability
    n_test = X_test.shape[0]
    probs = np.column_stack([predict_proba_scratch(X_test, weights[c]) for c in range(n_classes)])
    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y_test))
    log(f"Scratch one-vs-rest test accuracy = {acc:.4f}")
    return acc, weights


# ---------------------------------------------------------------------------
# (b) PyTorch: nn.BCELoss + torch.optim.SGD
# ---------------------------------------------------------------------------

def one_vs_rest_pytorch(X_train, y_train, X_test, y_test, n_classes, log,
                         lr=0.1, n_iters=3000):
    import torch
    import torch.nn as nn

    log("\n--- (b) One-vs-rest logistic regression, PyTorch (nn.BCELoss + SGD) ---")
    torch.manual_seed(0)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    models = {}
    for c in range(n_classes):
        y_bin_train = torch.tensor((y_train == c).astype(np.float32)).unsqueeze(1)

        model = nn.Sequential(nn.Linear(X_train.shape[1], 1), nn.Sigmoid())
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for _ in range(n_iters):
            optimizer.zero_grad()
            p = model(Xtr)
            loss = criterion(p, y_bin_train)
            loss.backward()
            optimizer.step()

        models[c] = model
        log(f"class {c} vs rest: final BCE loss = {loss.item():.4f}")

    with torch.no_grad():
        probs = torch.cat([models[c](Xte) for c in range(n_classes)], dim=1)
        y_pred = torch.argmax(probs, dim=1).numpy()

    acc = float(np.mean(y_pred == y_test))
    log(f"PyTorch one-vs-rest test accuracy = {acc:.4f}")
    return acc, models


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    def log(msg):
        print(msg)
        lines.append(str(msg))

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Expected the UCI iris file at {DATA_PATH}. "
            f"Download it from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data "
            f"and place it there."
        )

    X, y, classes = load_iris_data(DATA_PATH)
    log(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, classes={classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    # Standardize features (helps gradient descent converge for both scratch & torch)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = len(classes)

    acc_scratch, _ = one_vs_rest_scratch(X_train, y_train, X_test, y_test, n_classes, log)

    try:
        acc_torch, _ = one_vs_rest_pytorch(X_train, y_train, X_test, y_test, n_classes, log)
    except Exception as e:
        log(f"\nPyTorch unavailable ({e.__class__.__name__}) -- skipping part (b). "
            "Install with: pip install torch")
        acc_torch = None

    log("\n--- Comparison ---")
    log(f"Scratch (numpy) accuracy : {acc_scratch:.4f}")
    if acc_torch is not None:
        log(f"PyTorch accuracy         : {acc_torch:.4f}")
        log("Discussion: both implementations minimize the same binary "
            "cross-entropy objective with (full-batch) gradient descent, so "
            "their test accuracies are typically identical or within one "
            "misclassified sample of each other on this small, linearly "
            "near-separable dataset. Differences, when present, come from "
            "PyTorch's autograd-computed gradients / float32 precision vs "
            "our hand-derived float64 gradient, not from any difference in "
            "the underlying method.")

    with open(f"{OUT}/report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
