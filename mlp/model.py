"""PyTorch MLP and training/evaluation helpers for the Titanic task.

Shared by the hyperparameter search (``tune.py``), the final training script
(``train.py``), and the Streamlit inference app. The model is deliberately small
and regularised — with only ~712 training rows the priority is generalisation,
not capacity.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch for reproducible CPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Configurable feed-forward classifier outputting a single logit.

    ``hidden_dims`` lists the width of each hidden layer, e.g. ``[32]`` for one
    hidden layer or ``[64, 32]`` for two. Each hidden layer is Linear -> ReLU ->
    Dropout; the final layer is a single linear unit (logit for BCEWithLogits).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32)


def train_model(
    model: MLP,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_es: np.ndarray,
    y_es: np.ndarray,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 10,
    pos_weight: float | None = None,
    batch_size: int | None = None,
    monitor: str = "val_acc",
    verbose: bool = False,
) -> tuple[MLP, dict, int]:
    """Train with early stopping, evaluating on ``(x_es, y_es)`` each epoch.

    Adam optimiser. ``batch_size`` controls mini-batching: ``None`` (or a size
    >= the training set) means full-batch — one exact gradient step per epoch;
    a smaller value shuffles the data each epoch and steps once per mini-batch,
    trading gradient exactness for regularising noise + more updates per epoch.
    ``monitor`` selects the early-stopping criterion: ``"val_acc"`` keeps the
    weights with the highest validation accuracy (maximise); ``"val_loss"`` the
    lowest validation BCE loss (minimise). Training stops after ``patience``
    epochs with no improvement, then the best weights are restored. ``verbose``
    prints train/val accuracy each epoch. Returns ``(model, history, best_epoch)``.
    """
    xf, yf = _to_tensor(x_fit), _to_tensor(y_fit)
    xe, ye = _to_tensor(x_es), _to_tensor(y_es)

    pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    maximise = monitor == "val_acc"
    best_score = -float("inf") if maximise else float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0

    n = xf.shape[0]
    bs = n if not batch_size or batch_size >= n else batch_size  # None/large -> full-batch

    if verbose:
        print(f"{'epoch':>5} | {'train_acc':>9} | {'val_acc':>8} | {'val_loss':>9} | status")

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n)  # shuffle each epoch (seeded globally via set_seed)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            optimizer.zero_grad()
            loss = criterion(model(xf[idx]), yf[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(xf), yf).item()
            val_loss = criterion(model(xe), ye).item()
            train_acc = ((torch.sigmoid(model(xf)) >= 0.5).float() == yf).float().mean().item()
            val_acc = ((torch.sigmoid(model(xe)) >= 0.5).float() == ye).float().mean().item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        score = val_acc if maximise else val_loss
        improved = (score > best_score + 1e-9) if maximise else (score < best_score - 1e-9)
        if improved:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            status = f"* new best ({monitor}={best_score:.4f})" if improved \
                else f"no-improve {epochs_no_improve}/{patience}"
            print(f"{epoch:>5} | {train_acc:>9.3f} | {val_acc:>8.3f} | {val_loss:>9.4f} | {status}")

        if not improved and epochs_no_improve >= patience:
            if verbose:
                print(f"Early stop: no {monitor} improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def predict_proba(model: MLP, x: np.ndarray) -> np.ndarray:
    """Return P(survived) for each row."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_to_tensor(x))).numpy()


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Return accuracy, precision, recall, F1 and ROC-AUC."""
    proba = predict_proba(model, x)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
