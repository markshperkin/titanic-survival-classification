"""XGBoost classifier + training/evaluation helpers for the Titanic task.

Mirrors the role of ``mlp/model.py`` so the two strategies are directly
comparable: same 23-column preprocessed matrix in, same metrics out. The only
difference is the model — gradient-boosted trees instead of a neural net.

XGBoost grows trees one at a time, each correcting the previous trees' errors;
``early_stopping_rounds`` stops adding trees once the validation metric plateaus
(the tree-count analogue of the MLP's epoch early stopping).
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Hyperparameters not swept in the search (sensible, lightly-regularised defaults).
N_ESTIMATORS = 500          # upper bound; early stopping picks the real count
EARLY_STOPPING_ROUNDS = 20
COLSAMPLE_BYTREE = 0.8
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1


def make_classifier(params: dict, *, scale_pos_weight: float, seed: int = 42) -> xgb.XGBClassifier:
    """Build an XGBClassifier from a config dict (``max_depth``, ``learning_rate``, ``subsample``)."""
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=seed,
        n_jobs=0,
    )


def train_model(model: xgb.XGBClassifier, x_tr, y_tr, x_es, y_es) -> xgb.XGBClassifier:
    """Fit with early stopping on the monitor set ``(x_es, y_es)``.

    After fitting, ``model.best_iteration`` holds the chosen number of trees and
    predictions use that best round automatically.
    """
    model.fit(x_tr, y_tr, eval_set=[(x_es, y_es)], verbose=False)
    return model


def predict_proba(model: xgb.XGBClassifier, x: np.ndarray) -> np.ndarray:
    """Return P(survived) for each row."""
    return model.predict_proba(x)[:, 1]


def evaluate(model: xgb.XGBClassifier, x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Return accuracy, precision, recall, F1 and ROC-AUC (same metrics as the MLP)."""
    proba = predict_proba(model, x)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }
