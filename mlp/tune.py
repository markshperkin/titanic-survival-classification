"""Hyperparameter search for the Titanic MLP via stratified 5-fold CV.

Runs entirely inside the training split (``data/train_split.csv``); the held-out
validation split is never touched here, so it stays an honest final test. For
each config and each fold we fit a *fresh* preprocessor on the fold's training
rows only (no leakage), train an MLP with early stopping on the fold's val rows,
and score ROC-AUC. The winning config (by mean CV AUC) is written to
``models/best_config.json`` for ``train.py`` to consume.

Run:
    python tune.py
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import DATA_DIR, PROJECT_ROOT
from src.preprocessing import TitanicPreprocessor
from mlp.model import MLP, evaluate, set_seed, train_model

TRAIN_SPLIT = DATA_DIR / "train_split.csv"
OUT = PROJECT_ROOT / "models" / "mlp" / "best_config.json"

# ── Search grid (full) ───────────────────────────────────────────────────────
HIDDEN_DIMS = [[16], [32], [64], [32, 16], [64, 32]]
DROPOUTS = [0.0, 0.3]
LEARNING_RATES = [1e-3, 5e-3, 1e-2]
BATCH_SIZES = [16, 32, 64, 128, None]  # None = full-batch
# => 5 x 2 x 3 x 5 = 150 configurations.

# Fixed across all configs.
N_FOLDS = 5
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10
SEED = 42


def cross_validate(config: dict, train_df: pd.DataFrame) -> dict:
    """Return mean/std CV ROC-AUC + F1 + best-epoch for one config."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y = train_df["Survived"].to_numpy()
    aucs, f1s, best_epochs = [], [], []

    for tr_idx, va_idx in skf.split(train_df, y):
        fold_tr = train_df.iloc[tr_idx]
        fold_va = train_df.iloc[va_idx]

        # Fit preprocessing on the fold's TRAINING rows only — no leakage.
        pre = TitanicPreprocessor().fit(fold_tr)
        x_tr, y_tr = pre.transform(fold_tr)
        x_va, y_va = pre.transform(fold_va)

        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        set_seed(SEED)
        model = MLP(x_tr.shape[1], config["hidden_dims"], config["dropout"])
        model, _, best_epoch = train_model(
            model, x_tr, y_tr, x_va, y_va,
            lr=config["learning_rate"], weight_decay=WEIGHT_DECAY,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, pos_weight=pos_weight,
            batch_size=config["batch_size"],
        )
        m = evaluate(model, x_va, y_va)
        aucs.append(m["roc_auc"])
        f1s.append(m["f1"])
        best_epochs.append(best_epoch)

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_f1": float(np.mean(f1s)),
        "mean_best_epoch": float(np.mean(best_epochs)),
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_SPLIT)
    grid = [
        {"hidden_dims": h, "dropout": d, "learning_rate": lr, "batch_size": bs}
        for h, d, lr, bs in itertools.product(HIDDEN_DIMS, DROPOUTS, LEARNING_RATES, BATCH_SIZES)
    ]
    print(f"Searching {len(grid)} configs x {N_FOLDS} folds on {len(train_df)} train rows...\n")

    rows = []
    for i, cfg in enumerate(grid, 1):
        res = cross_validate(cfg, train_df)
        rows.append({**cfg, **res})
        batch_label = "full" if cfg["batch_size"] is None else str(cfg["batch_size"])
        print(f"[{i:3d}/{len(grid)}] hidden={str(cfg['hidden_dims']):9s} "
              f"dropout={cfg['dropout']} lr={cfg['learning_rate']:<6g} batch={batch_label:<4s} "
              f"-> AUC {res['mean_auc']:.4f} ± {res['std_auc']:.4f}")

    results = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    results["hidden_dims"] = results["hidden_dims"].apply(str)
    results["batch_size"] = results["batch_size"].apply(lambda b: "full" if pd.isna(b) else int(b))

    print("\n=== Ranked by mean CV ROC-AUC (top 15) ===")
    print(results[["hidden_dims", "dropout", "learning_rate", "batch_size",
                   "mean_auc", "std_auc", "mean_f1", "mean_best_epoch"]].head(15).round(4).to_string(index=False))

    best = results.iloc[0]
    best_config = {
        "hidden_dims": json.loads(best["hidden_dims"]),
        "dropout": float(best["dropout"]),
        "learning_rate": float(best["learning_rate"]),
        "batch_size": None if best["batch_size"] == "full" else int(best["batch_size"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(best_config, indent=2))

    print(f"\nBest config: {best_config}")
    print(f"  mean CV AUC {best['mean_auc']:.4f} ± {best['std_auc']:.4f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
