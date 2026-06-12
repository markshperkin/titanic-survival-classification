"""Hyperparameter search for the Titanic XGBoost model via stratified 5-fold CV.

Same protocol as ``mlp/tune.py`` so the two strategies are tuned with equal
rigor: the search runs entirely inside ``data/train_split.csv`` (the 179-row val
stays sealed), preprocessing is fit fresh inside each fold (no leakage), and the
winner (by mean CV ROC-AUC) is written to ``models/xgb/best_config.json``.

Run:
    python -m xgb.tune
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
from xgb.model import evaluate, make_classifier, train_model

TRAIN_SPLIT = DATA_DIR / "train_split.csv"
OUT = PROJECT_ROOT / "models" / "xgb" / "best_config.json"

# ── Search grid (full) ───────────────────────────────────────────────────────
MAX_DEPTHS = [2, 3, 4]
LEARNING_RATES = [0.01, 0.05, 0.10]
SUBSAMPLES = [0.8, 1.0]
# => 3 x 3 x 2 = 18 configurations. n_estimators is fixed high (500) and the
#    effective tree count is found per-fold by early stopping.

N_FOLDS = 5
SEED = 42


def cross_validate(config: dict, train_df: pd.DataFrame) -> dict:
    """Return mean/std CV ROC-AUC + F1 + mean best tree count for one config."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y = train_df["Survived"].to_numpy()
    aucs, f1s, n_trees = [], [], []

    for tr_idx, va_idx in skf.split(train_df, y):
        fold_tr = train_df.iloc[tr_idx]
        fold_va = train_df.iloc[va_idx]

        # Fit preprocessing on the fold's TRAINING rows only — no leakage.
        pre = TitanicPreprocessor().fit(fold_tr)
        x_tr, y_tr = pre.transform(fold_tr)
        x_va, y_va = pre.transform(fold_va)

        pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        model = make_classifier(config, scale_pos_weight=pos_weight, seed=SEED)
        model = train_model(model, x_tr, y_tr, x_va, y_va)

        m = evaluate(model, x_va, y_va)
        aucs.append(m["roc_auc"])
        f1s.append(m["f1"])
        n_trees.append(int(model.best_iteration) + 1)

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_f1": float(np.mean(f1s)),
        "mean_n_trees": float(np.mean(n_trees)),
    }


def main() -> None:
    train_df = pd.read_csv(TRAIN_SPLIT)
    grid = [
        {"max_depth": d, "learning_rate": lr, "subsample": s}
        for d, lr, s in itertools.product(MAX_DEPTHS, LEARNING_RATES, SUBSAMPLES)
    ]
    print(f"Searching {len(grid)} configs x {N_FOLDS} folds on {len(train_df)} train rows...\n")

    rows = []
    for i, cfg in enumerate(grid, 1):
        res = cross_validate(cfg, train_df)
        rows.append({**cfg, **res})
        print(f"[{i:2d}/{len(grid)}] depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']:<5g} subsample={cfg['subsample']} "
              f"-> AUC {res['mean_auc']:.4f} ± {res['std_auc']:.4f} "
              f"(~{res['mean_n_trees']:.0f} trees)")

    results = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    print("\n=== Ranked by mean CV ROC-AUC ===")
    print(results[["max_depth", "learning_rate", "subsample",
                   "mean_auc", "std_auc", "mean_f1", "mean_n_trees"]].round(4).to_string(index=False))

    best = results.iloc[0]
    best_config = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(best_config, indent=2))

    print(f"\nBest config: {best_config}")
    print(f"  mean CV AUC {best['mean_auc']:.4f} ± {best['std_auc']:.4f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
