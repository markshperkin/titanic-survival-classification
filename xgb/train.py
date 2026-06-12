"""Titanic XGBoost training entry point.

Mirrors ``mlp/train.py`` on the same split and the same 23-column preprocessed
matrix, so the held-out val metrics are directly comparable. Trains on the
712-row train split, uses the 179-row val for early stopping (tree count) and
reporting, saves the model + preprocessor, and prints val metrics.

Run:
    python -m xgb.train
    python -m xgb.train --force     # regenerate the train/val split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT
from src.preprocessing import TitanicPreprocessor
from src.split import load_or_make_split, verification_table
from xgb.model import evaluate, make_classifier, train_model

DEFAULT_OUT = PROJECT_ROOT / "models" / "xgb"

# Used when models/xgb/best_config.json is absent (winner of `python -m xgb.tune`).
DEFAULT_CONFIG = {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.9}


def load_config(out_dir: Path) -> dict:
    """Use the tuned config if present, else the baked-in default."""
    cfg_path = out_dir / "best_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        print(f"Using tuned config from {cfg_path}: {cfg}")
        return cfg
    print(f"No best_config.json found — using DEFAULT_CONFIG: {DEFAULT_CONFIG}")
    return DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic XGBoost training.")
    parser.add_argument("--data-path", default=None, help="Path to full train.csv (defaults to load_raw()).")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Where to save model + preprocessor.")
    parser.add_argument("--force", action="store_true", help="Regenerate the split files.")
    args = parser.parse_args()

    # Same split as the MLP (shared loader; reuses the on-disk files).
    train_df, val_df = load_or_make_split(args.val_size, args.seed, args.force, args.data_path)
    verification_table(train_df, val_df)

    print("\nFitting preprocessor on the training split...")
    pre = TitanicPreprocessor().fit(train_df)
    x_train, y_train = pre.transform(train_df)
    x_val, y_val = pre.transform(val_df)
    assert not pd.isna(x_train).any(), "NaNs in X_train"
    assert not pd.isna(x_val).any(), "NaNs in X_val"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre.save(out_dir / "preprocessor.joblib")
    print(f"Feature matrix: {len(pre.feature_names_)} columns  |  X_train {x_train.shape}  X_val {x_val.shape}")
    print(f"Saved preprocessor -> {out_dir / 'preprocessor.joblib'}")

    # ── Train the final XGBoost model ────────────────────────────────────────
    # Trees are added until val loss stops improving (early stopping on the val
    # set). As with the MLP, val is used for both stopping and reporting, so the
    # reported val metrics are mildly optimistic — standard validation-set use.
    config = load_config(out_dir)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print("\nTraining final XGBoost (early stopping on val log-loss)...")
    model = make_classifier(config, scale_pos_weight=pos_weight, seed=args.seed)
    model = train_model(model, x_train, y_train, x_val, y_val)
    print(f"Best tree count: {int(model.best_iteration) + 1}")

    model.save_model(out_dir / "xgb.json")
    print(f"Saved model -> {out_dir / 'xgb.json'}")

    # ── Report (val metrics) ─────────────────────────────────────────────────
    train_m = evaluate(model, x_train, y_train)
    val_m = evaluate(model, x_val, y_val)
    report = pd.DataFrame({"train": train_m, "val": val_m}).round(4)
    print("\nMetrics (train vs held-out val):")
    print(report.to_string())
    print("Baseline to beat (majority class): 0.6162")


if __name__ == "__main__":
    main()
