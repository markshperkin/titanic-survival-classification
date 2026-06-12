"""Titanic training entry point.

load the full dataset, make a stratified 
train/validation split, persist the splits, fit the preprocessing pipeline on
the training split, and save it to disk. Model definition and training are added
in the next step.

Run:
    python train.py                # default 80/20 split, seed 42
    python train.py --force        # regenerate the splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data import DATA_DIR, PROJECT_ROOT, load_raw
from src.preprocessing import TitanicPreprocessor
from mlp.model import MLP, evaluate, set_seed, train_model

TRAIN_SPLIT = DATA_DIR / "train_split.csv"
VAL_SPLIT = DATA_DIR / "val_split.csv"
DEFAULT_OUT = PROJECT_ROOT / "models" / "mlp"

# Used when models/best_config.json is absent (winner of `python -m mlp.tune`).
DEFAULT_CONFIG = {"hidden_dims": [32], "dropout": 0.3, "learning_rate": 1e-3, "batch_size": None}

# Fixed training hyperparameters.
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 10


def make_split(df: pd.DataFrame, val_size: float, seed: int, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df), reusing existing split files unless ``force``."""
    if TRAIN_SPLIT.exists() and VAL_SPLIT.exists() and not force:
        print(f"Reusing existing splits in {DATA_DIR} (use --force to regenerate).")
        return pd.read_csv(TRAIN_SPLIT), pd.read_csv(VAL_SPLIT)

    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["Survived"], random_state=seed
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT, index=False)
    val_df.to_csv(VAL_SPLIT, index=False)
    print(f"Wrote {len(train_df)} train / {len(val_df)} val rows to {DATA_DIR}")
    return train_df, val_df


def verification_table(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Confirm the stratified split kept the target balanced and features representative."""
    def stats(d: pd.DataFrame) -> dict:
        return {
            "n": len(d),
            "Survived %": d["Survived"].mean() * 100,
            "Sex female %": (d["Sex"] == "female").mean() * 100,
            "Pclass=3 %": (d["Pclass"] == 3).mean() * 100,
            "Age mean": d["Age"].mean(),
            "Fare mean": d["Fare"].mean(),
        }

    table = pd.DataFrame({"train": stats(train_df), "val": stats(val_df)}).round(2)
    print("\nSplit verification (train vs val should be close):")
    print(table.to_string())


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
    parser = argparse.ArgumentParser(description="Titanic data prep + preprocessing.")
    parser.add_argument("--data-path", default=None, help="Path to full train.csv (defaults to load_raw()).")
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Where to save the preprocessor.")
    parser.add_argument("--force", action="store_true", help="Regenerate the split files.")
    args = parser.parse_args()

    df = load_raw(args.data_path)
    print(f"Loaded full dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    train_df, val_df = make_split(df, args.val_size, args.seed, args.force)
    verification_table(train_df, val_df)

    print("\nFitting preprocessor on the training split...")
    pre = TitanicPreprocessor().fit(train_df)
    x_train, y_train = pre.transform(train_df)
    x_val, y_val = pre.transform(val_df)

    # Sanity checks.
    assert not pd.isna(x_train).any(), "NaNs in X_train"
    assert not pd.isna(x_val).any(), "NaNs in X_val"

    out_dir = Path(args.out_dir)
    pre.save(out_dir / "preprocessor.joblib")

    print(f"\nFeature matrix: {len(pre.feature_names_)} columns")
    print(f"  X_train: {x_train.shape}   (survived {y_train.mean():.3f})")
    print(f"  X_val:   {x_val.shape}   (survived {y_val.mean():.3f})")
    print(f"Saved preprocessor -> {out_dir / 'preprocessor.joblib'}")

    # ── Train the final MLP ──────────────────────────────────────────────────
    # Train on the full 712-row train split; evaluate on the 179-row val each
    # epoch. Early stopping keeps the weights with the best val accuracy. Note:
    # the val set is used both to pick the stop epoch and to report below, so the
    # reported val accuracy is mildly optimistic (standard validation-set use).
    config = load_config(out_dir)
    set_seed(args.seed)
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    batch_size = config.get("batch_size")  # None -> full-batch
    batch_label = "full" if batch_size is None else batch_size
    print(f"\nTraining final MLP (max {MAX_EPOCHS} epochs, batch={batch_label}, "
          f"early stop on val accuracy, patience {PATIENCE})...")
    model = MLP(x_train.shape[1], config["hidden_dims"], config["dropout"])
    model, _, best_epoch = train_model(
        model, x_train, y_train, x_val, y_val,
        lr=config["learning_rate"], weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, pos_weight=pos_weight,
        batch_size=batch_size, monitor="val_acc", verbose=True,
    )
    print(f"\nBest weights from epoch {best_epoch} (highest val accuracy).")

    # Save the model bundle so Streamlit / inference can rebuild it.
    bundle = {
        "state_dict": model.state_dict(),
        "config": config,
        "input_dim": x_train.shape[1],
        "feature_names": pre.feature_names_,
    }
    torch.save(bundle, out_dir / "mlp.pt")
    print(f"Saved model -> {out_dir / 'mlp.pt'}")

    # ── Report (honest val metrics) ──────────────────────────────────────────
    train_m = evaluate(model, x_train, y_train)
    val_m = evaluate(model, x_val, y_val)
    report = pd.DataFrame({"train": train_m, "val": val_m}).round(4)
    print("\nMetrics (train vs held-out val):")
    print(report.to_string())
    print(f"\nBaseline to beat (majority class): 0.6162")


if __name__ == "__main__":
    main()
