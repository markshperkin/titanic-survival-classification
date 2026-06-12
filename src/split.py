"""Shared stratified train/validation split.

Both model strategies (``mlp/`` and ``xgb/``) call this so they train and
evaluate on the *exact same* split — that is what makes the MLP-vs-XGBoost
comparison fair. The split files live in ``data/`` and are reused if present.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import DATA_DIR, load_raw

TRAIN_SPLIT = DATA_DIR / "train_split.csv"
VAL_SPLIT = DATA_DIR / "val_split.csv"


def load_or_make_split(
    val_size: float = 0.20, seed: int = 42, force: bool = False, data_path=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df), reusing the on-disk split unless ``force``."""
    if TRAIN_SPLIT.exists() and VAL_SPLIT.exists() and not force:
        print(f"Reusing existing splits in {DATA_DIR} (use --force to regenerate).")
        return pd.read_csv(TRAIN_SPLIT), pd.read_csv(VAL_SPLIT)

    df = load_raw(data_path)
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["Survived"], random_state=seed
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_SPLIT, index=False)
    val_df.to_csv(VAL_SPLIT, index=False)
    print(f"Wrote {len(train_df)} train / {len(val_df)} val rows to {DATA_DIR}")
    return train_df, val_df


def verification_table(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Confirm the split kept the target balanced and features representative."""
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
