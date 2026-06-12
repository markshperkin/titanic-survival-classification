"""Fit-on-train preprocessing for the Titanic classification task.

``TitanicPreprocessor`` learns everything from the training split only
(imputation values, the FareRank reference, the scaler, the one-hot encoder),
then applies the identical transform to validation and inference data. The
fitted object is saved to disk by ``train.py`` and reloaded by the Streamlit
inference app so both paths share exactly the same feature pipeline.

Encoding plan mirrors section 8 of ``notebooks/01_eda.ipynb``.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import add_engineered

# Column roles in the final feature matrix.
NUMERIC = ["Age", "LogFare", "FareRank", "FamilySize", "SibSp", "Parch", "TicketGroupSize"]
BINARY = ["IsAlone"]
CATEGORICAL = ["Sex", "Pclass", "Embarked", "Title"]

# Fixed category vocabularies so the one-hot output is always the same 23 columns,
# even when a small CV fold (or an inference batch) happens to omit a rare class
# such as the `Rev` / `Dr` titles. Order here defines the output column order.
CATEGORIES = [
    ["female", "male"],            # Sex
    [1, 2, 3],                     # Pclass
    ["C", "Q", "S"],               # Embarked
    ["Dr", "Master", "Miss", "Mr", "Mrs", "Other", "Rev"],  # Title
]


class TitanicPreprocessor:
    """Learn preprocessing params on train, apply to any frame.

    Use ``fit(train_df)`` once, then ``transform(df)`` for train / val / inference.
    ``transform`` returns ``(X, y)`` where ``y`` is ``None`` if the frame has no
    ``Survived`` column (the inference case).
    """

    def __init__(self) -> None:
        self.age_medians_by_title: dict = {}
        self.age_global_median: float = np.nan
        self.embarked_mode: str = "S"
        self.fare_median: float = np.nan
        self.farerank_ref: dict[int, np.ndarray] = {}
        self._farerank_ref_all: np.ndarray = np.array([])
        self.scaler: StandardScaler | None = None
        self.encoder: OneHotEncoder | None = None
        self.feature_names_: list[str] = []

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "TitanicPreprocessor":
        # Imputation references (learned on train only).
        self.fare_median = float(train_df["Fare"].median())
        self.embarked_mode = str(train_df["Embarked"].mode(dropna=True).iloc[0])

        base = add_engineered(train_df)
        self.age_global_median = float(base["Age"].median())
        self.age_medians_by_title = base.groupby("Title")["Age"].median().to_dict()

        # FareRank reference: sorted train fares per Pclass (+ a global fallback).
        fare_filled = train_df["Fare"].fillna(self.fare_median).to_numpy()
        pclass = train_df["Pclass"].to_numpy()
        self.farerank_ref = {
            int(p): np.sort(fare_filled[pclass == p]) for p in np.unique(pclass)
        }
        self._farerank_ref_all = np.sort(fare_filled)

        # Build the engineered/imputed train frame, then fit scaler + encoder on it.
        eng = self._build_features(train_df)
        self.scaler = StandardScaler().fit(eng[NUMERIC])
        self.encoder = OneHotEncoder(
            categories=CATEGORIES, handle_unknown="ignore", sparse_output=False
        ).fit(eng[CATEGORICAL])
        self.feature_names_ = (
            NUMERIC + BINARY + list(self.encoder.get_feature_names_out(CATEGORICAL))
        )
        return self

    # ── transforming ─────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        if self.scaler is None or self.encoder is None:
            raise RuntimeError("Preprocessor must be fit (or loaded) before transform.")

        eng = self._build_features(df)
        x_num = self.scaler.transform(eng[NUMERIC])
        x_bin = eng[BINARY].to_numpy(dtype=float)
        x_cat = self.encoder.transform(eng[CATEGORICAL])
        x = np.hstack([x_num, x_bin, x_cat]).astype(np.float32)

        y = eng["Survived"].to_numpy(dtype=np.float32) if "Survived" in eng else None
        return x, y

    def fit_transform(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        return self.fit(train_df).transform(train_df)

    # ── internals ────────────────────────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer + impute a frame using the learned params (no scaling/encoding)."""
        d = df.copy()
        d["Fare"] = d["Fare"].fillna(self.fare_median)  # before LogFare / FareRank
        eng = add_engineered(d)

        # Age: median by Title group, falling back to the global median.
        title_median = eng["Title"].map(self.age_medians_by_title).fillna(self.age_global_median)
        eng["Age"] = eng["Age"].fillna(title_median)

        eng["Embarked"] = eng["Embarked"].fillna(self.embarked_mode)
        eng["FareRank"] = self._fare_rank(eng["Fare"].to_numpy(), eng["Pclass"].to_numpy())
        return eng

    def _fare_rank(self, fares: np.ndarray, pclasses: np.ndarray) -> np.ndarray:
        """Percentile of each fare within its Pclass, vs the stored train reference."""
        out = np.empty(len(fares), dtype=float)
        seen = np.zeros(len(fares), dtype=bool)
        for p, ref in self.farerank_ref.items():
            mask = pclasses == p
            if mask.any():
                out[mask] = np.searchsorted(ref, fares[mask], side="right") / len(ref)
                seen |= mask
        if not seen.all():  # Pclass unseen in train — fall back to global distribution
            ref = self._farerank_ref_all
            out[~seen] = np.searchsorted(ref, fares[~seen], side="right") / len(ref)
        return out

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TitanicPreprocessor":
        return joblib.load(path)
