"""Stateless feature engineering for the Titanic dataset.

Every function here is a pure transform of the input frame — no fitting, no
learned state, no use of the target. The same operations run identically during
training and inference. Anything that must *learn* from the training split
(imputation values, scalers, encoders, the FareRank reference) lives in
``preprocessing.py`` instead.

Mirrors the decisions finalized in section 8 of ``notebooks/01_eda.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Titles kept as their own class; everything else collapses to "Other" (7 classes).
KEEP_TITLES = ["Mr", "Miss", "Mrs", "Master", "Dr", "Rev"]
_TITLE_RE = r",\s*([^\.]+)\."


def extract_title(names: pd.Series) -> pd.Series:
    """Extract the honorific from each name and group to 7 classes.

    e.g. "Braund, Mr. Owen Harris" -> "Mr". Rare/aristocratic/alternate titles
    (Major, Col, Mlle, the Countess, ...) all map to "Other".
    """
    raw = names.str.extract(_TITLE_RE)[0].str.strip()
    return raw.where(raw.isin(KEEP_TITLES), "Other")


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered columns added.

    Adds: ``Title``, ``FamilySize``, ``IsAlone``, ``TicketGroupSize``, ``LogFare``.

    ``TicketGroupSize`` is counted *within the given frame* — it is a property of
    whatever set of passengers is passed in. This matches inference (the app
    receives a CSV and counts groups within it); a lone passenger -> 1.
    """
    out = df.copy()
    out["Title"] = extract_title(out["Name"])
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["TicketGroupSize"] = out["Ticket"].map(out["Ticket"].value_counts())
    out["LogFare"] = np.log1p(out["Fare"])
    return out
