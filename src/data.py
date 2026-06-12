"""Data loading for the Titanic classification task.

Local-first, Kaggle-fallback. Reads ``data/train.csv`` if present; otherwise
downloads the competition data via the Kaggle API. Only ``train.csv`` is used
per the assignment (test.csv / gender_submission.csv are ignored).
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
KAGGLE_COMPETITION = "titanic"


def _download_from_kaggle(dest_dir: Path) -> None:
    """Download and extract the Titanic competition files via the Kaggle API.

    Requires Kaggle credentials (``~/.kaggle/kaggle.json`` or the
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` env vars). See README for setup.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "kaggle package not installed. Run `pip install kaggle` or place "
            f"train.csv manually at {TRAIN_CSV}."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(KAGGLE_COMPETITION, path=str(dest_dir), quiet=False)

    zip_path = dest_dir / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Return the raw Titanic training frame.

    Parameters
    ----------
    path : optional
        Explicit path to a Titanic ``train.csv``. If given, it is read directly
        and no download is attempted (used by the Streamlit inference UI).
    """
    if path is not None:
        return pd.read_csv(path)

    if not TRAIN_CSV.exists():
        _download_from_kaggle(DATA_DIR)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Could not find or download train.csv at {TRAIN_CSV}."
        )

    return pd.read_csv(TRAIN_CSV)


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols from {TRAIN_CSV}")
