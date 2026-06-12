"""Time-based train/val/test splits (strictly no leakage)."""
from __future__ import annotations

import pandas as pd

from wm.config import SplitsConfig


def split(df: pd.DataFrame, cfg: SplitsConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames based on date."""
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)
    date = df["date"] if "date" in df.columns else df.index
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test
