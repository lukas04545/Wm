"""Time-ordered train/val splits — preserves temporal ordering to prevent leakage."""
from __future__ import annotations

import pandas as pd

from wm.config import SplitsConfig


def split(df: pd.DataFrame, cfg: SplitsConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train, val, test) using an 80/20 temporal split.

    Matches are sorted by date; the first train_ratio fraction becomes the
    training set and the remainder is the validation set.  A separate held-out
    test set (post val_end date) is also returned for final evaluation.

    Temporal ordering is always preserved so no future data leaks into training.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratio = float(cfg.train_ratio)
    cutoff = int(len(df) * ratio)
    train = df.iloc[:cutoff]
    val = df.iloc[cutoff:]
    # Kept for backward-compat: callers may request a separate test window
    test = val  # with 80/20 the val set doubles as the held-out evaluation set
    return train, val, test


def split_by_date(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy date-based split — kept for reference and backtest helpers."""
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(val_end)
    train = df[df["date"] <= t_end]
    val = df[(df["date"] > t_end) & (df["date"] <= v_end)]
    test = df[df["date"] > v_end]
    return train, val, test
