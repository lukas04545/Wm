"""Tests for 80/20 time-ordered split."""
import pandas as pd
import numpy as np
import pytest

from wm.config import SplitsConfig
from wm.eval.splits import split, split_by_date


def make_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=n, freq="7D")
    return pd.DataFrame({
        "date": dates,
        "home_team": ["A"] * n,
        "away_team": ["B"] * n,
        "goals_home": np.random.randint(0, 4, n),
        "goals_away": np.random.randint(0, 4, n),
        "label_wdl": np.random.randint(0, 3, n),
    })


def test_80_20_ratio():
    df = make_df(100)
    cfg = SplitsConfig(train_ratio=0.8)
    train, val, _ = split(df, cfg)
    assert len(train) == 80
    assert len(val) == 20


def test_ratio_sums_to_total():
    df = make_df(200)
    cfg = SplitsConfig(train_ratio=0.8)
    train, val, _ = split(df, cfg)
    assert len(train) + len(val) == len(df)


def test_temporal_ordering_preserved():
    """Training set must contain only earlier dates than validation set."""
    df = make_df(100)
    cfg = SplitsConfig(train_ratio=0.8)
    train, val, _ = split(df, cfg)
    assert train["date"].max() <= val["date"].min()


def test_no_overlap():
    df = make_df(100)
    cfg = SplitsConfig(train_ratio=0.8)
    train, val, _ = split(df, cfg)
    assert len(set(train.index) & set(val.index)) == 0


def test_non_integer_sizes():
    """With 101 rows and 0.8 ratio: 80 train, 21 val."""
    df = make_df(101)
    cfg = SplitsConfig(train_ratio=0.8)
    train, val, _ = split(df, cfg)
    assert len(train) == 80
    assert len(val) == 21
    assert len(train) + len(val) == 101


def test_default_config_is_80_20():
    from wm.config import load
    cfg = load()
    assert cfg.splits.train_ratio == pytest.approx(0.8)
