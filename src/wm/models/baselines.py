"""Simple baseline models to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wm.features.matrix import TARGET_WDL


class EloBaseline:
    """Elo-only logistic regression baseline."""

    def __init__(self):
        # multinomial is the default for lbfgs; the multi_class kwarg was removed in sklearn 1.7
        self.lr = LogisticRegression(max_iter=1000, C=1.0)
        self.draw_rate_: float = 0.25

    def fit(self, df: pd.DataFrame) -> "EloBaseline":
        X = df[["elo_diff_before", "elo_expected_home", "is_neutral"]].fillna(0).values
        y = df[TARGET_WDL].astype(int).values
        self.lr.fit(X, y)
        self.draw_rate_ = (y == 1).mean()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[["elo_diff_before", "elo_expected_home", "is_neutral"]].fillna(0).values
        return self.lr.predict_proba(X)  # (N, 3)


class DrawRateBaseline:
    """Always predicts the historical W/D/L rates."""

    def __init__(self):
        self.rates_ = np.array([1 / 3, 1 / 3, 1 / 3])

    def fit(self, df: pd.DataFrame) -> "DrawRateBaseline":
        y = df[TARGET_WDL].astype(int).values
        rates = np.bincount(y, minlength=3).astype(float)
        self.rates_ = rates / rates.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.rates_, (len(df), 1))
