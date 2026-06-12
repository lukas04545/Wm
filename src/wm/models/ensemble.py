"""Ensemble: blend W/D/L classifier with scoreline-derived probabilities."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

from wm.models.calibrate import WDLCalibrator
from wm.models.goals_poisson import DixonColes


def scoreline_grid_to_wdl(grid: np.ndarray) -> np.ndarray:
    """Convert (G+1, G+1) scoreline grid to [p_away, p_draw, p_home]."""
    pa = float(np.triu(grid, 1).sum())
    pd_ = float(np.diag(grid).sum())
    ph = float(np.tril(grid, -1).sum())
    total = pa + pd_ + ph
    if total == 0:
        return np.array([1 / 3, 1 / 3, 1 / 3])
    return np.array([pa / total, pd_ / total, ph / total])


def independent_poisson_grid(lh: float, la: float, max_g: int = 10) -> np.ndarray:
    """P(home=i, away=j) for independent Poisson(lh) × Poisson(la)."""
    grid = np.outer(
        poisson.pmf(range(max_g + 1), lh),
        poisson.pmf(range(max_g + 1), la),
    )
    grid /= grid.sum()
    return grid


class MatchPredictor:
    """
    Blended match predictor combining:
      1. LightGBM W/D/L classifier
      2. LightGBM Poisson goal regressors → scoreline grid
      3. Temperature-scaled calibration
    """

    def __init__(
        self,
        clf: lgb.Booster,
        goals_home_model: lgb.Booster,
        goals_away_model: lgb.Booster,
        calibrator: WDLCalibrator,
        blend_weight: float = 0.5,
        max_goals: int = 10,
    ):
        self.clf = clf
        self.goals_home = goals_home_model
        self.goals_away = goals_away_model
        self.calibrator = calibrator
        self.blend_weight = blend_weight
        self.max_goals = max_goals

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Returns dict with keys:
          p_home_win, p_draw, p_away_win  – calibrated blended probs
          lambda_home, lambda_away        – expected goals
          score_grid                      – (N, max_g+1, max_g+1) scoreline grid
        """
        from wm.models.wdl_classifier import predict_proba
        from wm.models.goals_gbm import predict_lambdas

        # Classifier branch
        clf_probs = predict_proba(self.clf, df)  # (N, 3): [away, draw, home]
        clf_cal = self.calibrator.predict(clf_probs)

        # Poisson branch
        lh, la = predict_lambdas(self.goals_home, self.goals_away, df)
        lh = np.clip(lh, 0.05, 8.0)
        la = np.clip(la, 0.05, 8.0)

        grids = np.array([
            independent_poisson_grid(lh[i], la[i], self.max_goals)
            for i in range(len(df))
        ])
        poisson_wdl = np.array([scoreline_grid_to_wdl(grids[i]) for i in range(len(df))])

        # Blend
        w = self.blend_weight
        blended = w * clf_cal + (1 - w) * poisson_wdl
        blended /= blended.sum(axis=1, keepdims=True)

        return {
            "p_away_win": blended[:, 0],
            "p_draw": blended[:, 1],
            "p_home_win": blended[:, 2],
            "lambda_home": lh,
            "lambda_away": la,
            "score_grid": grids,
        }

    def predict_single(
        self,
        home: str,
        away: str,
        feature_row: pd.DataFrame,
    ) -> dict[str, float | np.ndarray]:
        result = self.predict(feature_row)
        return {
            "home": home,
            "away": away,
            "p_home_win": float(result["p_home_win"][0]),
            "p_draw": float(result["p_draw"][0]),
            "p_away_win": float(result["p_away_win"][0]),
            "lambda_home": float(result["lambda_home"][0]),
            "lambda_away": float(result["lambda_away"][0]),
            "score_grid": result["score_grid"][0],
        }
