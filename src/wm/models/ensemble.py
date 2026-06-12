"""Ensemble: blend GBM classifier, backprop neural net, and Poisson scorelines."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson

from wm.models.calibrate import WDLCalibrator
from wm.models.neural_net import MLPClassifier


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


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_blend_weights(prob_stack: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    """
    Find convex blend weights over the branch probability matrices that
    minimize log-loss on validation data. Parametrized through a softmax so
    the weights stay on the simplex.
    """
    from scipy.optimize import minimize

    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    idx = np.arange(n)

    def nll(theta: np.ndarray) -> float:
        w = _softmax(theta)
        p = sum(wi * P for wi, P in zip(w, prob_stack))
        return -float(np.mean(np.log(p[idx, labels] + 1e-12)))

    res = minimize(nll, np.zeros(len(prob_stack)), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
    return _softmax(res.x)


class MatchPredictor:
    """
    Blended match predictor combining:
      1. LightGBM W/D/L classifier (gradient boosting)
      2. Backprop neural network W/D/L classifier (optional)
      3. LightGBM Poisson goal regressors → scoreline grid
    Branch probabilities are mixed with validation-optimized convex weights,
    then temperature-calibrated.
    """

    def __init__(
        self,
        clf: lgb.Booster,
        goals_home_model: lgb.Booster,
        goals_away_model: lgb.Booster,
        calibrator: WDLCalibrator,
        nn: MLPClassifier | None = None,
        blend_weights: np.ndarray | None = None,
        blend_weight: float = 0.5,  # legacy 2-branch fallback
        max_goals: int = 10,
    ):
        self.clf = clf
        self.goals_home = goals_home_model
        self.goals_away = goals_away_model
        self.calibrator = calibrator
        self.nn = nn
        self.blend_weights = (
            np.asarray(blend_weights, dtype=float) if blend_weights is not None else None
        )
        self.blend_weight = blend_weight
        self.max_goals = max_goals

    def branch_probs(self, df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Return ([clf, (nn), poisson] prob matrices, lambdas home/away, grids)."""
        from wm.models.wdl_classifier import predict_proba
        from wm.models.goals_gbm import predict_lambdas

        clf_probs = predict_proba(self.clf, df)  # (N, 3): [away, draw, home]

        lh, la = predict_lambdas(self.goals_home, self.goals_away, df)
        lh = np.clip(lh, 0.05, 8.0)
        la = np.clip(la, 0.05, 8.0)
        grids = np.array([
            independent_poisson_grid(lh[i], la[i], self.max_goals)
            for i in range(len(df))
        ])
        poisson_wdl = np.array([scoreline_grid_to_wdl(grids[i]) for i in range(len(df))])

        branches = [clf_probs]
        if self.nn is not None:
            branches.append(self.nn.predict_proba(df))
        branches.append(poisson_wdl)
        return branches, lh, la, grids

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Returns dict with keys:
          p_home_win, p_draw, p_away_win  – calibrated blended probs
          lambda_home, lambda_away        – expected goals
          score_grid                      – (N, max_g+1, max_g+1) scoreline grid
        """
        branches, lh, la, grids = self.branch_probs(df)

        if self.blend_weights is not None and len(self.blend_weights) == len(branches):
            blended = sum(w * P for w, P in zip(self.blend_weights, branches))
        else:
            # legacy fallback: clf vs poisson at blend_weight
            w = self.blend_weight
            blended = w * branches[0] + (1 - w) * branches[-1]
        blended = np.clip(blended, 1e-9, 1.0)
        blended /= blended.sum(axis=1, keepdims=True)

        calibrated = self.calibrator.predict(blended)

        return {
            "p_away_win": calibrated[:, 0],
            "p_draw": calibrated[:, 1],
            "p_home_win": calibrated[:, 2],
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
