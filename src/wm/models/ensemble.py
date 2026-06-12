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


# factorials 0..30 — enough for any realistic goal grid
_FACTORIALS = np.cumprod(np.concatenate([[1.0], np.arange(1.0, 31.0)]))


def _poisson_pmf_vec(max_g: int, lam: float) -> np.ndarray:
    """Closed-form Poisson pmf vector — ~30x faster than scipy.stats in the
    simulation hot loop (520k+ grid builds per 5k-run simulation)."""
    k = np.arange(max_g + 1)
    return np.exp(-lam) * lam ** k / _FACTORIALS[: max_g + 1]


def independent_poisson_grid(
    lh: float, la: float, max_g: int = 10, rho: float = 0.0
) -> np.ndarray:
    """
    P(home=i, away=j) for independent Poisson(lh) × Poisson(la), optionally
    with the Dixon-Coles low-score dependence correction τ applied to the
    {0-0, 1-0, 0-1, 1-1} cells (negative rho inflates draws — the empirically
    observed pattern that independent Poissons miss).
    """
    grid = np.outer(_poisson_pmf_vec(max_g, lh), _poisson_pmf_vec(max_g, la))
    if rho != 0.0:
        from wm.models.goals_poisson import _tau
        for i in (0, 1):
            for j in (0, 1):
                grid[i, j] *= max(_tau(i, j, lh, la, rho), 1e-10)
    grid /= grid.sum()
    return grid


def fit_dc_rho(
    lambdas_home: np.ndarray,
    lambdas_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
) -> float:
    """
    Fit the Dixon-Coles low-score correction rho on validation scorelines by
    maximizing the corrected Poisson likelihood (1-D bounded search).
    """
    from scipy.optimize import minimize_scalar
    from wm.models.goals_poisson import _tau

    gh = goals_home.astype(int)
    ga = goals_away.astype(int)
    base = poisson.pmf(gh, lambdas_home) * poisson.pmf(ga, lambdas_away)
    low = (gh <= 1) & (ga <= 1)

    def nll(rho: float) -> float:
        tau = np.ones(len(gh))
        idx = np.where(low)[0]
        for i in idx:
            tau[i] = max(_tau(int(gh[i]), int(ga[i]), lambdas_home[i], lambdas_away[i], rho), 1e-10)
        return -float(np.mean(np.log(base * tau + 1e-12)))

    res = minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded")
    return float(res.x)


_REGION_MASKS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _region_masks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n not in _REGION_MASKS:
        ones = np.ones((n, n))
        _REGION_MASKS[n] = (np.triu(ones, 1), np.eye(n), np.tril(ones, -1))
    return _REGION_MASKS[n]


def reshape_grid_to_wdl(grid: np.ndarray, target_wdl: np.ndarray) -> np.ndarray:
    """
    Rescale a scoreline grid so its win/draw/loss marginals equal target_wdl
    ([p_away, p_draw, p_home]) while preserving the within-region shape of the
    scoreline distribution. This is how the simulator injects the strong,
    calibrated ensemble outcome probabilities into the Poisson-shaped grid —
    so the neural net and GBM affect sampled scorelines (and championship odds),
    not just the headline W/D/L numbers.
    """
    n = grid.shape[0]
    away_mask, draw_mask, home_mask = _region_masks(n)

    cur_away = float((grid * away_mask).sum())
    cur_draw = float((grid * draw_mask).sum())
    cur_home = float((grid * home_mask).sum())

    scale = np.ones((n, n))
    if cur_away > 1e-12:
        scale = np.where(away_mask > 0, target_wdl[0] / cur_away, scale)
    if cur_draw > 1e-12:
        scale = np.where(draw_mask > 0, target_wdl[1] / cur_draw, scale)
    if cur_home > 1e-12:
        scale = np.where(home_mask > 0, target_wdl[2] / cur_home, scale)

    out = grid * scale
    s = out.sum()
    return out / s if s > 0 else grid


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


STACKER_CONTEXT_COLS = ["elo_diff_before", "tournament_tier", "is_neutral"]


class StackedBlender:
    """
    Meta-learner over branch probabilities: multinomial logistic regression on
    the log-probs of every branch plus match context (Elo gap, tournament
    tier, neutrality). Unlike a fixed convex blend, it can route between
    branches per match — e.g. trust the Poisson branch more in lopsided
    matches and the NN more in close ones.
    """

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=2000, C=C)

    @staticmethod
    def _features(branches: list[np.ndarray], context_df: pd.DataFrame) -> np.ndarray:
        cols = [np.log(np.clip(P, 1e-9, 1.0)) for P in branches]
        ctx = context_df.reindex(columns=STACKER_CONTEXT_COLS)
        ctx_arr = ctx.to_numpy(dtype=float)
        ctx_arr = np.where(np.isnan(ctx_arr), 0.0, ctx_arr)
        ctx_arr[:, 0] = ctx_arr[:, 0] / 400.0   # elo_diff scale
        ctx_arr[:, 1] = ctx_arr[:, 1] / 5.0     # tier scale
        return np.column_stack(cols + [ctx_arr])

    def fit(self, branches: list[np.ndarray], context_df: pd.DataFrame, labels: np.ndarray) -> "StackedBlender":
        X = self._features(branches, context_df)
        self.lr.fit(X, np.asarray(labels, dtype=int))
        return self

    def predict_proba(self, branches: list[np.ndarray], context_df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self._features(branches, context_df))


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
        dc_rho: float = 0.0,
        stacker: "StackedBlender | None" = None,
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
        self.dc_rho = dc_rho
        self.stacker = stacker

    def branch_probs(self, df: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Return ([clf, (nn), poisson] prob matrices, lambdas home/away, grids)."""
        from wm.models.wdl_classifier import predict_proba
        from wm.models.goals_gbm import predict_lambdas

        clf_probs = predict_proba(self.clf, df)  # (N, 3): [away, draw, home]

        lh, la = predict_lambdas(self.goals_home, self.goals_away, df)
        lh = np.clip(lh, 0.05, 8.0)
        la = np.clip(la, 0.05, 8.0)
        grids = np.array([
            independent_poisson_grid(lh[i], la[i], self.max_goals, rho=self.dc_rho)
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

        if self.stacker is not None:
            blended = self.stacker.predict_proba(branches, df)
        elif self.blend_weights is not None and len(self.blend_weights) == len(branches):
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
