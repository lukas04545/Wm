"""
Dixon-Coles bivariate Poisson model for scoreline probabilities.

Implements:
  - Maximum likelihood estimation of attack/defence strengths per team
  - Time-decay weighting (half-life ~3 years for internationals)
  - Low-score dependence correction (τ parameter)
  - Scoreline grid P(home=i, away=j) for i,j in 0..max_g
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


class DixonColes:
    """Dixon-Coles bivariate Poisson with time-decay weights."""

    def __init__(self, xi: float = 0.002, max_goals: int = 10):
        self.xi = xi  # time-decay parameter (per day)
        self.max_goals = max_goals
        self.attack_: dict[str, float] = {}
        self.defence_: dict[str, float] = {}
        self.home_advantage_: float = 0.0
        self.rho_: float = 0.0
        self.teams_: list[str] = []

    def _weights(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        days = (ref_date - dates).dt.days.values.astype(float)
        return np.exp(-self.xi * days)

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        """Fit on matches DataFrame with columns: date, home_team, away_team, goals_home, goals_away."""
        matches = matches.dropna(subset=["goals_home", "goals_away"]).copy()
        ref_date = matches["date"].max()
        weights = self._weights(matches["date"], ref_date)

        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Initial params: attack=1.0, defence=1.0, home_adv=0.1, rho=-0.1
        x0 = np.ones(2 * n + 2)
        x0[2 * n] = 0.1    # home advantage (log scale)
        x0[2 * n + 1] = -0.1  # rho

        def neg_log_likelihood(params: np.ndarray) -> float:
            attack = np.exp(params[:n])
            defence = np.exp(params[n:2*n])
            home_adv = np.exp(params[2*n])
            rho = params[2*n + 1]

            ll = 0.0
            for i, row in enumerate(matches.itertuples()):
                h_idx = idx.get(row.home_team)
                a_idx = idx.get(row.away_team)
                if h_idx is None or a_idx is None:
                    continue
                lh = attack[h_idx] * defence[a_idx] * home_adv
                la = attack[a_idx] * defence[h_idx]
                gh = int(row.goals_home)
                ga = int(row.goals_away)
                tau = _tau(gh, ga, lh, la, rho)
                if tau <= 0:
                    tau = 1e-10
                ll_i = (
                    math.log(tau + 1e-10)
                    + math.log(_poisson_pmf(gh, lh) + 1e-10)
                    + math.log(_poisson_pmf(ga, la) + 1e-10)
                )
                ll += weights[i] * ll_i

            return -ll

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          options={"maxiter": 200, "ftol": 1e-9})

        params = result.x
        attack = np.exp(params[:n])
        defence = np.exp(params[n:2*n])
        self.home_advantage_ = np.exp(params[2*n])
        self.rho_ = params[2*n + 1]

        # Normalize so mean attack = 1.0
        mean_att = attack.mean()
        attack /= mean_att
        defence *= mean_att

        self.attack_ = dict(zip(teams, attack))
        self.defence_ = dict(zip(teams, defence))
        return self

    def predict_grid(
        self,
        home: str,
        away: str,
        neutral: bool = False,
        max_goals: int | None = None,
    ) -> np.ndarray:
        """Return (max_g+1, max_g+1) probability grid P(home=i, away=j)."""
        max_g = max_goals or self.max_goals
        att_h = self.attack_.get(home, 1.0)
        def_h = self.defence_.get(home, 1.0)
        att_a = self.attack_.get(away, 1.0)
        def_a = self.defence_.get(away, 1.0)
        ha = 1.0 if neutral else self.home_advantage_
        lh = att_h * def_a * ha
        la = att_a * def_h

        grid = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                p = _poisson_pmf(i, lh) * _poisson_pmf(j, la)
                if i <= 1 and j <= 1:
                    p *= _tau(i, j, lh, la, self.rho_)
                grid[i, j] = p

        grid = np.clip(grid, 0, None)
        grid /= grid.sum()
        return grid

    def predict_wdl(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """Return [p_away_win, p_draw, p_home_win]."""
        grid = self.predict_grid(home, away, neutral)
        ph = float(np.tril(grid, -1).sum())
        pd_ = float(np.diag(grid).sum())
        pa = float(np.triu(grid, 1).sum())
        total = ph + pd_ + pa
        return np.array([pa / total, pd_ / total, ph / total])
