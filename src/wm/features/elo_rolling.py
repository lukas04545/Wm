"""
Self-computed rolling Elo ratings (leak-free, computed forward in time).

Standard international Elo with:
  - K scaled by tournament tier
  - Goal-difference multiplier
  - Home advantage bonus (+100) on non-neutral venues
  - Starting rating 1500 for all teams first appearance
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from wm.config import EloConfig


def _k(tier: int, cfg: EloConfig) -> float:
    if tier >= 5:
        return cfg.k_wc
    if tier >= 4:
        return cfg.k_continental
    if tier >= 3:
        return cfg.k_qualifier
    return cfg.k_friendly


def _gd_multiplier(gd: int) -> float:
    """Goal-difference multiplier used in World Football Elo."""
    abs_gd = abs(gd)
    if abs_gd <= 1:
        return 1.0
    if abs_gd == 2:
        return 1.5
    return (11 + abs_gd) / 8.0


def _expected(elo_home: float, elo_away: float, is_neutral: bool, home_adv: float) -> float:
    dr = elo_home - elo_away + (0 if is_neutral else home_adv)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def compute(matches: pd.DataFrame, cfg: EloConfig) -> tuple[pd.DataFrame, dict]:
    """
    Compute rolling Elo plus attack/defence goal ratings for every match.

    Returns (per-match DataFrame, final_state) where final_state is
    {"elo": {team: rating}, "att": {team: log-attack}, "def": {team: log-defence}}.

    Attack/defence ratings (log scale, start 0.0) are updated by the Poisson
    log-likelihood gradient (goals − expected goals), giving the goal
    regressors a per-team scoring/conceding signal that a single Elo scalar
    cannot carry.
    """
    ratings: dict[str, float] = defaultdict(lambda: cfg.initial)
    att: dict[str, float] = defaultdict(float)
    dfc: dict[str, float] = defaultdict(float)
    records = []

    BASE_HOME, BASE_AWAY, BASE_NEUTRAL = math.log(1.45), math.log(1.15), math.log(1.30)
    KG_BASE = 0.03
    RATING_CLAMP = 1.5

    for _, row in matches.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("is_neutral", False))
        tier = int(row.get("tournament_tier", 2))
        gd = int(row.get("goal_diff", 0))
        gh = int(row.get("goals_home", 0))
        ga = int(row.get("goals_away", 0))
        outcome = int(row.get("outcome", 0))  # +1 home win, 0 draw, -1 away win

        r_h = ratings[home]
        r_a = ratings[away]
        exp_h = _expected(r_h, r_a, neutral, cfg.home_advantage)

        a_h, d_h = att[home], dfc[home]
        a_a, d_a = att[away], dfc[away]
        base_h = BASE_NEUTRAL if neutral else BASE_HOME
        base_a = BASE_NEUTRAL if neutral else BASE_AWAY
        exp_gh = math.exp(base_h + a_h + d_a)
        exp_ga = math.exp(base_a + a_a + d_h)

        # Actual score for home team (1, 0.5, 0)
        if outcome > 0:
            actual_h = 1.0
        elif outcome < 0:
            actual_h = 0.0
        else:
            actual_h = 0.5

        k = _k(tier, cfg)
        mult = _gd_multiplier(gd)
        delta = k * mult * (actual_h - exp_h)

        records.append({
            "match_id": row.get("match_id", len(records)),
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "elo_home_before": r_h,
            "elo_away_before": r_a,
            "elo_home_after": r_h + delta,
            "elo_away_after": r_a - delta,
            "att_home_before": a_h,
            "def_home_before": d_h,
            "att_away_before": a_a,
            "def_away_before": d_a,
        })

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

        # Attack/defence update: gradient of Poisson log-likelihood w.r.t.
        # log-rate is (observed − expected). High def = leaky defence.
        kg = KG_BASE * (k / 40.0)
        att[home] = float(np.clip(a_h + kg * (gh - exp_gh), -RATING_CLAMP, RATING_CLAMP))
        dfc[away] = float(np.clip(d_a + kg * (gh - exp_gh), -RATING_CLAMP, RATING_CLAMP))
        att[away] = float(np.clip(a_a + kg * (ga - exp_ga), -RATING_CLAMP, RATING_CLAMP))
        dfc[home] = float(np.clip(d_h + kg * (ga - exp_ga), -RATING_CLAMP, RATING_CLAMP))

    result = pd.DataFrame(records)
    result["elo_diff_before"] = result["elo_home_before"] - result["elo_away_before"]
    result["elo_expected_home"] = result.apply(
        lambda r: _expected(r["elo_home_before"], r["elo_away_before"], False, cfg.home_advantage),
        axis=1,
    )
    result["att_diff_before"] = result["att_home_before"] - result["att_away_before"]
    result["def_diff_before"] = result["def_home_before"] - result["def_away_before"]
    state = {"elo": dict(ratings), "att": dict(att), "def": dict(dfc)}
    return result, state


def get_current_ratings(matches: pd.DataFrame, cfg: EloConfig) -> dict[str, float]:
    """Return final Elo ratings after processing all matches."""
    _, state = compute(matches, cfg)
    return state["elo"]


def get_current_state(matches: pd.DataFrame, cfg: EloConfig) -> dict:
    """Return final {'elo', 'att', 'def'} rating dicts after all matches."""
    _, state = compute(matches, cfg)
    return state
