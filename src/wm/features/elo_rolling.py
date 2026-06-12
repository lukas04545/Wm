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


def compute(matches: pd.DataFrame, cfg: EloConfig) -> pd.DataFrame:
    """
    Compute rolling Elo for every match, returning columns:
      match_id, date, home_team, away_team,
      elo_home_before, elo_away_before,
      elo_home_after, elo_away_after
    """
    ratings: dict[str, float] = defaultdict(lambda: cfg.initial)
    records = []

    for _, row in matches.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("is_neutral", False))
        tier = int(row.get("tournament_tier", 2))
        gd = int(row.get("goal_diff", 0))
        outcome = int(row.get("outcome", 0))  # +1 home win, 0 draw, -1 away win

        r_h = ratings[home]
        r_a = ratings[away]
        exp_h = _expected(r_h, r_a, neutral, cfg.home_advantage)

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
        })

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    result = pd.DataFrame(records)
    result["elo_diff_before"] = result["elo_home_before"] - result["elo_away_before"]
    result["elo_expected_home"] = result.apply(
        lambda r: _expected(r["elo_home_before"], r["elo_away_before"], False, cfg.home_advantage),
        axis=1,
    )
    return result, dict(ratings)


def get_current_ratings(matches: pd.DataFrame, cfg: EloConfig) -> dict[str, float]:
    """Return final Elo ratings after processing all matches."""
    _, ratings = compute(matches, cfg)
    return ratings
