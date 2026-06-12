"""Tests for Elo rating computation."""
import pandas as pd
import pytest
from wm.config import EloConfig
from wm.features.elo_rolling import compute, _expected, _gd_multiplier


def make_match(home, away, gh, ga, neutral=False, tier=5):
    return {
        "match_id": 0,
        "date": pd.Timestamp("2022-01-01"),
        "home_team": home,
        "away_team": away,
        "goals_home": gh,
        "goals_away": ga,
        "goal_diff": gh - ga,
        "outcome": 1 if gh > ga else (0 if gh == ga else -1),
        "is_neutral": neutral,
        "tournament_tier": tier,
    }


def test_expected_probability():
    cfg = EloConfig()
    # Equal ratings → 50/50 on neutral
    p = _expected(1500, 1500, True, cfg.home_advantage)
    assert abs(p - 0.5) < 1e-9

    # Higher rating → higher expected
    p_high = _expected(1600, 1500, True, cfg.home_advantage)
    assert p_high > 0.5


def test_gd_multiplier():
    assert _gd_multiplier(0) == 1.0
    assert _gd_multiplier(1) == 1.0
    assert _gd_multiplier(2) == 1.5
    assert _gd_multiplier(3) > 1.5


def test_elo_update_home_win():
    cfg = EloConfig()
    matches = pd.DataFrame([make_match("TeamA", "TeamB", 2, 0, neutral=True)])
    elo_df, state = compute(matches, cfg)
    ratings = state["elo"]
    # Winner should gain rating
    assert ratings["TeamA"] > cfg.initial
    assert ratings["TeamB"] < cfg.initial


def test_elo_update_draw():
    cfg = EloConfig()
    matches = pd.DataFrame([make_match("TeamA", "TeamB", 1, 1, neutral=True)])
    elo_df, state = compute(matches, cfg)
    ratings = state["elo"]
    # Draw: equal teams → no change
    assert abs(ratings["TeamA"] - cfg.initial) < 1e-6
    assert abs(ratings["TeamB"] - cfg.initial) < 1e-6


def test_elo_conservation():
    """Total Elo should be conserved (sum is constant after each match)."""
    cfg = EloConfig()
    matches = pd.DataFrame([
        make_match("A", "B", 3, 0, neutral=True),
        make_match("C", "D", 1, 2, neutral=True),
        make_match("A", "C", 1, 1, neutral=True),
    ])
    matches["match_id"] = range(len(matches))
    _, state = compute(matches, cfg)
    ratings = state["elo"]
    # Sum of all ratings = initial * n_teams (conservation)
    total = sum(ratings.values())
    # 4 teams × 1500 = 6000
    assert abs(total - 4 * cfg.initial) < 1e-3


def test_home_advantage():
    cfg = EloConfig()
    # Home team with same rating gets higher expected score
    p_home = _expected(1500, 1500, False, cfg.home_advantage)
    p_neutral = _expected(1500, 1500, True, cfg.home_advantage)
    assert p_home > p_neutral
