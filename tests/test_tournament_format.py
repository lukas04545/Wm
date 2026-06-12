"""Tests for tournament format engine."""
import numpy as np
import pandas as pd
import pytest
from wm.sim.tournament import (
    TeamRecord,
    GroupResult,
    simulate_group_stage,
    get_third_place_ranking,
    build_r32_bracket,
    simulate_knockout_match,
)

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def dummy_predict_fn(home, away, neutral=True):
    """Deterministic: home team always scores 1-0."""
    rng = np.random.default_rng(42)
    grid = np.zeros((11, 11))
    grid[1, 0] = 1.0  # home scores 1, away scores 0
    return grid, (1.0, 0.5)


def test_group_stage_produces_all_groups():
    rng = np.random.default_rng(0)
    results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
    assert len(results) == 12
    for grp in "ABCDEFGHIJKL":
        assert grp in results


def test_group_stage_correct_team_count():
    rng = np.random.default_rng(0)
    results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
    for grp, gr in results.items():
        assert len(gr.teams) == 4


def test_group_points_sum():
    """Each group should have 3 points distributed per match (no draws → 3*6/3 = 6 pts per team across group)."""
    rng = np.random.default_rng(0)
    results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
    for grp, gr in results.items():
        total_pts = sum(r.pts for r in gr.teams)
        # 6 matches per group, each gives 3 points → 18 total
        assert total_pts == 18, f"Group {grp}: {total_pts} points"


def test_third_place_ranking_size():
    rng = np.random.default_rng(0)
    results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
    thirds = get_third_place_ranking(results)
    assert len(thirds) == 12  # 12 groups, 12 third-place teams


def test_r32_bracket_has_32_teams():
    rng = np.random.default_rng(0)
    results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
    thirds = get_third_place_ranking(results)
    bracket = build_r32_bracket(results, thirds)
    assert len(bracket) == 16  # 16 matches = 32 teams
    teams_in_bracket = set()
    for home, away in bracket:
        teams_in_bracket.add(home)
        teams_in_bracket.add(away)
    assert len(teams_in_bracket) == 32


def test_knockout_produces_winner():
    rng = np.random.default_rng(42)
    winner = simulate_knockout_match("Brazil", "Argentina", dummy_predict_fn, rng)
    assert winner in ("Brazil", "Argentina")


def test_group_standings_sorted():
    records = [
        TeamRecord("A", "X", pts=7, gf=5, ga=2, gd=3),
        TeamRecord("B", "X", pts=4, gf=3, ga=3, gd=0),
        TeamRecord("C", "X", pts=7, gf=6, ga=2, gd=4),
        TeamRecord("D", "X", pts=1, gf=1, ga=5, gd=-4),
    ]
    gr = GroupResult("X", records)
    standings = gr.standings()
    assert standings[0].team == "C"  # 7 pts, GD 4
    assert standings[1].team == "A"  # 7 pts, GD 3
    assert standings[2].team == "B"
    assert standings[3].team == "D"
