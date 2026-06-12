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
    for home, away, city in bracket:
        teams_in_bracket.add(home)
        teams_in_bracket.add(away)
        assert isinstance(city, str) and city  # real venue attached
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


def venue_aware_predict_fn(home, away, neutral=True, city=None):
    """Accepts the live-mode kwargs; home always wins 1-0."""
    grid = np.zeros((11, 11))
    grid[1, 0] = 1.0
    return grid, (1.0, 0.5)


def test_group_stage_locks_in_played_results():
    """Played fixtures must contribute their REAL score, not a simulated one."""
    rng = np.random.default_rng(0)
    fixtures = []
    for grp, teams in GROUPS.items():
        import itertools
        for h, a in itertools.combinations(teams, 2):
            fixtures.append({
                "group": grp, "home": h, "away": a, "city": "Houston",
                "neutral": True, "played": False,
                "goals_home": None, "goals_away": None,
            })
    # Lock in a real result that contradicts the predictor: South Africa 5-0 Mexico
    for fx in fixtures:
        if fx["home"] == "Mexico" and fx["away"] == "South Africa":
            fx.update(played=True, goals_home=0, goals_away=5)

    results = simulate_group_stage(GROUPS, venue_aware_predict_fn, rng, fixtures=fixtures)
    rec = {r.team: r for r in results["A"].teams}
    # South Africa got the locked-in 3 points and +5 GD from the real result
    assert rec["South Africa"].pts >= 3
    assert rec["South Africa"].gf >= 5
    assert rec["Mexico"].ga >= 5


def test_knockout_respects_played_result():
    rng = np.random.default_rng(0)
    played = {frozenset({"Brazil", "Argentina"}): "Argentina"}
    # predictor says home (Brazil) always wins — but the real result rules
    winner = simulate_knockout_match("Brazil", "Argentina", venue_aware_predict_fn,
                                     rng, played=played)
    assert winner == "Argentina"


def test_real_bracket_structure():
    """The encoded R32 list must reproduce FIFA's official bracket flow."""
    from wm.sim.tournament import R32_MATCHES

    assert len(R32_MATCHES) == 16
    match_by_no = {m[3]: m for m in R32_MATCHES}
    # Official R16 pairings: 89=(74,77) 90=(73,75) 93=(83,84) 94=(81,82)
    #                        91=(76,78) 92=(79,80) 95=(86,88) 96=(85,87)
    # Sequential pairing means positions (0,1),(2,3)... must be those pairs,
    # ordered so QF/SF flow also matches: [74,77,73,75,83,84,81,82,76,78,79,80,86,88,85,87]
    expected_order = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
    assert [m[3] for m in R32_MATCHES] == expected_order
    # Spot-check official slot compositions
    assert match_by_no[73][:2] == ("A2", "B2")
    assert match_by_no[76][:2] == ("C1", "F2")
    assert match_by_no[79][0] == "A1" and match_by_no[79][2] == "Mexico City"
    assert match_by_no[84][:2] == ("H1", "J2")
    assert match_by_no[88][:2] == ("D2", "G2")


def test_third_place_assignment_respects_allowed_sets():
    from wm.sim.tournament import assign_third_place_slots, R32_MATCHES

    allowed = {m[1]: set(m[1].split(":")[1]) for m in R32_MATCHES if m[1].startswith("3rd:")}

    # Try several realistic 8-group combinations
    for combo in [list("ABCDEFGH"), list("EFGHIJKL"), list("ACDFHIJL"), list("BCEFGIJK")]:
        assignment = assign_third_place_slots(combo)
        assert len(assignment) == 8
        assert sorted(assignment.values()) == sorted(combo)  # each third used once
        for slot, group in assignment.items():
            assert group in allowed[slot], f"{group} not allowed in {slot} for {combo}"
