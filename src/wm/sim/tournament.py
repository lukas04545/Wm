"""
2026 FIFA World Cup tournament format engine.

Groups A-L → top 2 + 8 best thirds → Round of 32 → R16 → QF → SF → Final
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from wm import config as cfg_mod

# All possible 3rd-place group combinations (12 groups, choose 8)
# FIFA pre-defines which R32 slots the 8 qualifiers fill for each combo.
# This table maps frozenset of group letters → list of (slot_a_source, slot_b_source)
# where source = "A1", "B2", "3rd_X" etc.
# Full mapping from FIFA 2026 tournament regulations Annex C.
# We implement a simplified but correct version where the bracket is determined
# by which 8 groups produced third-placers.

GROUP_LETTERS = list("ABCDEFGHIJKL")

# Real FIFA 2026 Round of 32 (matches 73-88), encoded in SEQUENTIAL-PAIRING
# ORDER: pairing consecutive winners reproduces the official bracket flow
#   R16:  89=(74,77) 90=(73,75) 93=(83,84) 94=(81,82)
#         91=(76,78) 92=(79,80) 95=(86,88) 96=(85,87)
#   QF:   97=(89,90) 98=(93,94) 99=(91,92) 100=(95,96)
#   SF:   101=(97,98) 102=(99,100)
# Slot syntax: "A1"=Group A winner, "B2"=runner-up, "3rd:ABCDF"=best third
# from one of those groups (allowed-set per official bracket).
# Each entry: (slot_home, slot_away, city, fifa_match_no)
R32_MATCHES = [
    ("E1", "3rd:ABCDF", "Foxborough", 74),
    ("I1", "3rd:CDFGH", "East Rutherford", 77),
    ("A2", "B2", "Inglewood", 73),
    ("F1", "C2", "Guadalupe", 75),
    ("K2", "L2", "Toronto", 83),
    ("H1", "J2", "Inglewood", 84),
    ("D1", "3rd:BEFIJ", "Santa Clara", 81),
    ("G1", "3rd:AEHIJ", "Seattle", 82),
    ("C1", "F2", "Houston", 76),
    ("E2", "I2", "Arlington", 78),
    ("A1", "3rd:CEFHI", "Mexico City", 79),
    ("L1", "3rd:EHIJK", "Atlanta", 80),
    ("J1", "H2", "Miami Gardens", 86),
    ("D2", "G2", "Arlington", 88),
    ("B1", "3rd:EFGIJ", "Vancouver", 85),
    ("K1", "3rd:DEIJL", "Kansas City", 87),
]

# Venues for later rounds, in the same sequential-pairing order
R16_CITIES = ["Houston", "Philadelphia", "Atlanta", "Seattle",
              "East Rutherford", "Mexico City", "Atlanta", "Vancouver"]
QF_CITIES = ["Foxborough", "Inglewood", "Miami Gardens", "Kansas City"]
SF_CITIES = ["Arlington", "Atlanta"]
FINAL_CITY = "East Rutherford"

# Legacy alias kept for callers that only need (home, away) slots
R32_BRACKET = [(m[0], m[1]) for m in R32_MATCHES]


def assign_third_place_slots(groups_with_thirds: list[str]) -> dict[str, str]:
    """
    Assign the 8 qualifying third-place groups to the eight "3rd:XXXXX" slots,
    respecting each slot's allowed-group set (FIFA Annex C resolves the 495
    combinations; we find a valid perfect matching by backtracking, which
    reproduces a legal assignment for every combination).
    Returns {slot_name: group_letter}.
    """
    slots = [m[1] for m in R32_MATCHES if m[1].startswith("3rd:")]
    allowed = {s: set(s.split(":")[1]) for s in slots}
    thirds = sorted(groups_with_thirds)

    # Backtracking perfect matching (8x8 — instant)
    assignment: dict[str, str] = {}

    def backtrack(i: int, used: set[str]) -> bool:
        if i == len(slots):
            return True
        slot = slots[i]
        for g in thirds:
            if g in used or g not in allowed[slot]:
                continue
            assignment[slot] = g
            if backtrack(i + 1, used | {g}):
                return True
            del assignment[slot]
        return False

    if not backtrack(0, set()):
        # No legal perfect matching (shouldn't happen for valid combos):
        # relax constraints and assign greedily so the simulation continues.
        assignment.clear()
        remaining = list(thirds)
        for slot in slots:
            pick = next((g for g in remaining if g in allowed[slot]), remaining[0])
            assignment[slot] = pick
            remaining.remove(pick)
    return assignment


@dataclass
class TeamRecord:
    team: str
    group: str
    pts: int = 0
    gf: int = 0
    ga: int = 0
    gd: int = 0
    fp: int = 0  # fair play (cards)
    # head-to-head record stored externally if needed


@dataclass
class GroupResult:
    group: str
    teams: list[TeamRecord]

    def standings(self) -> list[TeamRecord]:
        """Sort by FIFA tiebreaker rules."""
        return sorted(
            self.teams,
            key=lambda r: (r.pts, r.gd, r.gf, -r.fp),
            reverse=True,
        )

    def qualifier_1(self) -> str:
        return self.standings()[0].team

    def qualifier_2(self) -> str:
        return self.standings()[1].team

    def third_place(self) -> TeamRecord:
        return self.standings()[2]


def _apply_score(records: dict[str, TeamRecord], home: str, away: str, gh: int, ga: int) -> None:
    rh, ra = records[home], records[away]
    rh.gf += gh; rh.ga += ga; rh.gd += gh - ga
    ra.gf += ga; ra.ga += gh; ra.gd += ga - gh
    if gh > ga:
        rh.pts += 3
    elif gh == ga:
        rh.pts += 1; ra.pts += 1
    else:
        ra.pts += 3


def _sample_score(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = np.clip(grid.ravel(), 0, None)
    # inverse-CDF sampling: ~5x faster than rng.choice(p=...) in the hot loop
    cdf = np.cumsum(flat)
    idx = int(np.searchsorted(cdf, rng.random() * cdf[-1]))
    idx = min(idx, len(flat) - 1)
    n = grid.shape[0]
    return idx // n, idx % n


def simulate_group_stage(
    groups: dict[str, list[str]],
    predict_fn,
    rng: np.random.Generator,
    noise: float = 0.0,
    fixtures: list[dict] | None = None,
) -> dict[str, GroupResult]:
    """
    Simulate all group matches.
    predict_fn(home, away, neutral=True, city=None) → (score_grid, lambdas)
      where score_grid is (max_g+1, max_g+1) probability array
    fixtures: optional real fixture list (from wm.sim.fixtures) carrying real
      venue city, host/neutral flags, and — for matches already played —
      actual scores, which are locked in instead of being re-simulated.
    Returns dict group_letter → GroupResult
    """
    results: dict[str, GroupResult] = {}
    records_all: dict[str, dict[str, TeamRecord]] = {
        group: {t: TeamRecord(team=t, group=group) for t in team_list}
        for group, team_list in groups.items()
    }

    if fixtures:
        seen: dict[str, set[frozenset]] = {g: set() for g in groups}
        for fx in fixtures:
            group = fx["group"]
            records = records_all[group]
            home, away = fx["home"], fx["away"]
            if home not in records or away not in records:
                continue
            seen[group].add(frozenset({home, away}))
            if fx["played"]:
                gh, ga = fx["goals_home"], fx["goals_away"]
            else:
                grid, _ = predict_fn(home, away, neutral=fx["neutral"], city=fx.get("city"))
                gh, ga = _sample_score(grid, rng)
            _apply_score(records, home, away, gh, ga)
        # Any pairing missing from the fixture list (defensive): simulate it
        for group, team_list in groups.items():
            for home, away in itertools.combinations(team_list, 2):
                if frozenset({home, away}) not in seen[group]:
                    grid, _ = predict_fn(home, away, neutral=True)
                    gh, ga = _sample_score(grid, rng)
                    _apply_score(records_all[group], home, away, gh, ga)
    else:
        for group, team_list in groups.items():
            for home, away in itertools.combinations(team_list, 2):
                grid, _ = predict_fn(home, away, neutral=True)
                gh, ga = _sample_score(grid, rng)
                _apply_score(records_all[group], home, away, gh, ga)

    for group in groups:
        results[group] = GroupResult(group=group, teams=list(records_all[group].values()))

    return results


def get_third_place_ranking(group_results: dict[str, GroupResult]) -> list[TeamRecord]:
    """Rank all 12 third-place finishers to find 8 best."""
    thirds = [gr.third_place() for gr in group_results.values()]
    return sorted(thirds, key=lambda r: (r.pts, r.gd, r.gf, -r.fp), reverse=True)


def build_r32_bracket(
    group_results: dict[str, GroupResult],
    third_place_ranking: list[TeamRecord],
) -> list[tuple[str, str]]:
    """
    Build the Round of 32 bracket (16 matches).
    Returns list of (home_team, away_team) tuples.
    """
    # Qualified thirds
    best_8_thirds = [t.team for t in third_place_ranking[:8]]
    best_8_groups = [t.group for t in third_place_ranking[:8]]

    # Position lookups
    positions: dict[str, str] = {}
    for group, gr in group_results.items():
        standings = gr.standings()
        positions[f"{group}1"] = standings[0].team
        positions[f"{group}2"] = standings[1].team
        positions[f"{group}3"] = standings[2].team

    # Assign 3rd-place slots (constraint-respecting matching)
    slot_map = assign_third_place_slots(best_8_groups)
    for slot, group in slot_map.items():
        positions[slot] = positions.get(f"{group}3", "TBD")

    bracket = []
    for home_slot, away_slot, city, _match_no in R32_MATCHES:
        home = positions.get(home_slot, home_slot)
        away = positions.get(away_slot, away_slot)
        bracket.append((home, away, city))

    return bracket


def simulate_knockout_match(
    home: str,
    away: str,
    predict_fn,
    rng: np.random.Generator,
    max_goals: int = 10,
    et_factor: float = 0.333,
    base_pen: float = 0.75,
    played: dict | None = None,
    city: str | None = None,
) -> str:
    """
    Simulate a knockout match (must produce a winner).
    played: optional {frozenset({a, b}): winner} of already-decided knockout
    matches — if this pairing was played in reality, the real winner is
    returned instead of simulating.
    city: real venue city for venue-aware features.
    Returns winning team name.
    """
    if played:
        key = frozenset({home, away})
        if key in played:
            return played[key]

    try:
        grid, (lh, la) = predict_fn(home, away, neutral=True, city=city)
    except TypeError:  # predict_fn without city support (legacy/tests)
        grid, (lh, la) = predict_fn(home, away, neutral=True)
    gh, ga = _sample_score(grid, rng)

    if gh != ga:
        return home if gh > ga else away

    # Extra time: reduced lambda
    lh_et, la_et = lh * et_factor, la * et_factor
    gh_et = rng.poisson(lh_et)
    ga_et = rng.poisson(la_et)
    gh += gh_et; ga += ga_et

    if gh != ga:
        return home if gh > ga else away

    # Penalty shootout: slight Elo-weighted advantage
    p_home_pen = base_pen  # simplified 50/50
    return home if rng.random() < 0.5 else away


def simulate_knockout_stage(
    bracket: list[tuple[str, str]],
    predict_fn,
    rng: np.random.Generator,
    **kwargs,
) -> str:
    """
    Simulate knockout stage from R32 through Final.
    Returns champion name.
    """
    current_round = bracket

    while len(current_round) > 1:
        next_round = []
        for home, away in current_round:
            winner = simulate_knockout_match(home, away, predict_fn, rng, **kwargs)
            next_round.append(winner)
        # Pair winners for next round
        current_round = list(zip(next_round[::2], next_round[1::2]))

    # Final match
    if len(current_round) == 1:
        home, away = current_round[0]
        return simulate_knockout_match(home, away, predict_fn, rng, **kwargs)

    return "Unknown"
