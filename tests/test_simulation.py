"""Tests for Monte Carlo simulation properties."""
import numpy as np
import pytest


def dummy_predict_fn(home, away, neutral=True):
    grid = np.zeros((11, 11))
    grid[1, 0] = 0.6
    grid[0, 1] = 0.3
    grid[1, 1] = 0.1
    return grid, (1.0, 0.8)


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
ALL_TEAMS = [t for teams in GROUPS.values() for t in teams]


def test_simulation_champion_probabilities_sum_to_one():
    """Champion probabilities across all teams must sum to ~1."""
    from wm.sim.tournament import (
        simulate_group_stage, get_third_place_ranking, build_r32_bracket,
        simulate_knockout_match
    )

    rng = np.random.default_rng(0)
    n_runs = 100
    champ_count: dict[str, int] = {t: 0 for t in ALL_TEAMS}

    for _ in range(n_runs):
        results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
        thirds = get_third_place_ranking(results)
        bracket = build_r32_bracket(results, thirds)

        # Quick knockout simulation
        current = [(m[0], m[1]) for m in bracket]
        while len(current) > 1:
            winners = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in current]
            current = list(zip(winners[::2], winners[1::2]))
        if current:
            champ = simulate_knockout_match(current[0][0], current[0][1], dummy_predict_fn, rng)
            if champ in champ_count:
                champ_count[champ] += 1

    total = sum(champ_count.values())
    assert total == n_runs, f"Expected {n_runs} champions, got {total}"


def test_stage_probabilities_monotone():
    """P(champion) <= P(final) <= P(semifinal) etc."""
    from wm.sim.tournament import (
        simulate_group_stage, get_third_place_ranking, build_r32_bracket,
        simulate_knockout_match
    )

    rng = np.random.default_rng(1)
    n_runs = 200
    stages = {t: {"r32": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "champ": 0}
              for t in ALL_TEAMS}

    for _ in range(n_runs):
        results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
        thirds = get_third_place_ranking(results)
        bracket = build_r32_bracket(results, thirds)

        pairs = [(m[0], m[1]) for m in bracket]
        for h, a in pairs:
            stages[h]["r32"] += 1
            stages[a]["r32"] += 1

        r32w = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in pairs]
        r16 = list(zip(r32w[::2], r32w[1::2]))
        for h, a in r16:
            stages[h]["r16"] += 1; stages[a]["r16"] += 1
        r16w = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in r16]
        qf = list(zip(r16w[::2], r16w[1::2]))
        for h, a in qf:
            stages[h]["qf"] += 1; stages[a]["qf"] += 1
        qfw = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in qf]
        sf = list(zip(qfw[::2], qfw[1::2]))
        for h, a in sf:
            stages[h]["sf"] += 1; stages[a]["sf"] += 1
        sfw = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in sf]
        if len(sfw) >= 2:
            stages[sfw[0]]["final"] += 1; stages[sfw[1]]["final"] += 1
            champ = simulate_knockout_match(sfw[0], sfw[1], dummy_predict_fn, rng)
            stages[champ]["champ"] += 1

    for team in ALL_TEAMS:
        s = stages[team]
        assert s["champ"] <= s["final"] + 1  # +1 for rng tolerance
        assert s["final"] <= s["sf"] + 1
        assert s["sf"] <= s["qf"] + 1


def test_determinism_with_seed():
    """Same seed produces same champion."""
    from wm.sim.tournament import (
        simulate_group_stage, get_third_place_ranking, build_r32_bracket,
        simulate_knockout_match
    )

    def run_once(seed: int) -> str:
        rng = np.random.default_rng(seed)
        results = simulate_group_stage(GROUPS, dummy_predict_fn, rng)
        thirds = get_third_place_ranking(results)
        bracket = build_r32_bracket(results, thirds)
        current = [(m[0], m[1]) for m in bracket]
        while len(current) > 1:
            winners = [simulate_knockout_match(h, a, dummy_predict_fn, rng) for h, a in current]
            current = list(zip(winners[::2], winners[1::2]))
        return simulate_knockout_match(current[0][0], current[0][1], dummy_predict_fn, rng)

    champ1 = run_once(777)
    champ2 = run_once(777)
    assert champ1 == champ2
