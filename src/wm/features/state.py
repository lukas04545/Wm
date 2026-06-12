"""
Current per-team state snapshot for inference-time fixture rows.

Training rows get leak-free rolling features computed match by match; fixture
rows for 2026 (simulation, `wm predict`, the web UI) previously filled all
form/att/def features with NaN — the models could not use any of that signal
at prediction time. This module computes each team's CURRENT Elo,
attack/defence ratings, rolling form, opponent-adjusted performance and EWMA
goals from the full match history, so fixture rows carry the same features
the model saw in training.
"""
from __future__ import annotations

import pandas as pd

from wm import config as cfg_mod
from wm.features import elo_rolling, form


def compute_team_state(
    matches: pd.DataFrame,
    cfg: cfg_mod.Config,
    league_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Return {team: {elo, att, def, <form stats>, <lg_* latest-season>}}.
    league_df: MultiIndex [team, season] FBref aggregates; the latest season's
    real club-form values are attached per team for fixture-row building.
    """
    elo_df, rating_state = elo_rolling.compute(matches, cfg.elo)
    elo_lookup = {
        mid: (r["elo_home_before"], r["elo_away_before"])
        for mid, r in elo_df.set_index("match_id")[
            ["elo_home_before", "elo_away_before"]
        ].iterrows()
    }
    _, form_state = form.compute_form(
        matches, cfg.features.form_windows, elo_lookup=elo_lookup, return_state=True
    )

    league_latest: dict[str, dict] = {}
    if league_df is not None and not league_df.empty:
        for team in league_df.index.get_level_values(0).unique():
            sub = league_df.loc[team]
            league_latest[team] = sub.loc[sub.index.max()].to_dict()

    state: dict[str, dict] = {}
    teams = set(rating_state["elo"]) | set(form_state) | set(league_latest)
    for team in teams:
        s = dict(form_state.get(team, {}))
        s["elo"] = rating_state["elo"].get(team, cfg.elo.initial)
        s["att"] = rating_state["att"].get(team, 0.0)
        s["def"] = rating_state["def"].get(team, 0.0)
        s["league"] = league_latest.get(team)
        state[team] = s
    return state
