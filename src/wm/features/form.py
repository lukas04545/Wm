"""Rolling form features for each team (points, goals, GD over last N matches)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _points(outcome: int) -> int:
    if outcome == 1:
        return 3
    if outcome == 0:
        return 1
    return 0


def compute_form(matches: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    For each match, compute rolling form features for home and away teams.
    Uses strictly pre-match history (no leakage).

    Returns a DataFrame indexed the same as matches with columns:
      form_{team}_{W}_{stat} for team in {home,away}, W in windows,
        stat in {ppg, gf_pg, ga_pg, gd_pg, winrate}
    """
    matches = matches.sort_values("date").copy()
    matches["match_id"] = range(len(matches))

    # Build per-team match history as we iterate
    team_history: dict[str, list[dict]] = {}

    rows = []
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        mid = row["match_id"]

        feat: dict[str, float] = {}
        for side, team in [("home", home), ("away", away)]:
            hist = team_history.get(team, [])
            for W in windows:
                recent = [h for h in hist if h["date"] < date][-W:]
                if len(recent) == 0:
                    feat[f"{side}_ppg_{W}"] = float("nan")
                    feat[f"{side}_gf_pg_{W}"] = float("nan")
                    feat[f"{side}_ga_pg_{W}"] = float("nan")
                    feat[f"{side}_gd_pg_{W}"] = float("nan")
                    feat[f"{side}_winrate_{W}"] = float("nan")
                    feat[f"{side}_n_matches_{W}"] = 0.0
                else:
                    pts = [h["points"] for h in recent]
                    gf = [h["gf"] for h in recent]
                    ga = [h["ga"] for h in recent]
                    feat[f"{side}_ppg_{W}"] = sum(pts) / len(pts)
                    feat[f"{side}_gf_pg_{W}"] = sum(gf) / len(gf)
                    feat[f"{side}_ga_pg_{W}"] = sum(ga) / len(ga)
                    feat[f"{side}_gd_pg_{W}"] = (sum(gf) - sum(ga)) / len(gf)
                    feat[f"{side}_winrate_{W}"] = sum(1 for h in recent if h["points"] == 3) / len(recent)
                    feat[f"{side}_n_matches_{W}"] = float(len(recent))

        rows.append(feat)

        # Update history for both teams
        outcome = int(row.get("outcome", 0))
        for side, team, own_goals, opp_goals in [
            ("home", home, int(row.get("goals_home", 0)), int(row.get("goals_away", 0))),
            ("away", away, int(row.get("goals_away", 0)), int(row.get("goals_home", 0))),
        ]:
            pts = _points(outcome if side == "home" else -outcome)
            if team not in team_history:
                team_history[team] = []
            team_history[team].append({
                "date": date,
                "match_id": mid,
                "points": pts,
                "gf": own_goals,
                "ga": opp_goals,
            })

    return pd.DataFrame(rows, index=matches.index)


def compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute days since each team's last match before this one."""
    matches = matches.sort_values("date").copy()
    last_match: dict[str, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in matches.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]
        home_rest.append((date - last_match[home]).days if home in last_match else float("nan"))
        away_rest.append((date - last_match[away]).days if away in last_match else float("nan"))
        last_match[home] = date
        last_match[away] = date

    return pd.DataFrame(
        {"days_rest_home": home_rest, "days_rest_away": away_rest},
        index=matches.index,
    )


def compute_h2h(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute head-to-head record over last `window` meetings between teams."""
    matches = matches.sort_values("date").copy()
    h2h_history: dict[frozenset, list[dict]] = {}
    rows = []

    for _, row in matches.iterrows():
        home, away, date = row["home_team"], row["away_team"], row["date"]
        key = frozenset([home, away])
        hist = [h for h in h2h_history.get(key, []) if h["date"] < date][-window:]

        if not hist:
            rows.append({"h2h_ppg_home": float("nan"), "h2h_ppg_away": float("nan")})
        else:
            home_pts = [h["points_home"] for h in hist]
            away_pts = [h["points_away"] for h in hist]
            rows.append({
                "h2h_ppg_home": sum(home_pts) / len(home_pts),
                "h2h_ppg_away": sum(away_pts) / len(away_pts),
            })

        outcome = int(row.get("outcome", 0))
        if key not in h2h_history:
            h2h_history[key] = []
        h2h_history[key].append({
            "date": date,
            "points_home": _points(outcome),
            "points_away": _points(-outcome),
        })

    return pd.DataFrame(rows, index=matches.index)
