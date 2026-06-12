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


EWMA_HALFLIFE = 10  # matches
_EWMA_ALPHA = 1.0 - 0.5 ** (1.0 / EWMA_HALFLIFE)


def _elo_expected(elo_a: float, elo_b: float, neutral: bool, home_adv: float = 100.0) -> float:
    dr = elo_a - elo_b + (0.0 if neutral else home_adv)
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def compute_form(
    matches: pd.DataFrame,
    windows: list[int],
    elo_lookup: dict[int, tuple[float, float]] | None = None,
    return_state: bool = False,
):
    """
    For each match, compute rolling form features for home and away teams.
    Uses strictly pre-match history (no leakage).

    elo_lookup: optional {match_id: (elo_home_before, elo_away_before)} —
    enables opponent-adjusted features:
      {side}_perf_vs_elo_{W}: mean (actual score − Elo-expected score), i.e.
        how much the team over/under-performs its rating
      {side}_opp_elo_{W}: mean opponent Elo faced (schedule strength)
    plus EWMA goals for/against with a 10-match half-life.

    If return_state is True, returns (DataFrame, {team: {stat: value}}) with
    each team's CURRENT form — used to build leak-free fixture rows at
    simulation/predict time.
    """
    matches = matches.sort_values("date").copy()
    if "match_id" not in matches.columns:
        matches["match_id"] = range(len(matches))

    # Build per-team match history as we iterate
    team_history: dict[str, list[dict]] = {}
    team_ewma: dict[str, tuple[float, float]] = {}  # team -> (ewma_gf, ewma_ga)

    def window_stats(hist: list[dict], date, W: int) -> dict[str, float]:
        recent = [h for h in hist if h["date"] < date][-W:]
        out: dict[str, float] = {}
        if len(recent) == 0:
            for stat in ("ppg", "gf_pg", "ga_pg", "gd_pg", "winrate"):
                out[stat] = float("nan")
            out["n_matches"] = 0.0
            if elo_lookup is not None:
                out["perf_vs_elo"] = float("nan")
                out["opp_elo"] = float("nan")
        else:
            pts = [h["points"] for h in recent]
            gf = [h["gf"] for h in recent]
            ga = [h["ga"] for h in recent]
            out["ppg"] = sum(pts) / len(pts)
            out["gf_pg"] = sum(gf) / len(gf)
            out["ga_pg"] = sum(ga) / len(ga)
            out["gd_pg"] = (sum(gf) - sum(ga)) / len(gf)
            out["winrate"] = sum(1 for h in recent if h["points"] == 3) / len(recent)
            out["n_matches"] = float(len(recent))
            if elo_lookup is not None:
                perfs = [h["perf"] for h in recent if h["perf"] is not None]
                opps = [h["opp_elo"] for h in recent if h["opp_elo"] is not None]
                out["perf_vs_elo"] = sum(perfs) / len(perfs) if perfs else float("nan")
                out["opp_elo"] = sum(opps) / len(opps) if opps else float("nan")
        return out

    rows = []
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        mid = row["match_id"]
        neutral = bool(row.get("is_neutral", False))

        feat: dict[str, float] = {}
        for side, team in [("home", home), ("away", away)]:
            hist = team_history.get(team, [])
            for W in windows:
                stats = window_stats(hist, date, W)
                for stat, v in stats.items():
                    feat[f"{side}_{stat}_{W}"] = v
            ew = team_ewma.get(team)
            feat[f"{side}_ewma_gf"] = ew[0] if ew else float("nan")
            feat[f"{side}_ewma_ga"] = ew[1] if ew else float("nan")

        rows.append(feat)

        # Pre-match Elo for opponent-adjusted history records
        elo_h, elo_a = (elo_lookup.get(mid, (None, None)) if elo_lookup is not None
                        else (None, None))

        # Update history for both teams
        outcome = int(row.get("outcome", 0))
        for side, team, own_goals, opp_goals in [
            ("home", home, int(row.get("goals_home", 0)), int(row.get("goals_away", 0))),
            ("away", away, int(row.get("goals_away", 0)), int(row.get("goals_home", 0))),
        ]:
            pts = _points(outcome if side == "home" else -outcome)
            actual = {3: 1.0, 1: 0.5, 0: 0.0}[pts]
            perf = opp_elo = None
            if elo_h is not None and elo_a is not None:
                if side == "home":
                    perf = actual - _elo_expected(elo_h, elo_a, neutral)
                    opp_elo = elo_a
                else:
                    perf = actual - (1.0 - _elo_expected(elo_h, elo_a, neutral))
                    opp_elo = elo_h
            if team not in team_history:
                team_history[team] = []
            team_history[team].append({
                "date": date,
                "match_id": mid,
                "points": pts,
                "gf": own_goals,
                "ga": opp_goals,
                "perf": perf,
                "opp_elo": opp_elo,
            })
            ew = team_ewma.get(team)
            if ew is None:
                team_ewma[team] = (float(own_goals), float(opp_goals))
            else:
                team_ewma[team] = (
                    _EWMA_ALPHA * own_goals + (1 - _EWMA_ALPHA) * ew[0],
                    _EWMA_ALPHA * opp_goals + (1 - _EWMA_ALPHA) * ew[1],
                )

    result = pd.DataFrame(rows, index=matches.index)
    if not return_state:
        return result

    # Snapshot of each team's current form (as of after the last match)
    far_future = matches["date"].max() + pd.Timedelta(days=1)
    state: dict[str, dict[str, float]] = {}
    for team, hist in team_history.items():
        s: dict[str, float] = {}
        for W in windows:
            for stat, v in window_stats(hist, far_future, W).items():
                s[f"{stat}_{W}"] = v
        ew = team_ewma.get(team)
        s["ewma_gf"] = ew[0] if ew else float("nan")
        s["ewma_ga"] = ew[1] if ew else float("nan")
        s["last_match_date"] = hist[-1]["date"]
        state[team] = s
    return result, state


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
