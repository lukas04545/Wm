"""
Real 2026 World Cup fixture list, read from the ingested results dataset.

The martj42 dataset ships all 104 scheduled 2026 matches with city, date and
neutral flags; played matches carry scores (updated upstream as the
tournament progresses). This lets the simulator:
  - use the REAL venue per fixture (Azteca altitude, Houston heat, …)
  - honour real host/neutral flags (Mexico at Azteca is a home match)
  - condition on matches already played instead of re-simulating them
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from wm.data.teams import canonical


def load_group_fixtures(
    raw_dir: Path,
    groups: dict[str, list[str]],
    start: str = "2026-06-01",
) -> list[dict]:
    """
    Return the 72 group-stage fixtures as dicts:
      {group, home, away, date, city, neutral, played, goals_home, goals_away}
    Group is inferred from team membership (both teams share a group).
    """
    path = raw_dir / "results.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path, parse_dates=["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= pd.Timestamp(start))]

    team_group = {canonical(t): g for g, teams in groups.items() for t in teams}

    fixtures = []
    for _, row in wc.iterrows():
        home = canonical(str(row["home_team"]))
        away = canonical(str(row["away_team"]))
        g_h, g_a = team_group.get(home), team_group.get(away)
        if g_h is None or g_a is None or g_h != g_a:
            continue  # knockout placeholder or unknown team
        played = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
        fixtures.append({
            "group": g_h,
            "home": home,
            "away": away,
            "date": row["date"],
            "city": str(row["city"]),
            "neutral": bool(row["neutral"]),
            "played": played,
            "goals_home": int(row["home_score"]) if played else None,
            "goals_away": int(row["away_score"]) if played else None,
        })
    return fixtures


def load_played_knockouts(raw_dir: Path, start: str = "2026-06-27") -> dict[frozenset, str]:
    """
    Return {frozenset({team_a, team_b}): winner} for knockout matches already
    decided (knockout games always have a winner; the dataset records the
    90'/ET score — for shootout wins the shootouts.csv winner takes priority).
    """
    path = raw_dir / "results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, parse_dates=["date"])
    ko = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= pd.Timestamp(start))]
    ko = ko.dropna(subset=["home_score", "away_score"])

    shootout_winner: dict[frozenset, str] = {}
    sp = raw_dir / "shootouts.csv"
    if sp.exists():
        so = pd.read_csv(sp, parse_dates=["date"])
        so = so[so["date"] >= pd.Timestamp(start)]
        for _, r in so.iterrows():
            key = frozenset({canonical(str(r["home_team"])), canonical(str(r["away_team"]))})
            shootout_winner[key] = canonical(str(r["winner"]))

    result: dict[frozenset, str] = {}
    for _, row in ko.iterrows():
        home = canonical(str(row["home_team"]))
        away = canonical(str(row["away_team"]))
        key = frozenset({home, away})
        gh, ga = int(row["home_score"]), int(row["away_score"])
        if gh > ga:
            result[key] = home
        elif ga > gh:
            result[key] = away
        elif key in shootout_winner:
            result[key] = shootout_winner[key]
    return result
