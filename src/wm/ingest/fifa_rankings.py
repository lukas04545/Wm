"""
Load historical FIFA world rankings.

Expected: data/raw/rankings/fifa_rankings.csv with columns:
  rank_date, country_full, total_points, rank

Kaggle dataset: "FIFA International Soccer Men's Ranking 1993-Now"
  https://www.kaggle.com/datasets/cashncarry/fifaworldranking

If absent, this feature set is omitted (model degrades gracefully).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "rankings" / "fifa_rankings.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["rank_date"])
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalize column names across different Kaggle versions
    renames = {
        "country_full": "team",
        "countryname": "team",
        "country": "team",
        "total_points": "fifa_points",
        "points": "fifa_points",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    if "team" not in df.columns or "rank" not in df.columns:
        return None

    df = df[["rank_date", "team", "rank", "fifa_points"]].dropna()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["fifa_points"] = pd.to_numeric(df["fifa_points"], errors="coerce")
    return df.sort_values("rank_date")


def get_rank_as_of(rankings_df: pd.DataFrame, team: str, date: pd.Timestamp) -> tuple[float, float]:
    """Return (rank, points) for team on or before date. Returns (NaN, NaN) if unknown."""
    sub = rankings_df[(rankings_df["team"] == team) & (rankings_df["rank_date"] <= date)]
    if sub.empty:
        return float("nan"), float("nan")
    row = sub.iloc[-1]
    return float(row["rank"]), float(row.get("fifa_points", float("nan")))
