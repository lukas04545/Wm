"""
Historical FIFA world rankings.

Auto-downloaded by `wm ingest` from the Dato-Futbol mirror
(github.com/Dato-Futbol/fifa-ranking — monthly points per team, Dec 1992 to
~Sep 2024). Rank is derived per ranking date by sorting points.

A manually downloaded Kaggle file (rank_date/country_full/total_points/rank
schema) is also supported and takes precedence when present.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from wm.ingest.cache import download

DATO_FUTBOL_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv"
)


def fetch(raw_dir: Path, registry: Path, force: bool = False) -> Path:
    dest = raw_dir / "rankings" / "fifa_rankings_dato.csv"
    return download(DATO_FUTBOL_URL, dest, registry, force=force)


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize either supported schema to: rank_date, team, rank, fifa_points."""
    df.columns = [c.lower().strip() for c in df.columns]

    renames = {
        "country_full": "team",
        "countryname": "team",
        "country": "team",
        "total_points": "fifa_points",
        "points": "fifa_points",
        "date": "rank_date",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    if "team" not in df.columns or "rank_date" not in df.columns:
        return None

    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df["fifa_points"] = pd.to_numeric(df.get("fifa_points"), errors="coerce")
    df = df.dropna(subset=["rank_date", "team"])

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    else:
        # Dato-Futbol schema has points only — derive rank per ranking date
        df = df.dropna(subset=["fifa_points"])
        df["rank"] = (
            df.groupby("rank_date")["fifa_points"].rank(ascending=False, method="min")
        )

    from wm.data.teams import canonical
    df["team"] = df["team"].astype(str).map(canonical)

    return df[["rank_date", "team", "rank", "fifa_points"]].sort_values("rank_date")


def load(raw_dir: Path) -> pd.DataFrame | None:
    for fname in ("fifa_rankings.csv", "fifa_rankings_dato.csv"):
        path = raw_dir / "rankings" / fname
        if path.exists():
            df = _normalize(pd.read_csv(path, low_memory=False))
            if df is not None and not df.empty:
                return df
    return None


class RankIndex:
    """Fast as-of lookup of (rank, points) per team — O(log n) per query."""

    def __init__(self, rankings_df: pd.DataFrame, max_staleness_days: int = 1000):
        self.max_staleness = max_staleness_days
        self._idx: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for team, g in rankings_df.groupby("team"):
            g = g.sort_values("rank_date")
            self._idx[team] = (
                g["rank_date"].to_numpy(dtype="datetime64[ns]"),
                g["rank"].to_numpy(dtype=float),
                g["fifa_points"].to_numpy(dtype=float),
            )

    def lookup(self, team: str, date: pd.Timestamp) -> tuple[float, float]:
        """Return (rank, points) as of date, or (NaN, NaN) if unknown/too stale."""
        entry = self._idx.get(team)
        if entry is None:
            return float("nan"), float("nan")
        dates, ranks, points = entry
        i = np.searchsorted(dates, np.datetime64(date), side="right") - 1
        if i < 0:
            return float("nan"), float("nan")
        age_days = (np.datetime64(date) - dates[i]) / np.timedelta64(1, "D")
        if age_days > self.max_staleness:
            return float("nan"), float("nan")
        return float(ranks[i]), float(points[i])


def get_rank_as_of(rankings_df: pd.DataFrame, team: str, date: pd.Timestamp) -> tuple[float, float]:
    """Single-query convenience lookup (builds no index — fine for one-offs)."""
    sub = rankings_df[(rankings_df["team"] == team) & (rankings_df["rank_date"] <= date)]
    if sub.empty:
        return float("nan"), float("nan")
    row = sub.iloc[-1]
    return float(row["rank"]), float(row.get("fifa_points", float("nan")))
