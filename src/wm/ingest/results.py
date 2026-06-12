"""Ingest international football results from martj42/international_results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from wm.ingest.cache import download

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
)

# Map tournament strings to tier numbers (higher = more important)
TOURNAMENT_TIERS = {
    "FIFA World Cup": 5,
    "FIFA World Cup qualification": 4,
    "UEFA Euro": 4,
    "Copa América": 4,
    "Africa Cup of Nations": 4,
    "Asian Cup": 4,
    "CONCACAF Gold Cup": 3,
    "OFC Nations Cup": 3,
    "UEFA Nations League": 3,
    "CONCACAF Nations League": 3,
    "UEFA Euro qualification": 3,
    "Copa América qualification": 3,
    "Africa Cup of Nations qualification": 3,
    "Asian Cup qualification": 3,
    "Friendly": 1,
}


def fetch(raw_dir: Path, registry: Path, force: bool = False) -> None:
    download(RESULTS_URL, raw_dir / "results.csv", registry, force=force)
    download(SHOOTOUTS_URL, raw_dir / "shootouts.csv", registry, force=force)


def load(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(raw_dir / "results.csv", parse_dates=["date"])
    shootouts = pd.read_csv(raw_dir / "shootouts.csv", parse_dates=["date"])
    return results, shootouts


def get_tournament_tier(tournament: str) -> int:
    for key, tier in TOURNAMENT_TIERS.items():
        if key.lower() in tournament.lower():
            return tier
    if "qualification" in tournament.lower() or "qualifier" in tournament.lower():
        return 3
    if "friendly" in tournament.lower():
        return 1
    return 2  # default: minor tournament


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw results dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index
    df["tournament_tier"] = df["tournament"].apply(get_tournament_tier)
    df["is_neutral"] = df["neutral"].astype(bool)
    df["goals_home"] = df["home_score"].astype(int)
    df["goals_away"] = df["away_score"].astype(int)
    df["goal_diff"] = df["goals_home"] - df["goals_away"]
    df["outcome"] = (
        (df["goals_home"] > df["goals_away"]).astype(int)
        - (df["goals_home"] < df["goals_away"]).astype(int)
    )  # +1 home win, 0 draw, -1 away win
    df["result"] = df["outcome"].map({1: "W", 0: "D", -1: "L"})
    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False) & ~df[
        "tournament"
    ].str.contains("qualification", case=False, na=False)
    return df
