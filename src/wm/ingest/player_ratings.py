"""
Load aggregated squad strength from EA FC / FIFA player rating datasets.

Expected input: a CSV at data/raw/players/ea_fc_ratings.csv with columns:
  nationality, overall, potential, age, position, release_clause_eur, value_eur
  (standard sofifa/Kaggle FIFA/EA FC format)

If the file is absent, returns NaN features (model degrades gracefully).

Data can be obtained from:
  https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset
  (or equivalent for any EA FC edition, FIFA 22/23/24/25)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POSITIONS = {
    "GK": ["GK"],
    "DEF": ["CB", "LB", "RB", "LWB", "RWB", "SW"],
    "MID": ["CM", "CDM", "CAM", "LM", "RM", "DM"],
    "ATT": ["ST", "CF", "LW", "RW", "SS", "RF", "LF"],
}


def _position_group(pos: str | None) -> str:
    if not pos:
        return "MID"
    p = str(pos).upper().strip()
    for group, positions in POSITIONS.items():
        if p in positions:
            return group
    return "MID"


def load(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "players" / "ea_fc_ratings.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalize nationality column name
    for col in ("nationality_name", "nationality", "nation"):
        if col in df.columns:
            df = df.rename(columns={col: "nationality"})
            break

    if "nationality" not in df.columns:
        return None

    df = df[["nationality", "overall", "potential", "age"]].dropna(subset=["nationality", "overall"])
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["overall"])

    return df


def aggregate_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player ratings to team-level squad strength features."""
    rows = []
    for nat, grp in df.groupby("nationality"):
        grp = grp.sort_values("overall", ascending=False)
        top25 = grp.head(25)
        top11 = grp.head(11)

        row = {
            "team": nat,
            "squad_mean_top25": top25["overall"].mean(),
            "squad_mean_top11": top11["overall"].mean(),
            "squad_max": grp["overall"].max(),
            "squad_mean_age": grp.head(25)["age"].mean(),
            "squad_depth": len(grp),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("team")


def get_squad_features(raw_dir: Path) -> pd.DataFrame | None:
    df = load(raw_dir)
    if df is None:
        return None
    return aggregate_by_team(df)
