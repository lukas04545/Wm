"""
Squad-strength features from player rating datasets.

Two supported sources (checked in order):

1. data/raw/players/ea_fc_ratings.csv — manually downloaded Kaggle EA FC
   dataset (sofifa schema: nationality, overall, potential, age). Preferred
   when present because `overall` is EA's official rating.

2. data/raw/players/fifaindex_players.csv — auto-downloaded by `wm ingest`
   from github.com/reh1548/FIFA-24-Player-Dataset (FIFA 24 era, scraped from
   fifaindex.com). Has no `overall` column, so a rating proxy is computed as
   the mean of each player's 6 best attributes (4 best GK attributes for
   keepers). Attributes share the 1-99 scale of EA overalls, so the proxy is
   comparable across teams.

The snapshot reflects 2023/24 player quality. To avoid leaking future squad
quality into decades-old training rows, the returned frame carries
`attrs["valid_from"]` — feature building only applies squad features to
matches after that date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wm.ingest.cache import download

FIFAINDEX_URL = (
    "https://raw.githubusercontent.com/reh1548/FIFA-24-Player-Dataset/main/player_stats.csv"
)

# Date from which the FC24-era snapshot is a valid description of squads
SNAPSHOT_VALID_FROM = "2023-07-01"

OUTFIELD_ATTRS = [
    "ball_control", "dribbling", "marking", "slide_tackle", "stand_tackle",
    "aggression", "reactions", "att_position", "interceptions", "vision",
    "composure", "crossing", "short_pass", "long_pass", "acceleration",
    "stamina", "strength", "sprint_speed", "agility", "jumping", "heading",
    "shot_power", "finishing", "long_shots", "curve", "fk_acc", "penalties",
    "volleys",
]
GK_ATTRS = ["gk_positioning", "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes"]


def fetch(raw_dir: Path, registry: Path, force: bool = False) -> Path:
    """Download the auto-ingestable player dataset."""
    dest = raw_dir / "players" / "fifaindex_players.csv"
    return download(FIFAINDEX_URL, dest, registry, force=force)


def _load_kaggle_ea_fc(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    for col in ("nationality_name", "nationality", "nation", "country"):
        if col in df.columns:
            df = df.rename(columns={col: "nationality"})
            break
    if "nationality" not in df.columns or "overall" not in df.columns:
        return None
    df = df[["nationality", "overall", "age"] + (["potential"] if "potential" in df.columns else [])]
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df.dropna(subset=["nationality", "overall"])


def _load_fifaindex(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    if "country" not in df.columns:
        return None

    attr_cols = [c for c in OUTFIELD_ATTRS if c in df.columns]
    gk_cols = [c for c in GK_ATTRS if c in df.columns]
    if not attr_cols:
        return None

    for c in attr_cols + gk_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Outfield proxy: mean of the player's 6 strongest attributes
    outfield = np.sort(df[attr_cols].to_numpy(dtype=float), axis=1)[:, -6:].mean(axis=1)
    # GK proxy: mean of the 4 strongest GK attributes
    gk = (
        np.sort(df[gk_cols].to_numpy(dtype=float), axis=1)[:, -4:].mean(axis=1)
        if gk_cols else np.zeros(len(df))
    )
    rating = np.nanmax(np.column_stack([outfield, gk]), axis=1)

    out = pd.DataFrame({
        "nationality": df["country"],
        "overall": rating,
        "age": pd.to_numeric(df.get("age"), errors="coerce"),
    })
    return out.dropna(subset=["nationality", "overall"])


def load(raw_dir: Path) -> pd.DataFrame | None:
    kaggle_path = raw_dir / "players" / "ea_fc_ratings.csv"
    if kaggle_path.exists():
        df = _load_kaggle_ea_fc(kaggle_path)
        if df is not None:
            return df

    fifaindex_path = raw_dir / "players" / "fifaindex_players.csv"
    if fifaindex_path.exists():
        return _load_fifaindex(fifaindex_path)

    return None


def aggregate_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player ratings to team-level squad strength features."""
    from wm.data.teams import canonical

    df = df.copy()
    df["nationality"] = df["nationality"].astype(str).map(canonical)

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
            "squad_mean_age": top25["age"].mean(),
            "squad_depth": len(grp),
        }
        rows.append(row)

    result = pd.DataFrame(rows).set_index("team")
    result.attrs["valid_from"] = SNAPSHOT_VALID_FROM
    return result


def get_squad_features(raw_dir: Path) -> pd.DataFrame | None:
    df = load(raw_dir)
    if df is None:
        return None
    return aggregate_by_team(df)
