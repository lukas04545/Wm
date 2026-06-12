"""Assemble the final leak-free feature matrix for training and inference."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from wm import config as cfg_mod
from wm.data import build as build_mod
from wm.data.teams import canonical, load_confederations, CONFEDERATION_STRENGTH
from wm.features import elo_rolling, form, context
from wm.ingest import player_ratings as pr_mod, fifa_rankings as rank_mod

# Target columns for model training
TARGET_WDL = "label_wdl"     # 2=home win, 1=draw, 0=away win
TARGET_GH = "goals_home"
TARGET_GA = "goals_away"

# conf_home/conf_away stay as reference columns but are excluded from model
# features (LightGBM rejects string dtype); conf_strength_* carries the signal.
CATEGORICAL_FEATURES: list[str] = []


def build_matrix(
    matches: pd.DataFrame,
    cfg: cfg_mod.Config,
    squad_df: pd.DataFrame | None = None,
    rankings_df: pd.DataFrame | None = None,
    weather_cache: dict | None = None,
) -> pd.DataFrame:
    """
    Build complete feature matrix. All features use only information
    available strictly before each match date.
    """
    wc_path = cfg_mod.ROOT / "config" / "wc2026.yaml"
    confs = load_confederations(wc_path) if wc_path.exists() else {}
    elo_cfg = cfg.elo

    # 1. Elo features
    elo_df, _ = elo_rolling.compute(matches, elo_cfg)
    elo_df = elo_df.set_index("match_id")

    # 2. Form features
    form_df = form.compute_form(matches, cfg.features.form_windows)
    rest_df = form.compute_rest_days(matches)
    h2h_df = form.compute_h2h(matches, cfg.features.h2h_window)

    # 3. Context features
    travel_df = context.compute_travel(matches)
    alt_df = context.compute_altitude_features(matches)
    weather_df = (
        context.attach_weather(matches, weather_cache)
        if weather_cache
        else pd.DataFrame(index=matches.index)
    )

    # 4. Confederation features
    conf_home = matches["home_team"].apply(lambda t: confs.get(canonical(t), "OTHER"))
    conf_away = matches["away_team"].apply(lambda t: confs.get(canonical(t), "OTHER"))
    conf_str_home = conf_home.map(CONFEDERATION_STRENGTH).fillna(0.35)
    conf_str_away = conf_away.map(CONFEDERATION_STRENGTH).fillna(0.35)

    # 5. FIFA rankings (optional)
    rank_feats = _rank_features(matches, rankings_df)

    # 6. Squad strength (optional, joined by year)
    squad_feats = _squad_features(matches, squad_df)

    # Assemble
    feat = matches[["match_id", "date", "home_team", "away_team",
                     "goals_home", "goals_away", "tournament_tier", "is_neutral"]].copy()

    feat = feat.join(elo_df[["elo_home_before", "elo_away_before",
                               "elo_diff_before", "elo_expected_home"]], on="match_id")
    feat = feat.join(form_df)
    feat = feat.join(rest_df)
    feat = feat.join(h2h_df)
    feat = feat.join(travel_df)
    feat = feat.join(alt_df)
    feat = feat.join(weather_df)
    feat["conf_home"] = conf_home.values
    feat["conf_away"] = conf_away.values
    feat["conf_strength_home"] = conf_str_home.values
    feat["conf_strength_away"] = conf_str_away.values
    feat["conf_strength_diff"] = feat["conf_strength_home"] - feat["conf_strength_away"]

    if rank_feats is not None:
        feat = feat.join(rank_feats)
    if squad_feats is not None:
        feat = feat.join(squad_feats)

    # Target
    feat[TARGET_WDL] = matches["outcome"].map({1: 2, 0: 1, -1: 0}).values

    # Sample weight: exponential decay by age, scaled by tournament tier
    ref_date = matches["date"].max()
    halflife_days = cfg.features.weight_halflife_years * 365.25
    age_days = (ref_date - matches["date"]).dt.days
    feat["sample_weight"] = (
        np.exp(-age_days * math.log(2) / halflife_days)
        * matches["tournament_tier"].clip(1, 5)
        / 5.0
    )

    return feat


def _rank_features(matches: pd.DataFrame, rankings_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if rankings_df is None:
        return None
    rows = []
    for _, row in matches.iterrows():
        rh, ph = rank_mod.get_rank_as_of(rankings_df, row["home_team"], row["date"])
        ra, pa = rank_mod.get_rank_as_of(rankings_df, row["away_team"], row["date"])
        rows.append({
            "fifa_rank_home": rh,
            "fifa_rank_away": ra,
            "fifa_rank_diff": (ra - rh) if not (pd.isna(rh) or pd.isna(ra)) else float("nan"),
            "fifa_points_diff": (ph - pa) if not (pd.isna(ph) or pd.isna(pa)) else float("nan"),
        })
    return pd.DataFrame(rows, index=matches.index)


def _squad_features(matches: pd.DataFrame, squad_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if squad_df is None:
        return None
    rows = []
    for _, row in matches.iterrows():
        sh = squad_df.loc[row["home_team"]] if row["home_team"] in squad_df.index else None
        sa = squad_df.loc[row["away_team"]] if row["away_team"] in squad_df.index else None
        feat: dict[str, float] = {}
        for side, sq in [("home", sh), ("away", sa)]:
            if sq is not None:
                feat[f"squad_mean_top25_{side}"] = float(sq.get("squad_mean_top25", float("nan")))
                feat[f"squad_mean_top11_{side}"] = float(sq.get("squad_mean_top11", float("nan")))
                feat[f"squad_max_{side}"] = float(sq.get("squad_max", float("nan")))
                feat[f"squad_age_{side}"] = float(sq.get("squad_mean_age", float("nan")))
            else:
                for k in ["squad_mean_top25", "squad_mean_top11", "squad_max", "squad_age"]:
                    feat[f"{k}_{side}"] = float("nan")
        if "squad_mean_top25_home" in feat and "squad_mean_top25_away" in feat:
            feat["squad_diff_top25"] = feat["squad_mean_top25_home"] - feat["squad_mean_top25_away"]
        rows.append(feat)
    return pd.DataFrame(rows, index=matches.index)


