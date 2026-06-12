"""Assemble the final leak-free feature matrix for training and inference."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from wm import config as cfg_mod
from wm.data import build as build_mod
from wm.data.teams import canonical, load_confederations, CONFEDERATION_STRENGTH
from wm.features import elo_rolling, form, context, team_meta
from wm.ingest import player_ratings as pr_mod, fifa_rankings as rank_mod

# 2026 World Cup host nations get a home-advantage signal at neutral venues
HOST_NATIONS = {"United States", "Mexico", "Canada"}

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

    # 1. Elo + attack/defence rating features
    elo_df, _ = elo_rolling.compute(matches, elo_cfg)
    elo_df = elo_df.set_index("match_id")

    # 2. Form features (opponent-adjusted via pre-match Elo)
    elo_lookup = {
        mid: (r["elo_home_before"], r["elo_away_before"])
        for mid, r in elo_df[["elo_home_before", "elo_away_before"]].iterrows()
    }
    form_df = form.compute_form(matches, cfg.features.form_windows, elo_lookup=elo_lookup)
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

    # 7. Socioeconomic & World Cup pedigree background covariates
    meta_feats = _meta_features(matches)

    # Assemble
    feat = matches[["match_id", "date", "home_team", "away_team",
                     "goals_home", "goals_away", "tournament_tier", "is_neutral"]].copy()

    feat = feat.join(elo_df[["elo_home_before", "elo_away_before",
                               "elo_diff_before", "elo_expected_home",
                               "att_home_before", "def_home_before",
                               "att_away_before", "def_away_before",
                               "att_diff_before", "def_diff_before"]], on="match_id")
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
    feat = feat.join(meta_feats)

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


def _meta_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Socioeconomic and World Cup pedigree covariates (static team priors)."""
    rows = []
    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        gdp_h, gdp_a = team_meta.gdp_per_capita(h), team_meta.gdp_per_capita(a)
        pop_h, pop_a = team_meta.population(h), team_meta.population(a)
        title_h, title_a = team_meta.wc_titles(h), team_meta.wc_titles(a)
        app_h, app_a = team_meta.wc_appearances(h), team_meta.wc_appearances(a)
        is_host_h = float(canonical(h) in HOST_NATIONS)
        is_host_a = float(canonical(a) in HOST_NATIONS)
        rows.append({
            "log_gdp_home": math.log(gdp_h), "log_gdp_away": math.log(gdp_a),
            "log_gdp_diff": math.log(gdp_h) - math.log(gdp_a),
            "log_pop_home": math.log(pop_h), "log_pop_away": math.log(pop_a),
            "log_pop_diff": math.log(pop_h) - math.log(pop_a),
            "wc_titles_home": title_h, "wc_titles_away": title_a,
            "wc_titles_diff": title_h - title_a,
            "wc_apps_home": app_h, "wc_apps_away": app_a,
            "wc_apps_diff": app_h - app_a,
            "is_host_home": is_host_h, "is_host_away": is_host_a,
            "host_diff": is_host_h - is_host_a,
        })
    return pd.DataFrame(rows, index=matches.index)


def _rank_features(matches: pd.DataFrame, rankings_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if rankings_df is None:
        return None
    index = rank_mod.RankIndex(rankings_df)
    rows = []
    for _, row in matches.iterrows():
        rh, ph = index.lookup(row["home_team"], row["date"])
        ra, pa = index.lookup(row["away_team"], row["date"])
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
    # Squad snapshots describe a specific era; applying them to older matches
    # would leak future player quality into historical training rows.
    valid_from = pd.Timestamp(squad_df.attrs.get("valid_from", "1900-01-01"))
    nan_feat = {f"{k}_{side}": float("nan")
                for k in ["squad_mean_top25", "squad_mean_top11", "squad_max", "squad_age"]
                for side in ["home", "away"]}
    nan_feat["squad_diff_top25"] = float("nan")

    rows = []
    for _, row in matches.iterrows():
        if row["date"] < valid_from:
            rows.append(dict(nan_feat))
            continue
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
        h_sq = feat["squad_mean_top25_home"]
        a_sq = feat["squad_mean_top25_away"]
        feat["squad_diff_top25"] = (
            h_sq - a_sq if not (pd.isna(h_sq) or pd.isna(a_sq)) else float("nan")
        )
        rows.append(feat)
    return pd.DataFrame(rows, index=matches.index)


