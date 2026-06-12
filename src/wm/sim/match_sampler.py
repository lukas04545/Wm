"""
Feature-row builder for hypothetical 2026 WC fixtures.
Converts (home_team, away_team, venue, date) into a feature row
compatible with the trained model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wm import config as cfg_mod
from wm.data.teams import canonical, load_confederations, CONFEDERATION_STRENGTH
from wm.features.context import haversine
from wm.data.build import CITY_ALTITUDES, TEAM_CAPITALS

# 2026 WC venue climate (pre-computed or loaded from cache)
VENUE_CLIMATE: dict[str, dict] = {
    "Estadio Azteca": {"temp_c": 18.0, "precip_mm": 2.5, "humidity_pct": 60, "altitude_m": 2216, "roof": 0},
    "Estadio BBVA": {"temp_c": 30.0, "precip_mm": 1.5, "humidity_pct": 55, "altitude_m": 539, "roof": 0},
    "Estadio Akron": {"temp_c": 22.0, "precip_mm": 2.0, "humidity_pct": 55, "altitude_m": 1560, "roof": 0},
    "BMO Field": {"temp_c": 22.0, "precip_mm": 1.8, "humidity_pct": 65, "altitude_m": 76, "roof": 0},
    "BC Place": {"temp_c": 18.0, "precip_mm": 1.2, "humidity_pct": 70, "altitude_m": 5, "roof": 1},
    "MetLife Stadium": {"temp_c": 25.0, "precip_mm": 2.5, "humidity_pct": 70, "altitude_m": 10, "roof": 0},
    "AT&T Stadium": {"temp_c": 33.0, "precip_mm": 1.5, "humidity_pct": 55, "altitude_m": 163, "roof": 1},
    "SoFi Stadium": {"temp_c": 26.0, "precip_mm": 0.2, "humidity_pct": 65, "altitude_m": 28, "roof": 1},
    "Rose Bowl": {"temp_c": 26.0, "precip_mm": 0.1, "humidity_pct": 65, "altitude_m": 100, "roof": 0},
    "Levi's Stadium": {"temp_c": 22.0, "precip_mm": 0.3, "humidity_pct": 65, "altitude_m": 10, "roof": 0},
    "Allegiant Stadium": {"temp_c": 38.0, "precip_mm": 0.1, "humidity_pct": 15, "altitude_m": 610, "roof": 1},
    "State Farm Stadium": {"temp_c": 36.0, "precip_mm": 0.8, "humidity_pct": 20, "altitude_m": 340, "roof": 1},
    "NRG Stadium": {"temp_c": 32.0, "precip_mm": 4.5, "humidity_pct": 75, "altitude_m": 12, "roof": 1},
    "Arrowhead Stadium": {"temp_c": 28.0, "precip_mm": 2.5, "humidity_pct": 65, "altitude_m": 271, "roof": 0},
    "Hard Rock Stadium": {"temp_c": 30.0, "precip_mm": 5.0, "humidity_pct": 80, "altitude_m": 3, "roof": 0},
    "Lincoln Financial Field": {"temp_c": 25.0, "precip_mm": 3.0, "humidity_pct": 70, "altitude_m": 7, "roof": 0},
    "Gillette Stadium": {"temp_c": 23.0, "precip_mm": 2.5, "humidity_pct": 70, "altitude_m": 9, "roof": 0},
    "Lumen Field": {"temp_c": 18.0, "precip_mm": 2.0, "humidity_pct": 75, "altitude_m": 7, "roof": 0},
}


def build_fixture_row(
    home: str,
    away: str,
    elo_home: float,
    elo_away: float,
    venue: str,
    date: pd.Timestamp,
    is_neutral: bool,
    is_host_home: bool,
    squad_df: pd.DataFrame | None,
    rankings_df: pd.DataFrame | None,
    cfg: cfg_mod.Config,
    historical_form: dict | None = None,
) -> pd.DataFrame:
    """
    Build a single feature row for a fixture.
    historical_form: optional dict with pre-computed form features for teams.
    """
    wc_path = cfg_mod.ROOT / "config" / "wc2026.yaml"
    confs = load_confederations(wc_path) if wc_path.exists() else {}

    climate = VENUE_CLIMATE.get(venue, {"temp_c": 25.0, "precip_mm": 2.0, "humidity_pct": 60, "altitude_m": 0, "roof": 0})

    # Travel distance
    hlat, hlon = TEAM_CAPITALS.get(canonical(home), (0.0, 0.0))
    alat, alon = TEAM_CAPITALS.get(canonical(away), (0.0, 0.0))
    venue_lat, venue_lon = 40.0, -100.0  # default USA center

    elo_diff = elo_home - elo_away
    elo_exp_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

    conf_h = confs.get(canonical(home), "OTHER")
    conf_a = confs.get(canonical(away), "OTHER")

    row: dict = {
        "match_id": -1,
        "date": date,
        "home_team": home,
        "away_team": away,
        "goals_home": np.nan,
        "goals_away": np.nan,
        "tournament_tier": 5,  # World Cup
        "is_neutral": int(is_neutral),
        "elo_home_before": elo_home,
        "elo_away_before": elo_away,
        "elo_diff_before": elo_diff,
        "elo_expected_home": elo_exp_home,
        "altitude_m": climate["altitude_m"],
        "altitude_diff_home": climate["altitude_m"] - 50.0,
        "altitude_diff_away": climate["altitude_m"] - 50.0,
        "is_high_altitude": int(climate["altitude_m"] > 1500),
        "weather_temp_c": climate["temp_c"],
        "weather_precip_mm": climate["precip_mm"],
        "weather_humidity": climate["humidity_pct"],
        "weather_roof": int(climate["roof"]),
        "travel_km_home": haversine(hlat, hlon, venue_lat, venue_lon),
        "travel_km_away": haversine(alat, alon, venue_lat, venue_lon),
        "conf_home": conf_h,
        "conf_away": conf_a,
        "conf_strength_home": CONFEDERATION_STRENGTH.get(conf_h, 0.35),
        "conf_strength_away": CONFEDERATION_STRENGTH.get(conf_a, 0.35),
        "conf_strength_diff": CONFEDERATION_STRENGTH.get(conf_h, 0.35) - CONFEDERATION_STRENGTH.get(conf_a, 0.35),
        "days_rest_home": 4.0,
        "days_rest_away": 4.0,
        "h2h_ppg_home": np.nan,
        "h2h_ppg_away": np.nan,
        "sample_weight": 1.0,
    }

    # Add form features with defaults (NaN = unknown → model handles)
    for side in ["home", "away"]:
        for W in cfg.features.form_windows:
            for stat in ["ppg", "gf_pg", "ga_pg", "gd_pg", "winrate"]:
                key = f"{side}_{stat}_{W}"
                if historical_form and key in historical_form:
                    row[key] = historical_form[key]
                else:
                    row[key] = np.nan
            row[f"{side}_n_matches_{W}"] = 0.0

    # Squad features
    if squad_df is not None:
        for side, team in [("home", home), ("away", away)]:
            sq = squad_df.loc[team] if team in squad_df.index else None
            if sq is not None:
                row[f"squad_mean_top25_{side}"] = float(sq.get("squad_mean_top25", np.nan))
                row[f"squad_mean_top11_{side}"] = float(sq.get("squad_mean_top11", np.nan))
                row[f"squad_max_{side}"] = float(sq.get("squad_max", np.nan))
                row[f"squad_age_{side}"] = float(sq.get("squad_mean_age", np.nan))
            else:
                row[f"squad_mean_top25_{side}"] = np.nan
                row[f"squad_mean_top11_{side}"] = np.nan
                row[f"squad_max_{side}"] = np.nan
                row[f"squad_age_{side}"] = np.nan
        if f"squad_mean_top25_home" in row and f"squad_mean_top25_away" in row:
            h_sq = row["squad_mean_top25_home"]
            a_sq = row["squad_mean_top25_away"]
            row["squad_diff_top25"] = h_sq - a_sq if not (pd.isna(h_sq) or pd.isna(a_sq)) else np.nan

    # FIFA rankings
    if rankings_df is not None:
        from wm.ingest.fifa_rankings import get_rank_as_of
        rh, ph = get_rank_as_of(rankings_df, home, date)
        ra, pa = get_rank_as_of(rankings_df, away, date)
        row["fifa_rank_home"] = rh
        row["fifa_rank_away"] = ra
        row["fifa_rank_diff"] = (ra - rh) if not (pd.isna(rh) or pd.isna(ra)) else np.nan
        row["fifa_points_diff"] = (ph - pa) if not (pd.isna(ph) or pd.isna(pa)) else np.nan

    return pd.DataFrame([row])
