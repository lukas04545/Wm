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

# 2026 WC venue climate (pre-computed or loaded from cache); lat/lon for travel
VENUE_CLIMATE: dict[str, dict] = {
    "Estadio Azteca": {"temp_c": 18.0, "precip_mm": 2.5, "humidity_pct": 60, "altitude_m": 2216, "roof": 0, "lat": 19.303, "lon": -99.151},
    "Estadio BBVA": {"temp_c": 30.0, "precip_mm": 1.5, "humidity_pct": 55, "altitude_m": 539, "roof": 0, "lat": 25.669, "lon": -100.246},
    "Estadio Akron": {"temp_c": 22.0, "precip_mm": 2.0, "humidity_pct": 55, "altitude_m": 1560, "roof": 0, "lat": 20.688, "lon": -103.464},
    "BMO Field": {"temp_c": 22.0, "precip_mm": 1.8, "humidity_pct": 65, "altitude_m": 76, "roof": 0, "lat": 43.633, "lon": -79.419},
    "BC Place": {"temp_c": 18.0, "precip_mm": 1.2, "humidity_pct": 70, "altitude_m": 5, "roof": 1, "lat": 49.277, "lon": -123.112},
    "MetLife Stadium": {"temp_c": 25.0, "precip_mm": 2.5, "humidity_pct": 70, "altitude_m": 10, "roof": 0, "lat": 40.813, "lon": -74.075},
    "AT&T Stadium": {"temp_c": 33.0, "precip_mm": 1.5, "humidity_pct": 55, "altitude_m": 163, "roof": 1, "lat": 32.748, "lon": -97.093},
    "SoFi Stadium": {"temp_c": 26.0, "precip_mm": 0.2, "humidity_pct": 65, "altitude_m": 28, "roof": 1, "lat": 33.954, "lon": -118.339},
    "Rose Bowl": {"temp_c": 26.0, "precip_mm": 0.1, "humidity_pct": 65, "altitude_m": 100, "roof": 0, "lat": 34.162, "lon": -118.168},
    "Levi's Stadium": {"temp_c": 22.0, "precip_mm": 0.3, "humidity_pct": 65, "altitude_m": 10, "roof": 0, "lat": 37.403, "lon": -121.970},
    "Allegiant Stadium": {"temp_c": 38.0, "precip_mm": 0.1, "humidity_pct": 15, "altitude_m": 610, "roof": 1, "lat": 36.091, "lon": -115.184},
    "State Farm Stadium": {"temp_c": 36.0, "precip_mm": 0.8, "humidity_pct": 20, "altitude_m": 340, "roof": 1, "lat": 33.528, "lon": -112.263},
    "NRG Stadium": {"temp_c": 32.0, "precip_mm": 4.5, "humidity_pct": 75, "altitude_m": 12, "roof": 1, "lat": 29.685, "lon": -95.410},
    "Arrowhead Stadium": {"temp_c": 28.0, "precip_mm": 2.5, "humidity_pct": 65, "altitude_m": 271, "roof": 0, "lat": 39.049, "lon": -94.484},
    "Hard Rock Stadium": {"temp_c": 30.0, "precip_mm": 5.0, "humidity_pct": 80, "altitude_m": 3, "roof": 0, "lat": 25.958, "lon": -80.239},
    "Lincoln Financial Field": {"temp_c": 25.0, "precip_mm": 3.0, "humidity_pct": 70, "altitude_m": 7, "roof": 0, "lat": 39.901, "lon": -75.168},
    "Gillette Stadium": {"temp_c": 23.0, "precip_mm": 2.5, "humidity_pct": 70, "altitude_m": 9, "roof": 0, "lat": 42.091, "lon": -71.264},
    "Lumen Field": {"temp_c": 18.0, "precip_mm": 2.0, "humidity_pct": 75, "altitude_m": 7, "roof": 0, "lat": 47.595, "lon": -122.332},
    "Mercedes-Benz Stadium": {"temp_c": 31.0, "precip_mm": 3.5, "humidity_pct": 70, "altitude_m": 320, "roof": 1, "lat": 33.755, "lon": -84.401},
}

# Dataset city names (martj42 results.csv) → venue keys above
CITY_TO_VENUE: dict[str, str] = {
    "Mexico City": "Estadio Azteca",
    "Zapopan": "Estadio Akron",          # Guadalajara metro
    "Guadalajara": "Estadio Akron",
    "Guadalupe": "Estadio BBVA",         # Monterrey metro
    "Monterrey": "Estadio BBVA",
    "Toronto": "BMO Field",
    "Vancouver": "BC Place",
    "East Rutherford": "MetLife Stadium",
    "Arlington": "AT&T Stadium",
    "Inglewood": "SoFi Stadium",
    "Los Angeles": "SoFi Stadium",
    "Santa Clara": "Levi's Stadium",
    "Houston": "NRG Stadium",
    "Kansas City": "Arrowhead Stadium",
    "Miami Gardens": "Hard Rock Stadium",
    "Miami": "Hard Rock Stadium",
    "Philadelphia": "Lincoln Financial Field",
    "Foxborough": "Gillette Stadium",
    "Boston": "Gillette Stadium",
    "Seattle": "Lumen Field",
    "Atlanta": "Mercedes-Benz Stadium",
}


def venue_for_city(city: str, default: str = "MetLife Stadium") -> str:
    return CITY_TO_VENUE.get(city, default)


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
    team_state: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Build a single feature row for a fixture.
    team_state: {team: state} from wm.features.state.compute_team_state —
    supplies real current form/att/def features instead of NaN.
    historical_form: legacy per-key override dict (kept for compatibility).
    """
    wc_path = cfg_mod.ROOT / "config" / "wc2026.yaml"
    confs = load_confederations(wc_path) if wc_path.exists() else {}

    climate = VENUE_CLIMATE.get(venue, {"temp_c": 25.0, "precip_mm": 2.0, "humidity_pct": 60, "altitude_m": 0, "roof": 0})

    # Travel distance (real venue coordinates when known)
    hlat, hlon = TEAM_CAPITALS.get(canonical(home), (0.0, 0.0))
    alat, alon = TEAM_CAPITALS.get(canonical(away), (0.0, 0.0))
    venue_lat = float(climate.get("lat", 40.0))
    venue_lon = float(climate.get("lon", -100.0))

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

    # Form + rating-state features from the team-state snapshot
    # (NaN where unknown → model handles)
    state_h = (team_state or {}).get(home, {})
    state_a = (team_state or {}).get(away, {})
    for side, st in [("home", state_h), ("away", state_a)]:
        for W in cfg.features.form_windows:
            for stat in ["ppg", "gf_pg", "ga_pg", "gd_pg", "winrate",
                         "perf_vs_elo", "opp_elo"]:
                key = f"{side}_{stat}_{W}"
                if historical_form and key in historical_form:
                    row[key] = historical_form[key]
                else:
                    row[key] = st.get(f"{stat}_{W}", np.nan)
            row[f"{side}_n_matches_{W}"] = st.get(f"n_matches_{W}", 0.0)
        row[f"{side}_ewma_gf"] = st.get("ewma_gf", np.nan)
        row[f"{side}_ewma_ga"] = st.get("ewma_ga", np.nan)
    row["att_home_before"] = state_h.get("att", np.nan)
    row["def_home_before"] = state_h.get("def", np.nan)
    row["att_away_before"] = state_a.get("att", np.nan)
    row["def_away_before"] = state_a.get("def", np.nan)
    if state_h and state_a:
        row["att_diff_before"] = state_h.get("att", 0.0) - state_a.get("att", 0.0)
        row["def_diff_before"] = state_h.get("def", 0.0) - state_a.get("def", 0.0)
    else:
        row["att_diff_before"] = np.nan
        row["def_diff_before"] = np.nan

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

    # Socioeconomic & World Cup pedigree covariates (must match training matrix)
    import math
    from wm.features import team_meta
    from wm.features.matrix import HOST_NATIONS
    gdp_h, gdp_a = team_meta.gdp_per_capita(home), team_meta.gdp_per_capita(away)
    pop_h, pop_a = team_meta.population(home), team_meta.population(away)
    title_h, title_a = team_meta.wc_titles(home), team_meta.wc_titles(away)
    app_h, app_a = team_meta.wc_appearances(home), team_meta.wc_appearances(away)
    host_h = float(canonical(home) in HOST_NATIONS)
    host_a = float(canonical(away) in HOST_NATIONS)
    row.update({
        "log_gdp_home": math.log(gdp_h), "log_gdp_away": math.log(gdp_a),
        "log_gdp_diff": math.log(gdp_h) - math.log(gdp_a),
        "log_pop_home": math.log(pop_h), "log_pop_away": math.log(pop_a),
        "log_pop_diff": math.log(pop_h) - math.log(pop_a),
        "wc_titles_home": title_h, "wc_titles_away": title_a,
        "wc_titles_diff": title_h - title_a,
        "wc_apps_home": app_h, "wc_apps_away": app_a, "wc_apps_diff": app_h - app_a,
        "is_host_home": host_h, "is_host_away": host_a, "host_diff": host_h - host_a,
    })

    return pd.DataFrame([row])
