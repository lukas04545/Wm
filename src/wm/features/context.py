"""Context features: travel distance, altitude, weather, host status."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from wm.data.build import TEAM_CAPITALS
from wm.data.teams import canonical


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _team_home_latlon(team: str) -> tuple[float, float]:
    return TEAM_CAPITALS.get(canonical(team), (0.0, 0.0))


def compute_travel(matches: pd.DataFrame) -> pd.DataFrame:
    """Approximate travel distance for each team as distance from capital to match venue."""
    rows = []
    for _, row in matches.iterrows():
        vlat = float(row.get("lat", 0.0) or 0.0)
        vlon = float(row.get("lon", 0.0) or 0.0)
        hlat, hlon = _team_home_latlon(str(row["home_team"]))
        alat, alon = _team_home_latlon(str(row["away_team"]))
        rows.append({
            "travel_km_home": haversine(hlat, hlon, vlat, vlon) if vlat and vlon else float("nan"),
            "travel_km_away": haversine(alat, alon, vlat, vlon) if vlat and vlon else float("nan"),
        })
    return pd.DataFrame(rows, index=matches.index)


def compute_altitude_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Altitude at venue and relative to team's home country altitude."""
    home_alt = {
        "Mexico": 2250.0, "Bolivia": 3625.0, "Colombia": 2625.0, "Ecuador": 2850.0,
        "Peru": 100.0, "Argentina": 25.0, "Brazil": 900.0, "Chile": 520.0,
        "Switzerland": 400.0, "Austria": 800.0, "Spain": 650.0, "Germany": 200.0,
    }
    rows = []
    for _, row in matches.iterrows():
        venue_alt = float(row.get("altitude_m", 0.0) or 0.0)
        home_native = home_alt.get(str(row.get("home_team", "")), 50.0)
        away_native = home_alt.get(str(row.get("away_team", "")), 50.0)
        rows.append({
            "altitude_m": venue_alt,
            "altitude_diff_home": venue_alt - home_native,
            "altitude_diff_away": venue_alt - away_native,
            "is_high_altitude": float(venue_alt > 1500),
        })
    return pd.DataFrame(rows, index=matches.index)


def attach_weather(matches: pd.DataFrame, weather_cache: dict[str, dict]) -> pd.DataFrame:
    """Attach weather features from a venue→weather dict (from open-meteo)."""
    rows = []
    for _, row in matches.iterrows():
        city = str(row.get("city", ""))
        country = str(row.get("country", ""))
        key = f"{city},{country}"
        weather = weather_cache.get(key, weather_cache.get(city, {}))
        rows.append({
            "weather_temp_c": weather.get("temp_c", float("nan")),
            "weather_precip_mm": weather.get("precip_mm", float("nan")),
            "weather_humidity": weather.get("humidity_pct", float("nan")),
            "weather_roof": float(weather.get("roof", 0)),
        })
    return pd.DataFrame(rows, index=matches.index)
