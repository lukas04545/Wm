"""Build canonical match table from raw ingested data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from wm.data.teams import canonical
from wm.ingest import results as res_mod


# Static venue altitude / lat-lon lookup (city level; used for historical matches)
CITY_ALTITUDES: dict[str, float] = {
    "Mexico City": 2250.0,
    "Guadalajara": 1560.0,
    "Monterrey": 539.0,
    "Bogotá": 2625.0,
    "Quito": 2850.0,
    "La Paz": 3625.0,
    "Addis Ababa": 2355.0,
    "Nairobi": 1795.0,
    "Johannesburg": 1753.0,
    "Pretoria": 1330.0,
    "Kampala": 1190.0,
    "Denver": 1609.0,
    "Calgary": 1045.0,
    "Madrid": 650.0,
    "Bern": 540.0,
    "Tehran": 1200.0,
    "Kabul": 1791.0,
    "Kathmandu": 1400.0,
}

TEAM_CAPITALS: dict[str, tuple[float, float]] = {
    "United States": (38.9, -77.0),
    "Mexico": (19.4, -99.1),
    "Canada": (45.4, -75.7),
    "Brazil": (-15.8, -47.9),
    "Argentina": (-34.6, -58.4),
    "Germany": (52.5, 13.4),
    "France": (48.9, 2.4),
    "Spain": (40.4, -3.7),
    "England": (51.5, -0.1),
    "Portugal": (38.7, -9.1),
    "Netherlands": (52.4, 4.9),
    "Belgium": (50.8, 4.4),
    "Italy": (41.9, 12.5),
    "Croatia": (45.8, 16.0),
    "Japan": (35.7, 139.7),
    "South Korea": (37.6, 127.0),
    "Australia": (-35.3, 149.1),
    "Morocco": (34.0, -6.9),
    "Senegal": (14.7, -17.5),
    "Nigeria": (9.1, 7.4),
    "Egypt": (30.1, 31.4),
    "Saudi Arabia": (24.7, 46.7),
    "Iran": (35.7, 51.4),
    "Colombia": (4.7, -74.1),
    "Uruguay": (-34.9, -56.2),
    "Ecuador": (-0.2, -78.5),
    "Switzerland": (46.9, 7.4),
    "Sweden": (59.3, 18.1),
    "Norway": (59.9, 10.7),
    "Denmark": (55.7, 12.6),
    "Austria": (48.2, 16.4),
    "Turkey": (39.9, 32.9),
    "Poland": (52.2, 21.0),
    "Serbia": (44.8, 20.5),
    "Hungary": (47.5, 19.0),
    "Scotland": (55.9, -3.2),
    "Wales": (51.5, -3.2),
    "Paraguay": (-25.3, -57.6),
    "Bolivia": (-16.5, -68.1),
    "Peru": (-12.1, -77.0),
    "Chile": (-33.5, -70.6),
    "Venezuela": (10.5, -66.9),
    "Panama": (9.0, -79.5),
    "Honduras": (14.1, -87.2),
    "Jamaica": (18.0, -76.8),
    "Costa Rica": (9.9, -84.1),
    "Haiti": (18.6, -72.3),
    "Ivory Coast": (5.4, -4.0),
    "Ghana": (5.6, -0.2),
    "Cameroon": (3.9, 11.5),
    "Algeria": (36.7, 3.1),
    "Tunisia": (36.8, 10.2),
    "South Africa": (-25.7, 28.2),
    "DR Congo": (-4.3, 15.3),
    "Uzbekistan": (41.3, 69.3),
    "Iraq": (33.3, 44.4),
    "Jordan": (31.9, 35.9),
    "Qatar": (25.3, 51.5),
    "New Zealand": (-41.3, 174.8),
    "Curaçao": (12.1, -68.9),
    "Cape Verde": (14.9, -23.5),
    "Bosnia and Herzegovina": (43.8, 18.4),
    "Czech Republic": (50.1, 14.4),
}


def build(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical match table with normalized team names and enriched metadata."""
    df = res_mod.clean(results_df)
    df["home_team"] = df["home_team"].apply(canonical)
    df["away_team"] = df["away_team"].apply(canonical)
    df["altitude_m"] = df["city"].map(CITY_ALTITUDES).fillna(0.0)
    df["lat"] = df["home_team"].map(lambda t: TEAM_CAPITALS.get(t, (0.0, 0.0))[0])
    df["lon"] = df["home_team"].map(lambda t: TEAM_CAPITALS.get(t, (0.0, 0.0))[1])
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    from wm.data.io import save_df
    save_df(df, path)


def load(path: Path) -> pd.DataFrame:
    from wm.data.io import load_df
    return load_df(path)
