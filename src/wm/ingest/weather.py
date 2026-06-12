"""Fetch climate normals and historical weather from Open-Meteo (free, no key)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

# Open-Meteo climate normals endpoint (30-year averages by month)
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _get(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def fetch_climate_normal(lat: float, lon: float, month: int, cache_dir: Path) -> dict[str, float]:
    """Return 30-year climate normal for a location in a given month (1-12)."""
    cache_file = cache_dir / f"climate_{lat:.3f}_{lon:.3f}_m{month:02d}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1991-01-01",
        "end_date": "2020-12-31",
        "models": "EC_Earth3P_HR",
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_mean",
    }
    data = _get(CLIMATE_URL, params)

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    precips = daily.get("precipitation_sum", [])
    humidities = daily.get("relative_humidity_2m_mean", [])

    month_temps = [t for d, t in zip(dates, temps) if d and f"-{month:02d}-" in d and t is not None]
    month_precips = [p for d, p in zip(dates, precips) if d and f"-{month:02d}-" in d and p is not None]
    month_hum = [h for d, h in zip(dates, humidities) if d and f"-{month:02d}-" in d and h is not None]

    result = {
        "temp_c": sum(month_temps) / len(month_temps) if month_temps else 20.0,
        "precip_mm": sum(month_precips) / len(month_precips) if month_precips else 0.0,
        "humidity_pct": sum(month_hum) / len(month_hum) if month_hum else 50.0,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result, f)

    return result


def fetch_venue_climates(venues: dict, cache_dir: Path) -> dict[str, dict[str, float]]:
    """Fetch June/July climate normals for all WC 2026 venues (WC months)."""
    result = {}
    for name, info in venues.items():
        lat, lon = info["lat"], info["lon"]
        june = fetch_climate_normal(lat, lon, 6, cache_dir)
        july = fetch_climate_normal(lat, lon, 7, cache_dir)
        result[name] = {
            "temp_c": (june["temp_c"] + july["temp_c"]) / 2,
            "precip_mm": (june["precip_mm"] + july["precip_mm"]) / 2,
            "humidity_pct": (june["humidity_pct"] + july["humidity_pct"]) / 2,
            "altitude_m": info.get("altitude_m", 0),
            "roof": int(info.get("roof", False)),
        }
    return result
