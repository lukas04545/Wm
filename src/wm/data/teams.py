"""Canonical team name mapping and confederation lookup."""
from __future__ import annotations

from pathlib import Path

import yaml

# Alias → canonical name (handles historical names and abbreviations)
ALIASES: dict[str, str] = {
    "United States": "United States",
    "USA": "United States",
    "US": "United States",
    "America": "United States",
    "England": "England",
    "UK": "England",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Korea": "South Korea",
    "Republic of Korea": "South Korea",
    "North Korea": "North Korea",
    "DPR Korea": "North Korea",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Zaire": "DR Congo",
    "Turkey": "Turkey",
    "Türkiye": "Turkey",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "New Zealand": "New Zealand",
    "West Germany": "Germany",
    "Czechoslovakia": "Czech Republic",  # simplification; some results go to Slovakia
    "Yugoslavia": "Serbia",  # imperfect but reasonable
    "Soviet Union": "Russia",
    "Netherlands Antilles": "Curaçao",
    # FIFA-ranking style names
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Korea DPR": "North Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei",
    "Hong Kong, China": "Hong Kong",
    "Macau, China": "Macau",
    "Chinese Taipei": "Taiwan",
    "FYR Macedonia": "North Macedonia",
    "Macedonia FYR": "North Macedonia",
    "Swaziland": "Eswatini",
}


def canonical(name: str) -> str:
    """Return the canonical team name for name."""
    if not name:
        return name
    return ALIASES.get(name, name)


def load_confederations(config_path: Path) -> dict[str, str]:
    """Return team → confederation mapping from wc2026.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    result: dict[str, str] = {}
    for conf, teams in cfg.get("confederations", {}).items():
        for team in teams:
            result[canonical(team)] = conf
    return result


CONFEDERATION_STRENGTH = {
    "UEFA": 0.7,
    "CONMEBOL": 0.65,
    "CONCACAF": 0.45,
    "CAF": 0.42,
    "AFC": 0.40,
    "OFC": 0.25,
}
