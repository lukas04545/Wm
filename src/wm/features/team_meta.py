"""
Static national-team background covariates.

These are slowly-varying structural attributes used as strength priors:
  - GDP per capita (USD, ~2023) — wealth correlates with football infrastructure
  - population (millions) — talent-pool size
  - World Cup pedigree — titles and tournament appearances (through 2022)

They are treated as constant background features (not time-varying forecasts),
which is appropriate for covariates that change slowly relative to match cadence.
Unknown teams fall back to conservative low-end defaults so minnows are not
accidentally rewarded.

Values are approximate and curated for the ~60 teams most relevant to the
2026 cycle; everything else uses the defaults.
"""
from __future__ import annotations

from wm.data.teams import canonical

# team: (gdp_per_capita_usd, population_millions)
_ECONOMY: dict[str, tuple[float, float]] = {
    "United States": (80000, 335), "Mexico": (13800, 128), "Canada": (54000, 40),
    "Brazil": (10300, 216), "Argentina": (13700, 46), "Uruguay": (20800, 3.4),
    "Colombia": (6900, 52), "Ecuador": (6500, 18), "Paraguay": (6200, 6.8),
    "Peru": (7900, 34), "Chile": (17100, 19.6), "Venezuela": (3500, 28),
    "Bolivia": (3700, 12),
    "France": (44400, 68), "Germany": (52800, 84), "Spain": (32700, 48),
    "England": (49000, 56), "Portugal": (27400, 10.3), "Netherlands": (61000, 17.8),
    "Belgium": (53400, 11.7), "Italy": (38400, 59), "Croatia": (18400, 3.9),
    "Switzerland": (93500, 8.8), "Austria": (56500, 9.1), "Poland": (22100, 37),
    "Serbia": (11400, 6.6), "Denmark": (68000, 5.9), "Sweden": (56300, 10.5),
    "Norway": (87900, 5.5), "Scotland": (38000, 5.5), "Turkey": (13400, 85),
    "Czech Republic": (30400, 10.5), "Hungary": (22500, 9.6), "Ukraine": (4500, 38),
    "Wales": (34000, 3.1), "Bosnia and Herzegovina": (7600, 3.2),
    "Japan": (33800, 125), "South Korea": (33100, 51.7), "Australia": (64500, 26),
    "Iran": (4700, 88), "Saudi Arabia": (30400, 36.9), "Qatar": (80200, 2.7),
    "Iraq": (5900, 43), "Uzbekistan": (2300, 35), "Jordan": (4200, 11.3),
    "China": (12600, 1412),
    "Morocco": (3700, 37), "Senegal": (1700, 17.7), "Nigeria": (2200, 223),
    "Ivory Coast": (2700, 28), "Egypt": (3500, 110), "Algeria": (5000, 45),
    "Cameroon": (1700, 28), "Ghana": (2200, 33.5), "Tunisia": (3900, 12.4),
    "South Africa": (6200, 60), "DR Congo": (650, 102), "Mali": (900, 22),
    "Cape Verde": (4300, 0.6), "Curaçao": (15000, 0.15),
    "New Zealand": (48400, 5.2),
    "Haiti": (1700, 11.6), "Panama": (18000, 4.4), "Honduras": (3100, 10.4),
    "Jamaica": (6800, 2.8), "Costa Rica": (13400, 5.2),
}
_DEFAULT_GDP, _DEFAULT_POP = 5000.0, 15.0

# team: (world_cup_titles, world_cup_appearances through 2022)
_PEDIGREE: dict[str, tuple[int, int]] = {
    "Brazil": (5, 22), "Germany": (4, 20), "Italy": (4, 18), "Argentina": (3, 18),
    "France": (2, 16), "Uruguay": (2, 14), "England": (1, 16), "Spain": (1, 16),
    "Netherlands": (0, 11), "Mexico": (0, 17), "Sweden": (0, 12), "Belgium": (0, 14),
    "Serbia": (0, 13), "Switzerland": (0, 12), "Poland": (0, 9), "United States": (0, 11),
    "Portugal": (0, 8), "Croatia": (0, 6), "Austria": (0, 7), "Hungary": (0, 9),
    "Chile": (0, 9), "Paraguay": (0, 8), "South Korea": (0, 11), "Japan": (0, 7),
    "Czech Republic": (0, 10), "Denmark": (0, 6), "Colombia": (0, 6), "Russia": (0, 11),
    "Scotland": (0, 8), "Nigeria": (0, 6), "Cameroon": (0, 8), "Morocco": (0, 6),
    "Ghana": (0, 4), "Australia": (0, 6), "Iran": (0, 6), "Saudi Arabia": (0, 6),
    "Ecuador": (0, 4), "Tunisia": (0, 6), "Costa Rica": (0, 6), "Algeria": (0, 5),
    "Peru": (0, 5), "Senegal": (0, 3), "Norway": (0, 3), "Ivory Coast": (0, 3),
    "Egypt": (0, 3), "Turkey": (0, 2),
}


def gdp_per_capita(team: str) -> float:
    return _ECONOMY.get(canonical(team), (_DEFAULT_GDP, _DEFAULT_POP))[0]


def population(team: str) -> float:
    return _ECONOMY.get(canonical(team), (_DEFAULT_GDP, _DEFAULT_POP))[1]


def wc_titles(team: str) -> int:
    return _PEDIGREE.get(canonical(team), (0, 0))[0]


def wc_appearances(team: str) -> int:
    return _PEDIGREE.get(canonical(team), (0, 0))[1]
