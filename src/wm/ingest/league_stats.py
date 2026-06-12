"""
Real club-football performance features from FBref Big-5 league data.

Replaces the EA FC / FIFA video-game player ratings with *real* on-pitch
statistics: every player-season in the Premier League, La Liga, Serie A,
Bundesliga and Ligue 1 (2010-2023), with goals, assists, xG, minutes, and
discipline — including **fouls committed and drawn**, cards, tackles,
interceptions and aerial duels.

Source: JaseZiv/worldfootballR_data (pre-scraped FBref tables, committed as
.rds). Parsed with the pure-Python `rdata` reader (no R, no C extensions —
stays Termux-portable). If `rdata` or the files are unavailable the features
degrade gracefully to absent, like every other optional source.

Players are aggregated to their NATIONAL TEAM per season, so each
international match can join the club form of that nation's players for the
relevant season (leak-free, time-varying — see features/matrix.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wm.ingest.cache import download

BASE = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
        "data/fb_big5_advanced_season_stats")
MISC_URL = f"{BASE}/big5_player_misc.rds"
STANDARD_URL = f"{BASE}/big5_player_standard.rds"

# Static league-quality weights (UEFA-coefficient style) — used only to weight
# a "talent playing in strong leagues" score. Not fabricated player data.
LEAGUE_WEIGHT = {
    "Premier League": 1.00,
    "La Liga": 0.97,
    "Serie A": 0.95,
    "Bundesliga": 0.94,
    "Ligue 1": 0.88,
}

MIN_90S = 5.0          # ignore cameo seasons (< ~450 minutes)
SEASON_MIN, SEASON_MAX = 2010, 2023

# FBref 3-letter nation codes → canonical team names (WC2026 + major nations)
FBREF_NATION = {
    "ESP": "Spain", "FRA": "France", "ITA": "Italy", "GER": "Germany",
    "ENG": "England", "BRA": "Brazil", "ARG": "Argentina", "POR": "Portugal",
    "NED": "Netherlands", "SEN": "Senegal", "SRB": "Serbia", "BEL": "Belgium",
    "CIV": "Ivory Coast", "URU": "Uruguay", "SUI": "Switzerland", "CRO": "Croatia",
    "MAR": "Morocco", "DEN": "Denmark", "AUT": "Austria", "POL": "Poland",
    "ALG": "Algeria", "MLI": "Mali", "CMR": "Cameroon", "GHA": "Ghana",
    "IRL": "Ireland", "MEX": "Mexico", "RSA": "South Africa", "KOR": "South Korea",
    "CZE": "Czech Republic", "CAN": "Canada", "BIH": "Bosnia and Herzegovina",
    "QAT": "Qatar", "HAI": "Haiti", "SCO": "Scotland", "USA": "United States",
    "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkey", "CUW": "Curaçao",
    "ECU": "Ecuador", "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia",
    "EGY": "Egypt", "IRN": "Iran", "NZL": "New Zealand", "CPV": "Cape Verde",
    "KSA": "Saudi Arabia", "IRQ": "Iraq", "NOR": "Norway", "JOR": "Jordan",
    "COD": "DR Congo", "UZB": "Uzbekistan", "COL": "Colombia", "PAN": "Panama",
    "NGA": "Nigeria", "WAL": "Wales", "CHI": "Chile", "PER": "Peru",
    "VEN": "Venezuela", "BOL": "Bolivia", "HON": "Honduras", "JAM": "Jamaica",
    "CRC": "Costa Rica", "PAN ": "Panama", "GRE": "Greece", "HUN": "Hungary",
    "UKR": "Ukraine", "SVN": "Slovenia", "SVK": "Slovakia", "ROU": "Romania",
    "ISL": "Iceland", "FIN": "Finland", "ALB": "Albania", "MKD": "North Macedonia",
    "GAB": "Gabon", "GUI": "Guinea", "BFA": "Burkina Faso", "COG": "Congo",
    "ANG": "Angola", "ZAM": "Zambia", "TOG": "Togo", "BEN": "Benin",
}


def fetch(raw_dir: Path, registry: Path, force: bool = False) -> Path:
    """Download the FBref Big-5 misc + standard season-stat tables."""
    dest_dir = raw_dir / "players"
    download(MISC_URL, dest_dir / "fbref_big5_misc.rds", registry, force=force)
    return download(STANDARD_URL, dest_dir / "fbref_big5_standard.rds", registry, force=force)


def _read_rds(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        import warnings
        import rdata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = rdata.read_rds(str(path))
    except Exception:
        return None
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def load_player_seasons(raw_dir: Path) -> pd.DataFrame | None:
    """Merge misc (discipline) + standard (attacking) per player-season."""
    misc = _read_rds(raw_dir / "players" / "fbref_big5_misc.rds")
    if misc is None:
        return None
    keep_misc = ["Season_End_Year", "Squad", "Comp", "Player", "Nation",
                 "Mins_Per_90", "CrdY", "CrdR", "Fls", "Fld", "Int", "TklW",
                 "Won_Aerial", "Lost_Aerial"]
    misc = misc[[c for c in keep_misc if c in misc.columns]].copy()

    std = _read_rds(raw_dir / "players" / "fbref_big5_standard.rds")
    if std is not None:
        keep_std = ["Season_End_Year", "Squad", "Player", "Comp", "Nation",
                    "Gls", "Ast", "xG_Expected", "xAG_Expected"]
        std = std[[c for c in keep_std if c in std.columns]].copy()
        df = misc.merge(std, on=["Season_End_Year", "Squad", "Player", "Comp", "Nation"],
                        how="left")
    else:
        df = misc

    for c in ["Mins_Per_90", "CrdY", "CrdR", "Fls", "Fld", "Int", "TklW",
              "Won_Aerial", "Lost_Aerial", "Gls", "Ast", "xG_Expected", "xAG_Expected"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["team"] = df["Nation"].map(FBREF_NATION)
    df = df.dropna(subset=["team", "Mins_Per_90"])
    df = df[df["Mins_Per_90"] >= MIN_90S]
    df["lw"] = df["Comp"].map(LEAGUE_WEIGHT).fillna(0.85)
    return df


def nation_season_features(raw_dir: Path) -> pd.DataFrame | None:
    """
    Return per-(team, season) minutes-weighted real club-form features,
    indexed by [team, season_end_year]. Columns prefixed `lg_`.
    """
    df = load_player_seasons(raw_dir)
    if df is None or df.empty:
        return None

    rows = []
    for (team, season), g in df.groupby(["team", "Season_End_Year"]):
        nineties = g["Mins_Per_90"].sum()
        if nineties <= 0:
            continue
        aer_won, aer_lost = g["Won_Aerial"].sum(), g["Lost_Aerial"].sum()
        rows.append({
            "team": team,
            "season_end_year": int(season),
            "lg_n_players": int((g["Mins_Per_90"] >= MIN_90S).sum()),
            "lg_total_90s": float(nineties),
            "lg_fouls_per90": g["Fls"].sum() / nineties,
            "lg_fouls_drawn_per90": g["Fld"].sum() / nineties,
            "lg_cards_per90": (g["CrdY"].sum() + g["CrdR"].sum()) / nineties,
            "lg_ga_per90": (g["Gls"].sum() + g["Ast"].sum()) / nineties,
            "lg_xgxag_per90": (g["xG_Expected"].sum() + g["xAG_Expected"].sum()) / nineties,
            "lg_def_per90": (g["TklW"].sum() + g["Int"].sum()) / nineties,
            "lg_aerial_pct": aer_won / (aer_won + aer_lost) if (aer_won + aer_lost) > 0 else np.nan,
            "lg_talent_score": float((g["Mins_Per_90"] * g["lw"]).sum()),
        })
    out = pd.DataFrame(rows).set_index(["team", "season_end_year"]).sort_index()
    out.attrs["season_max"] = SEASON_MAX
    return out
