"""Tests for real FBref club-stat features (replacing EA FC video-game ratings)."""
import numpy as np
import pandas as pd
import pytest

from wm.features.matrix import completed_season, _league_features, LEAGUE_COLS


def test_completed_season_is_leak_free():
    # Club season ends in May; before July, the latest COMPLETED season is prior year
    assert completed_season(pd.Timestamp("2019-03-15"), 2023) == 2018
    assert completed_season(pd.Timestamp("2019-08-01"), 2023) == 2019
    assert completed_season(pd.Timestamp("2019-07-10"), 2023) == 2019
    # Clamp to available data range (no future leakage past the dataset)
    assert completed_season(pd.Timestamp("2026-06-12"), 2023) == 2023


def _fake_league_df():
    idx = pd.MultiIndex.from_tuples(
        [("Brazil", 2018), ("Brazil", 2019), ("France", 2019)],
        names=["team", "season_end_year"],
    )
    data = {c: [1.0, 2.0, 3.0] for c in LEAGUE_COLS}
    df = pd.DataFrame(data, index=idx)
    df.attrs["season_max"] = 2023
    return df


def test_league_features_join_uses_completed_season():
    df = _fake_league_df()
    matches = pd.DataFrame({
        "date": [pd.Timestamp("2019-03-01")],  # → completed season 2018
        "home_team": ["Brazil"], "away_team": ["France"],
    })
    feats = _league_features(matches, df)
    # Brazil 2018 row has value 1.0; France has no 2018 row → NaN
    assert feats.iloc[0]["lg_fouls_per90_home"] == pytest.approx(1.0)
    assert np.isnan(feats.iloc[0]["lg_fouls_per90_away"])


def test_league_features_diff_columns():
    df = _fake_league_df()
    matches = pd.DataFrame({
        "date": [pd.Timestamp("2019-10-01")],  # → completed season 2019
        "home_team": ["Brazil"], "away_team": ["France"],
    })
    feats = _league_features(matches, df).iloc[0]
    # Brazil 2019 (2.0) vs France 2019 (3.0)
    assert feats["lg_talent_score_home"] == pytest.approx(2.0)
    assert feats["lg_talent_score_away"] == pytest.approx(3.0)
    assert feats["lg_talent_score_diff"] == pytest.approx(-1.0)


def test_league_features_absent_when_no_data():
    matches = pd.DataFrame({"date": [pd.Timestamp("2019-01-01")],
                            "home_team": ["X"], "away_team": ["Y"]})
    assert _league_features(matches, None) is None


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/players/fbref_big5_misc.rds").exists(),
    reason="FBref rds not downloaded",
)
def test_real_fbref_aggregation_has_fouls_and_nations():
    from pathlib import Path
    from wm.ingest.league_stats import nation_season_features
    f = nation_season_features(Path("data/raw"))
    assert f is not None
    assert "lg_fouls_per90" in f.columns
    assert (f["lg_fouls_per90"] >= 0).all()
    teams = set(f.index.get_level_values(0))
    assert {"Brazil", "France", "England", "Argentina"} <= teams
