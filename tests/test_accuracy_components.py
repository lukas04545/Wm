"""Tests for the accuracy-push components: DC tau, stacker, vector scaling,
attack/defence Elo, and opponent-adjusted form."""
import numpy as np
import pandas as pd
import pytest

from wm.config import EloConfig


def test_dc_tau_grid_normalized_and_inflates_draws():
    from wm.models.ensemble import independent_poisson_grid, scoreline_grid_to_wdl
    plain = independent_poisson_grid(1.3, 1.1, max_g=10)
    corrected = independent_poisson_grid(1.3, 1.1, max_g=10, rho=-0.05)
    assert abs(corrected.sum() - 1.0) < 1e-9
    # negative rho inflates 0-0 and 1-1 relative weight → higher draw prob
    assert scoreline_grid_to_wdl(corrected)[1] > scoreline_grid_to_wdl(plain)[1]


def test_fit_dc_rho_recovers_sign():
    """On draw-heavy synthetic low scores, fitted rho must be negative."""
    from wm.models.ensemble import fit_dc_rho
    rng = np.random.default_rng(0)
    n = 2000
    lh = np.full(n, 1.2)
    la = np.full(n, 1.0)
    gh = rng.poisson(lh)
    ga = rng.poisson(la)
    # inject extra 0-0 and 1-1 draws
    extra = rng.random(n) < 0.08
    gh[extra] = ga[extra] = rng.integers(0, 2, extra.sum())
    rho = fit_dc_rho(lh, la, gh.astype(float), ga.astype(float))
    assert rho < 0


def test_stacked_blender_valid_probs():
    from wm.models.ensemble import StackedBlender
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    noise = rng.dirichlet(np.ones(3), n)
    ctx = pd.DataFrame({
        "elo_diff_before": rng.normal(0, 200, n),
        "tournament_tier": rng.integers(1, 6, n),
        "is_neutral": rng.integers(0, 2, n),
    })
    st = StackedBlender().fit([good, noise], ctx, y)
    p = st.predict_proba([good, noise], ctx)
    assert p.shape == (n, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    # must learn to trust the informative branch
    ll = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    assert ll < 0.7


def test_vector_scaling_beats_or_matches_identity():
    from wm.models.calibrate import VectorScalingCalibrator
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.integers(0, 3, n)
    # systematically draw-underconfident probabilities
    p = np.full((n, 3), 1 / 3.0)
    p[np.arange(n), y] += 0.2
    p[:, 1] *= 0.7
    p /= p.sum(axis=1, keepdims=True)
    cal = VectorScalingCalibrator().fit(p, y)
    out = cal.predict(p)
    assert np.allclose(out.sum(axis=1), 1.0)
    ll_raw = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    ll_cal = -np.mean(np.log(out[np.arange(n), y] + 1e-12))
    assert ll_cal <= ll_raw + 1e-9


def _mk_matches(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["goal_diff"] = df["goals_home"] - df["goals_away"]
    df["outcome"] = np.sign(df["goal_diff"]).astype(int)
    df["is_neutral"] = True
    df["tournament_tier"] = 3
    df["match_id"] = range(len(df))
    return df


def test_attack_defence_ratings_move_correctly():
    from wm.features.elo_rolling import compute
    matches = _mk_matches([
        {"date": "2020-01-01", "home_team": "Strong", "away_team": "Weak",
         "goals_home": 4, "goals_away": 0},
        {"date": "2020-02-01", "home_team": "Strong", "away_team": "Weak",
         "goals_home": 3, "goals_away": 0},
    ])
    _, state = compute(matches, EloConfig())
    # Strong scores far above expectation → attack up; concedes none → def down
    assert state["att"]["Strong"] > 0
    assert state["def"]["Strong"] < 0
    # Weak concedes a lot → leaky defence (positive def); scores none → att down
    assert state["def"]["Weak"] > 0
    assert state["att"]["Weak"] < 0


def test_opponent_adjusted_form_is_leak_free():
    """perf_vs_elo for a team's FIRST match must be NaN (no history)."""
    from wm.features.elo_rolling import compute
    from wm.features.form import compute_form
    matches = _mk_matches([
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "goals_home": 2, "goals_away": 0},
        {"date": "2020-02-01", "home_team": "A", "away_team": "B",
         "goals_home": 0, "goals_away": 1},
    ])
    elo_df, _ = compute(matches, EloConfig())
    lookup = {r["match_id"]: (r["elo_home_before"], r["elo_away_before"])
              for _, r in elo_df.iterrows()}
    form_df = compute_form(matches, [5], elo_lookup=lookup)
    assert np.isnan(form_df.iloc[0]["home_perf_vs_elo_5"])
    # Second match: A won match 1 vs equal-rated B → positive performance
    assert form_df.iloc[1]["home_perf_vs_elo_5"] > 0
    assert form_df.iloc[1]["home_opp_elo_5"] == pytest.approx(1500.0)


def test_form_state_snapshot_matches_history():
    from wm.features.form import compute_form
    matches = _mk_matches([
        {"date": "2020-01-01", "home_team": "A", "away_team": "B",
         "goals_home": 3, "goals_away": 1},
        {"date": "2020-02-01", "home_team": "B", "away_team": "A",
         "goals_home": 0, "goals_away": 2},
    ])
    _, state = compute_form(matches, [5], return_state=True)
    # A won both matches → current ppg over window 5 is 3.0
    assert state["A"]["ppg_5"] == pytest.approx(3.0)
    assert state["B"]["ppg_5"] == pytest.approx(0.0)
    assert state["A"]["n_matches_5"] == 2.0
