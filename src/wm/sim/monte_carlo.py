"""
Monte Carlo tournament simulation for FIFA World Cup 2026.

Runs N independent full-tournament simulations and aggregates:
  - P(champion), P(final), P(semifinal), P(quarterfinal), P(round of 16), P(round of 32)
  - Expected group points and goals
  - Most likely final matchups
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from wm import config as cfg_mod
from wm.models.ensemble import MatchPredictor, independent_poisson_grid
from wm.sim import tournament as trn
from wm.sim.match_sampler import build_fixture_row


class TournamentSimulator:
    """
    Runs the full 2026 WC Monte Carlo simulation.

    predict_fn(home, away, neutral) → (score_grid, (lambda_home, lambda_away))
    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        predictor: MatchPredictor,
        elo_ratings: dict[str, float],
        cfg: cfg_mod.Config,
        squad_df: pd.DataFrame | None = None,
        rankings_df: pd.DataFrame | None = None,
    ):
        self.groups = groups
        self.predictor = predictor
        self.elo_ratings = elo_ratings.copy()
        self.cfg = cfg
        self.squad_df = squad_df
        self.rankings_df = rankings_df
        self._lambda_cache: dict[tuple, tuple[float, float]] = {}

    def _base_lambdas(
        self,
        home: str,
        away: str,
        neutral: bool,
        elo_home: float | None = None,
        elo_away: float | None = None,
    ) -> tuple[float, float]:
        """Model-predicted expected goals, cached per matchup (the expensive part)."""
        cache_key = (home, away, neutral)
        if cache_key not in self._lambda_cache:
            from wm.models.goals_gbm import predict_lambdas

            row = self._make_row(home, away, neutral, elo_home, elo_away)
            lh_arr, la_arr = predict_lambdas(
                self.predictor.goals_home, self.predictor.goals_away, row
            )
            self._lambda_cache[cache_key] = (
                float(np.clip(lh_arr[0], 0.05, 8.0)),
                float(np.clip(la_arr[0], 0.05, 8.0)),
            )
        return self._lambda_cache[cache_key]

    def _predict_fn(
        self,
        home: str,
        away: str,
        neutral: bool,
        elo_home: float | None = None,
        elo_away: float | None = None,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        lh, la = self._base_lambdas(home, away, neutral, elo_home, elo_away)

        if noise > 0 and rng is not None:
            lh *= np.exp(rng.normal(0, noise))
            la *= np.exp(rng.normal(0, noise))

        grid = independent_poisson_grid(lh, la, self.cfg.simulation.max_goals_grid)
        return grid, (lh, la)

    def _make_row(
        self,
        home: str,
        away: str,
        neutral: bool,
        elo_home: float | None = None,
        elo_away: float | None = None,
    ) -> pd.DataFrame:
        eh = elo_home if elo_home is not None else self.elo_ratings.get(home, 1500.0)
        ea = elo_away if elo_away is not None else self.elo_ratings.get(away, 1500.0)
        return build_fixture_row(
            home=home,
            away=away,
            elo_home=eh,
            elo_away=ea,
            venue="MetLife Stadium",  # default; ideally venue-specific
            date=pd.Timestamp("2026-07-01"),
            is_neutral=neutral,
            is_host_home=False,
            squad_df=self.squad_df,
            rankings_df=self.rankings_df,
            cfg=self.cfg,
        )

    def _predict_fn_wrapper(self, noise: float, rng: np.random.Generator):
        def fn(home: str, away: str, neutral: bool = True):
            return self._predict_fn(home, away, neutral, noise=noise, rng=rng)
        return fn

    def run(self, n_runs: int, seed: int = 42) -> dict[str, dict[str, float]]:
        """
        Run Monte Carlo simulation.
        Returns dict with stage probabilities per team.
        """
        sim_cfg = self.cfg.simulation
        all_teams = [t for teams in self.groups.values() for t in teams]

        # Counters
        champion_count: dict[str, int] = defaultdict(int)
        stage_counts: dict[str, dict[str, int]] = {
            "champion": defaultdict(int),
            "final": defaultdict(int),
            "semifinal": defaultdict(int),
            "quarterfinal": defaultdict(int),
            "r16": defaultdict(int),
            "r32": defaultdict(int),
            "group_qualified": defaultdict(int),
        }
        group_points: dict[str, list[float]] = defaultdict(list)
        champion_pairs: dict[tuple, int] = defaultdict(int)

        rng_master = np.random.default_rng(seed)

        for run_idx in tqdm(range(n_runs), desc="Simulating", unit="run"):
            run_seed = int(rng_master.integers(0, 2**31))
            rng = np.random.default_rng(run_seed)
            noise = sim_cfg.squad_noise_std

            predict_fn = self._predict_fn_wrapper(noise, rng)

            # Group stage
            group_results = trn.simulate_group_stage(self.groups, predict_fn, rng, noise=noise)
            third_ranking = trn.get_third_place_ranking(group_results)

            # Track group points
            for grp, gr in group_results.items():
                for rec in gr.teams:
                    group_points[rec.team].append(rec.pts)

            # Qualified teams
            qualifiers: set[str] = set()
            for grp, gr in group_results.items():
                standings = gr.standings()
                stage_counts["group_qualified"][standings[0].team] += 1
                stage_counts["group_qualified"][standings[1].team] += 1
                qualifiers.add(standings[0].team)
                qualifiers.add(standings[1].team)
            for rec in third_ranking[:8]:
                stage_counts["group_qualified"][rec.team] += 1
                qualifiers.add(rec.team)

            # Build R32 bracket
            bracket = trn.build_r32_bracket(group_results, third_ranking)

            # Round of 32
            for home, away in bracket:
                stage_counts["r32"][home] += 1
                stage_counts["r32"][away] += 1

            r32_winners = []
            for home, away in bracket:
                winner = trn.simulate_knockout_match(
                    home, away, predict_fn, rng,
                    max_goals=sim_cfg.max_goals_grid,
                    et_factor=sim_cfg.et_lambda_factor,
                    base_pen=sim_cfg.base_penalty_conversion,
                )
                r32_winners.append(winner)

            # Round of 16
            r16_pairs = list(zip(r32_winners[::2], r32_winners[1::2]))
            for h, a in r16_pairs:
                stage_counts["r16"][h] += 1
                stage_counts["r16"][a] += 1

            r16_winners = [
                trn.simulate_knockout_match(h, a, predict_fn, rng,
                                             max_goals=sim_cfg.max_goals_grid,
                                             et_factor=sim_cfg.et_lambda_factor,
                                             base_pen=sim_cfg.base_penalty_conversion)
                for h, a in r16_pairs
            ]

            # Quarterfinals
            qf_pairs = list(zip(r16_winners[::2], r16_winners[1::2]))
            for h, a in qf_pairs:
                stage_counts["quarterfinal"][h] += 1
                stage_counts["quarterfinal"][a] += 1

            qf_winners = [
                trn.simulate_knockout_match(h, a, predict_fn, rng,
                                             max_goals=sim_cfg.max_goals_grid,
                                             et_factor=sim_cfg.et_lambda_factor,
                                             base_pen=sim_cfg.base_penalty_conversion)
                for h, a in qf_pairs
            ]

            # Semifinals
            sf_pairs = list(zip(qf_winners[::2], qf_winners[1::2]))
            for h, a in sf_pairs:
                stage_counts["semifinal"][h] += 1
                stage_counts["semifinal"][a] += 1

            sf_winners = [
                trn.simulate_knockout_match(h, a, predict_fn, rng,
                                             max_goals=sim_cfg.max_goals_grid,
                                             et_factor=sim_cfg.et_lambda_factor,
                                             base_pen=sim_cfg.base_penalty_conversion)
                for h, a in sf_pairs
            ]

            # Final
            if len(sf_winners) >= 2:
                finalist_1, finalist_2 = sf_winners[0], sf_winners[1]
                stage_counts["final"][finalist_1] += 1
                stage_counts["final"][finalist_2] += 1
                pair = tuple(sorted([finalist_1, finalist_2]))
                champion_pairs[pair] += 1

                champion = trn.simulate_knockout_match(
                    finalist_1, finalist_2, predict_fn, rng,
                    max_goals=sim_cfg.max_goals_grid,
                    et_factor=sim_cfg.et_lambda_factor,
                    base_pen=sim_cfg.base_penalty_conversion,
                )
                stage_counts["champion"][champion] += 1

        # Aggregate
        results: dict[str, dict] = {}
        for team in all_teams:
            results[team] = {
                stage: stage_counts[stage].get(team, 0) / n_runs
                for stage in stage_counts
            }
            results[team]["avg_group_points"] = (
                np.mean(group_points[team]) if group_points[team] else 0.0
            )

        # Top final pairings
        top_pairs = sorted(champion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        results["_meta"] = {
            "n_runs": n_runs,
            "seed": seed,
            "top_final_pairings": [(list(pair), count / n_runs) for pair, count in top_pairs],
        }

        return results
