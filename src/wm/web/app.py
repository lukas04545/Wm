"""FastAPI web application for FIFA 2026 predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wm import config as cfg_mod
from wm import config as wc_cfg_mod

ROOT = Path(__file__).parent.parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"
_state: dict = {}


# NOTE: request models must live at module level — with
# `from __future__ import annotations` FastAPI cannot resolve classes
# defined inside create_app()'s local scope (body would silently become
# a query parameter).
class PredictRequest(BaseModel):
    home: str
    away: str
    venue: Optional[str] = "MetLife Stadium"
    neutral: Optional[bool] = True


class SimulateRequest(BaseModel):
    runs: int = 10000
    seed: int = 42


def _load_state(cfg: cfg_mod.Config) -> None:
    """Attempt to load models and cached simulation results."""
    models_dir = cfg.path("models")
    sim_path = cfg.path("reports") / "simulation" / "results.json"
    interim_dir = cfg.path("data_interim")

    # Load WC config
    wc = wc_cfg_mod.load_wc2026()
    _state["groups"] = {g: d["teams"] for g, d in wc["groups"].items()}
    _state["confederations"] = wc.get("confederations", {})
    _state["venues"] = wc.get("venues", {})
    _state["all_teams"] = [t for teams in _state["groups"].values() for t in teams]

    # Load simulation results
    if sim_path.exists():
        with open(sim_path) as f:
            _state["sim_results"] = json.load(f)

    # Load models
    try:
        from wm.models import persistence
        from wm.features.state import compute_team_state
        from wm.data import build as build_mod

        predictor, dc = persistence.load(models_dir)
        _state["predictor"] = predictor
        _state["dc"] = dc

        try:
            matches = build_mod.load(interim_dir / "matches.parquet")
            team_state = compute_team_state(matches, cfg)
            _state["team_state"] = team_state
            _state["elo_ratings"] = {t: s["elo"] for t, s in team_state.items()}
        except FileNotFoundError:
            _state["team_state"] = None
            _state["elo_ratings"] = {}

        # Optional enrichments
        from wm.ingest import player_ratings as pr_mod, fifa_rankings as rank_mod
        raw_dir = cfg.path("data_raw")
        _state["squad_df"] = pr_mod.get_squad_features(raw_dir)
        _state["rankings_df"] = rank_mod.load(raw_dir)
        _state["models_loaded"] = True
    except Exception as e:
        _state["models_loaded"] = False
        _state["models_error"] = str(e)


def create_app(config_path: Path | None = None) -> FastAPI:
    cfg = cfg_mod.load(config_path)

    app = FastAPI(title="FIFA World Cup 2026 Predictor", version="1.0.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    _load_state(cfg)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(STATIC_DIR / "index.html") as f:
            return f.read()

    @app.get("/api/status")
    async def status():
        sim_path = cfg.path("reports") / "simulation" / "results.json"
        raw_path = cfg.path("data_raw") / "results.csv"
        return {
            "data_ingested": raw_path.exists(),
            "models_loaded": _state.get("models_loaded", False),
            "models_error": _state.get("models_error"),
            "simulation_available": "sim_results" in _state or sim_path.exists(),
            "n_teams": len(_state.get("all_teams", [])),
        }

    @app.get("/api/groups")
    async def get_groups():
        return {"groups": _state.get("groups", {})}

    @app.get("/api/teams")
    async def get_teams():
        teams = sorted(_state.get("all_teams", []))
        elos = _state.get("elo_ratings", {})
        return {
            "teams": [
                {"name": t, "elo": round(elos.get(t, 1500.0))}
                for t in teams
            ]
        }

    @app.get("/api/simulation")
    async def get_simulation():
        # Try memory first, then disk
        if "sim_results" in _state:
            return _state["sim_results"]
        sim_path = cfg.path("reports") / "simulation" / "results.json"
        if sim_path.exists():
            with open(sim_path) as f:
                data = json.load(f)
            _state["sim_results"] = data
            return data
        raise HTTPException(status_code=404, detail="Simulation results not available. Run `wm simulate` first.")

    @app.get("/api/elo")
    async def get_elo():
        elos = _state.get("elo_ratings", {})
        sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
        return {"rankings": [{"team": t, "elo": round(v)} for t, v in sorted_elos[:48]]}

    @app.post("/api/predict")
    async def predict_match(req: PredictRequest):
        if not _state.get("models_loaded"):
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Run `wm train` first."
            )
        try:
            from wm.sim.match_sampler import build_fixture_row
            predictor = _state["predictor"]
            elos = _state.get("elo_ratings", {})

            row = build_fixture_row(
                home=req.home,
                away=req.away,
                elo_home=elos.get(req.home, 1500.0),
                elo_away=elos.get(req.away, 1500.0),
                venue=req.venue or "MetLife Stadium",
                date=pd.Timestamp("2026-07-01"),
                is_neutral=req.neutral if req.neutral is not None else True,
                is_host_home=False,
                squad_df=_state.get("squad_df"),
                rankings_df=_state.get("rankings_df"),
                cfg=cfg,
                team_state=_state.get("team_state"),
            )

            result = predictor.predict(row)
            ph = float(result["p_home_win"][0])
            pd_ = float(result["p_draw"][0])
            pa = float(result["p_away_win"][0])
            lh = float(result["lambda_home"][0])
            la = float(result["lambda_away"][0])

            # Top scorelines
            grid = result["score_grid"][0]
            max_g = grid.shape[0]
            scorelines = []
            for i in range(min(max_g, 6)):
                for j in range(min(max_g, 6)):
                    scorelines.append({"home": i, "away": j, "prob": float(grid[i, j])})
            scorelines.sort(key=lambda x: x["prob"], reverse=True)

            return {
                "home": req.home,
                "away": req.away,
                "p_home_win": round(ph * 100, 1),
                "p_draw": round(pd_ * 100, 1),
                "p_away_win": round(pa * 100, 1),
                "lambda_home": round(lh, 2),
                "lambda_away": round(la, 2),
                "elo_home": round(elos.get(req.home, 1500.0)),
                "elo_away": round(elos.get(req.away, 1500.0)),
                "top_scorelines": scorelines[:9],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/simulate")
    async def run_simulation(req: SimulateRequest, background_tasks: BackgroundTasks):
        if not _state.get("models_loaded"):
            raise HTTPException(status_code=503, detail="Models not loaded. Run `wm train` first.")
        if _state.get("simulation_running"):
            return {"status": "running", "message": "Simulation already in progress"}

        def _run():
            try:
                _state["simulation_running"] = True
                from wm.sim.monte_carlo import TournamentSimulator
                from wm.sim.fixtures import load_group_fixtures, load_played_knockouts
                predictor = _state["predictor"]
                elos = _state.get("elo_ratings", {})
                groups = _state["groups"]
                raw_dir = cfg.path("data_raw")

                simulator = TournamentSimulator(
                    groups=groups,
                    predictor=predictor,
                    elo_ratings=elos,
                    cfg=cfg,
                    squad_df=_state.get("squad_df"),
                    rankings_df=_state.get("rankings_df"),
                    team_state=_state.get("team_state"),
                    fixtures=load_group_fixtures(raw_dir, groups) or None,
                    played_knockouts=load_played_knockouts(raw_dir) or None,
                )
                results = simulator.run(req.runs, seed=req.seed)
                _state["sim_results"] = results

                sim_path = cfg.path("reports") / "simulation" / "results.json"
                sim_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sim_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
            finally:
                _state["simulation_running"] = False

        background_tasks.add_task(_run)
        return {"status": "started", "message": f"Running {req.runs:,} simulations in background"}

    @app.get("/api/simulate/status")
    async def simulation_status():
        return {
            "running": _state.get("simulation_running", False),
            "available": "sim_results" in _state,
        }

    return app
