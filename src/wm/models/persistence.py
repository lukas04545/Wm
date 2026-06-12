"""Save and load model bundles."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb

from wm.models.calibrate import WDLCalibrator
from wm.models.ensemble import MatchPredictor
from wm.models.goals_poisson import DixonColes


def save(
    predictor: MatchPredictor,
    dixon_coles: DixonColes | None,
    model_dir: Path,
    meta: dict | None = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.clf.save_model(str(model_dir / "wdl_clf.lgb"))
    predictor.goals_home.save_model(str(model_dir / "goals_home.lgb"))
    predictor.goals_away.save_model(str(model_dir / "goals_away.lgb"))
    with open(model_dir / "calibrator.pkl", "wb") as f:
        pickle.dump(predictor.calibrator, f)
    if predictor.nn is not None:
        with open(model_dir / "neural_net.pkl", "wb") as f:
            pickle.dump(predictor.nn, f)
    stacker_path = model_dir / "stacker.pkl"
    if predictor.stacker is not None:
        with open(stacker_path, "wb") as f:
            pickle.dump(predictor.stacker, f)
    elif stacker_path.exists():
        stacker_path.unlink()  # stale stacker from a previous train run
    with open(model_dir / "meta.json", "w") as f:
        json.dump(
            {
                "blend_weight": predictor.blend_weight,
                "blend_weights": (
                    predictor.blend_weights.tolist()
                    if predictor.blend_weights is not None else None
                ),
                "max_goals": predictor.max_goals,
                "dc_rho": predictor.dc_rho,
                **(meta or {}),
            },
            f,
            indent=2,
        )
    if dixon_coles is not None:
        with open(model_dir / "dixon_coles.pkl", "wb") as f:
            pickle.dump(dixon_coles, f)


def load(model_dir: Path) -> tuple[MatchPredictor, DixonColes | None]:
    clf = lgb.Booster(model_file=str(model_dir / "wdl_clf.lgb"))
    gh = lgb.Booster(model_file=str(model_dir / "goals_home.lgb"))
    ga = lgb.Booster(model_file=str(model_dir / "goals_away.lgb"))
    with open(model_dir / "calibrator.pkl", "rb") as f:
        calibrator: WDLCalibrator = pickle.load(f)
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)

    nn = None
    nn_path = model_dir / "neural_net.pkl"
    if nn_path.exists():
        with open(nn_path, "rb") as f:
            nn = pickle.load(f)

    stacker = None
    stacker_path = model_dir / "stacker.pkl"
    if stacker_path.exists():
        with open(stacker_path, "rb") as f:
            stacker = pickle.load(f)

    predictor = MatchPredictor(
        clf=clf,
        goals_home_model=gh,
        goals_away_model=ga,
        calibrator=calibrator,
        nn=nn,
        blend_weights=meta.get("blend_weights"),
        blend_weight=meta.get("blend_weight", 0.5),
        max_goals=meta.get("max_goals", 10),
        dc_rho=meta.get("dc_rho", 0.0),
        stacker=stacker,
    )

    dc = None
    dc_path = model_dir / "dixon_coles.pkl"
    if dc_path.exists():
        with open(dc_path, "rb") as f:
            dc = pickle.load(f)

    return predictor, dc
