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
    with open(model_dir / "meta.json", "w") as f:
        json.dump(
            {
                "blend_weight": predictor.blend_weight,
                "max_goals": predictor.max_goals,
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

    predictor = MatchPredictor(
        clf=clf,
        goals_home_model=gh,
        goals_away_model=ga,
        calibrator=calibrator,
        blend_weight=meta.get("blend_weight", 0.5),
        max_goals=meta.get("max_goals", 10),
    )

    dc = None
    dc_path = model_dir / "dixon_coles.pkl"
    if dc_path.exists():
        with open(dc_path, "rb") as f:
            dc = pickle.load(f)

    return predictor, dc
