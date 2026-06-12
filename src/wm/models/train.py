"""Reusable ensemble training — shared by the `train` CLI and rolling backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wm import config as cfg_mod
from wm.models import wdl_classifier, goals_gbm, calibrate, ensemble
from wm.models.neural_net import BaggedMLPClassifier
from wm.models.wdl_classifier import get_feature_cols
from wm.features.matrix import TARGET_WDL


def _nn_kwargs(cfg: cfg_mod.Config) -> tuple[int, dict]:
    nn_cfg = dict(cfg.model.nn) if cfg.model.nn else {}
    n_models = nn_cfg.pop("n_models", 3)
    nn_cfg.setdefault("hidden", (192, 96, 48))
    nn_cfg["hidden"] = tuple(nn_cfg["hidden"])
    nn_cfg.setdefault("lr", nn_cfg.pop("learning_rate", 8e-4))
    return n_models, nn_cfg


def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: cfg_mod.Config,
    progress=None,
) -> tuple[ensemble.MatchPredictor, dict]:
    """
    Train the full three-branch ensemble (GBM classifier + bagged backprop NN
    + Poisson goal regressors), fit convex blend weights, and calibrate.

    Returns (predictor, info) where info holds blend weights and branch sizes.
    `progress` is an optional callable(str) for status messages.
    """
    def log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    log("Training W/D/L classifier")
    clf = wdl_classifier.train(train_df, val_df, params=cfg.model.wdl)

    log("Training goal regressors")
    m_home, m_away = goals_gbm.train(train_df, val_df, params=cfg.model.goals)

    log("Training bagged neural net (backprop)")
    feat_cols = get_feature_cols(train_df)
    y_train = train_df[TARGET_WDL].astype(int).values
    y_val = val_df[TARGET_WDL].astype(int).values
    n_models, nn_cfg = _nn_kwargs(cfg)
    nn = BaggedMLPClassifier(n_models=n_models, base_seed=cfg.simulation.seed, **nn_cfg)
    nn.fit(train_df[feat_cols], y_train, val_df[feat_cols], y_val,
           sample_weight=train_df.get("sample_weight"))

    log("Fitting Dixon-Coles rho on validation scorelines")
    from wm.models.goals_gbm import predict_lambdas
    lh_val, la_val = predict_lambdas(m_home, m_away, val_df)
    lh_val = np.clip(lh_val, 0.05, 8.0)
    la_val = np.clip(la_val, 0.05, 8.0)
    dc_rho = ensemble.fit_dc_rho(
        lh_val, la_val,
        val_df["goals_home"].to_numpy(), val_df["goals_away"].to_numpy(),
    )

    log("Fitting blend (convex weights vs stacked meta-learner) + calibration")
    probe = ensemble.MatchPredictor(
        clf=clf, goals_home_model=m_home, goals_away_model=m_away,
        calibrator=calibrate.WDLCalibrator(), nn=nn,
        max_goals=cfg.simulation.max_goals_grid, dc_rho=dc_rho,
    )
    branches, _, _, _ = probe.branch_probs(val_df)

    def _ll(p: np.ndarray) -> float:
        idx = np.arange(len(y_val))
        return -float(np.mean(np.log(p[idx, y_val] + 1e-12)))

    # Candidate A: convex blend
    blend_w = ensemble.fit_blend_weights(branches, y_val)
    convex_p = sum(w * P for w, P in zip(blend_w, branches))
    convex_p = convex_p / convex_p.sum(axis=1, keepdims=True)

    # Candidate B: stacked meta-learner (log-prob features + match context)
    stacker = ensemble.StackedBlender().fit(branches, val_df, y_val)
    stacked_p = stacker.predict_proba(branches, val_df)

    use_stacker = _ll(stacked_p) < _ll(convex_p)
    blended_val = stacked_p if use_stacker else convex_p

    # Calibration: vector scaling vs temperature, keep the better on val
    temp_cal = calibrate.WDLCalibrator().fit(blended_val, y_val)
    vec_cal = calibrate.VectorScalingCalibrator().fit(blended_val, y_val)
    calibrator = (
        vec_cal if _ll(vec_cal.predict(blended_val)) < _ll(temp_cal.predict(blended_val))
        else temp_cal
    )

    predictor = ensemble.MatchPredictor(
        clf=clf, goals_home_model=m_home, goals_away_model=m_away,
        calibrator=calibrator, nn=nn, blend_weights=blend_w,
        blend_weight=cfg.model.blend_weight, max_goals=cfg.simulation.max_goals_grid,
        dc_rho=dc_rho, stacker=stacker if use_stacker else None,
    )
    info = {
        "blend_weights": blend_w.tolist(),
        "use_stacker": use_stacker,
        "calibrator": type(calibrator).__name__,
        "nn_val_loss": nn.best_val_loss_,
        "clf_best_iter": clf.best_iteration,
        "n_nn_models": n_models,
        "dc_rho": dc_rho,
    }
    return predictor, info
