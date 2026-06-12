"""LightGBM Poisson regressors for home and away expected goals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from wm.features.matrix import CATEGORICAL_FEATURES, TARGET_WDL


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {TARGET_WDL, "goals_home", "goals_away", "match_id", "date",
               "home_team", "away_team", "sample_weight", "outcome", "result",
               "is_wc", "tournament", "city", "country", "neutral",
               "conf_home", "conf_away"}
    return [c for c in df.columns if c not in exclude and not c.startswith("label_")]


def train_goals_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    params: dict | None = None,
) -> lgb.Booster:
    feat_cols = get_feature_cols(train_df)
    feat_cols = [c for c in feat_cols if c in train_df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_cols]

    defaults = {
        "objective": "poisson",
        "metric": "poisson",
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)

    X_train = train_df[feat_cols]
    y_train = train_df[target].clip(0, 15).astype(float)
    w_train = train_df.get("sample_weight", pd.Series(np.ones(len(train_df)), index=train_df.index))

    X_val = val_df[feat_cols]
    y_val = val_df[target].clip(0, 15).astype(float)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols)

    early_stop = defaults.pop("early_stopping_rounds", 50)
    n_est = defaults.pop("n_estimators", 1000)

    callbacks = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(
        defaults,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict | None = None,
) -> tuple[lgb.Booster, lgb.Booster]:
    """Train and return (goals_home_model, goals_away_model)."""
    m_home = train_goals_model(train_df, val_df, "goals_home", params)
    m_away = train_goals_model(train_df, val_df, "goals_away", params)
    return m_home, m_away


def predict_lambdas(
    model_home: lgb.Booster,
    model_away: lgb.Booster,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda_home, lambda_away) arrays of expected goals."""
    # Reindex to the exact training feature set/order; missing columns → NaN
    X = df.reindex(columns=model_home.feature_name())
    return model_home.predict(X), model_away.predict(X)
