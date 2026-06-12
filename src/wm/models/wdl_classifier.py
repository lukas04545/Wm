"""LightGBM multiclass W/D/L classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

from wm.features.matrix import TARGET_WDL, CATEGORICAL_FEATURES


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {TARGET_WDL, "goals_home", "goals_away", "match_id", "date",
               "home_team", "away_team", "sample_weight", "outcome", "result",
               "is_wc", "tournament", "city", "country", "neutral",
               "conf_home", "conf_away"}
    return [c for c in df.columns if c not in exclude and not c.startswith("label_")]


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    params: dict | None = None,
) -> lgb.Booster:
    feat_cols = get_feature_cols(train_df)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_cols]

    defaults = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)

    X_train = train_df[feat_cols]
    y_train = train_df[TARGET_WDL].astype(int)
    w_train = train_df.get("sample_weight", pd.Series(np.ones(len(train_df)), index=train_df.index))

    X_val = val_df[feat_cols]
    y_val = val_df[TARGET_WDL].astype(int)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols)

    early_stop = defaults.pop("early_stopping_rounds", 50)
    n_est = defaults.pop("n_estimators", 2000)

    callbacks = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(
        defaults,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return model


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Return shape (N, 3) array: [p_away_win, p_draw, p_home_win]."""
    # Reindex to the exact training feature set/order; missing columns → NaN
    X = df.reindex(columns=model.feature_name())
    raw = model.predict(X)
    return raw  # already [p_class0=away_win, p_class1=draw, p_class2=home_win]
