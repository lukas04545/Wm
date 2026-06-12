"""Probability calibration for multiclass W/D/L predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling: divide logits by T, re-softmax."""
    logits = np.log(np.clip(probs, 1e-7, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_ = np.exp(scaled)
    return exp_ / exp_.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find optimal temperature T that minimizes log-loss on validation data."""
    from scipy.optimize import minimize_scalar

    def neg_ll(T: float) -> float:
        if T <= 0:
            return 1e9
        cal = temperature_scale(probs, T)
        n = len(labels)
        ll = 0.0
        for i in range(n):
            ll += np.log(cal[i, int(labels[i])] + 1e-10)
        return -ll / n

    res = minimize_scalar(neg_ll, bounds=(0.1, 5.0), method="bounded")
    return float(res.x)


class WDLCalibrator:
    """Temperature scaling calibrator for W/D/L probabilities."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "WDLCalibrator":
        """Fit on validation set. labels: 0=away win, 1=draw, 2=home win."""
        self.temperature = fit_temperature(probs, labels)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return temperature_scale(probs, self.temperature)
