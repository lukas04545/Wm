"""Evaluation metrics: log-loss, Brier, RPS, calibration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


def brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score (lower is better)."""
    n, k = probs.shape
    y_oh = np.zeros_like(probs)
    y_oh[np.arange(n), y_true.astype(int)] = 1
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def rps(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Ranked Probability Score for ordinal classes [away_win, draw, home_win].
    Lower is better.
    """
    n = len(y_true)
    k = probs.shape[1]
    cum_probs = np.cumsum(probs, axis=1)
    cum_actual = np.zeros_like(cum_probs)
    for i in range(n):
        cum_actual[i, int(y_true[i]):] = 1.0
    return float(np.mean(np.sum((cum_probs - cum_actual) ** 2, axis=1)) / (k - 1))


def evaluate(y_true: np.ndarray, probs: np.ndarray, name: str = "") -> dict[str, float]:
    ll = log_loss(y_true, probs, labels=[0, 1, 2])
    br = brier_multiclass(y_true, probs)
    rp = rps(y_true, probs)
    metrics = {"log_loss": ll, "brier": br, "rps": rp}
    if name:
        metrics = {f"{name}_{k}": v for k, v in metrics.items()}
    return metrics


def calibration_plot(
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: list[str],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Save a calibration reliability plot for each class."""
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4))
    if len(labels) == 1:
        axes = [axes]

    for ax, label, i in zip(axes, labels, range(len(labels))):
        p = probs[:, i]
        y = (y_true == i).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        means_pred, means_act = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() >= 5:
                means_pred.append(p[mask].mean())
                means_act.append(y[mask].mean())
        ax.plot(means_pred, means_act, "o-", label=label)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(label)
        ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
