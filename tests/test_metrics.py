"""Tests for evaluation metrics."""
import numpy as np
import pytest
from wm.eval.metrics import brier_multiclass, rps, evaluate


def test_brier_perfect():
    y = np.array([0, 1, 2])
    probs = np.eye(3)[[0, 1, 2]]
    assert brier_multiclass(y, probs) == pytest.approx(0.0)


def test_brier_random():
    y = np.array([0, 1, 2, 0])
    probs = np.array([[1/3, 1/3, 1/3]] * 4)
    score = brier_multiclass(y, probs)
    assert 0 < score < 1


def test_rps_perfect():
    y = np.array([0, 1, 2])
    # Perfect prediction → RPS = 0
    probs = np.eye(3)[[0, 1, 2]]
    assert rps(y, probs) == pytest.approx(0.0)


def test_rps_worst_case():
    # Predicting opposite of truth
    y = np.array([2])
    probs = np.array([[1.0, 0.0, 0.0]])
    score = rps(y, probs)
    assert score > 0


def test_evaluate_returns_all_keys():
    y = np.array([0, 1, 2, 0, 1])
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7],
                      [0.6, 0.2, 0.2], [0.2, 0.5, 0.3]])
    result = evaluate(y, probs)
    assert "log_loss" in result
    assert "brier" in result
    assert "rps" in result


def test_probabilities_sum_to_one():
    from wm.models.ensemble import independent_poisson_grid
    grid = independent_poisson_grid(1.5, 1.2, max_g=10)
    assert abs(grid.sum() - 1.0) < 1e-6
