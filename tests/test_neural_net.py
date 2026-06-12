"""Tests for the backpropagation MLP."""
import numpy as np
import pandas as pd
import pickle
import pytest

from wm.models.neural_net import MLPClassifier


def make_data(n=600, d=8, seed=0):
    """Synthetic 3-class problem with a learnable linear structure."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logits = X[:, :3] * 2.0 + rng.normal(scale=0.3, size=(n, 3))
    y = logits.argmax(axis=1)
    cols = [f"f{i}" for i in range(d)]
    return pd.DataFrame(X, columns=cols), y


def test_backprop_gradients_match_numerical():
    """Analytic backprop gradients must equal finite-difference gradients."""
    rng = np.random.default_rng(1)
    n, d = 12, 5
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 3, n)
    y_oh = np.eye(3)[y]
    sw = rng.uniform(0.5, 1.5, n)

    net = MLPClassifier(hidden=(7, 6), dropout=0.0, l2=1e-3, seed=2)
    net._init_params(d, np.random.default_rng(3))

    loss0, w_grads, b_grads = net._loss_and_grads(X, y_oh, sw, dropout_masks=None)
    eps = 1e-6

    for layer in range(len(net.weights_)):
        W = net.weights_[layer]
        # spot-check a handful of entries per layer
        for (i, j) in [(0, 0), (W.shape[0] - 1, W.shape[1] - 1), (W.shape[0] // 2, 0)]:
            W[i, j] += eps
            loss_plus, _, _ = net._loss_and_grads(X, y_oh, sw, dropout_masks=None)
            W[i, j] -= 2 * eps
            loss_minus, _, _ = net._loss_and_grads(X, y_oh, sw, dropout_masks=None)
            W[i, j] += eps
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = w_grads[layer][i, j]
            assert analytic == pytest.approx(numerical, rel=1e-4, abs=1e-7), \
                f"layer {layer} W[{i},{j}]: analytic {analytic} vs numerical {numerical}"

        b = net.biases_[layer]
        b[0] += eps
        loss_plus, _, _ = net._loss_and_grads(X, y_oh, sw, dropout_masks=None)
        b[0] -= 2 * eps
        loss_minus, _, _ = net._loss_and_grads(X, y_oh, sw, dropout_masks=None)
        b[0] += eps
        numerical = (loss_plus - loss_minus) / (2 * eps)
        assert b_grads[layer][0] == pytest.approx(numerical, rel=1e-4, abs=1e-7)


def test_training_learns_signal():
    """Backprop training must beat the uniform-prediction log-loss by a wide margin."""
    X, y = make_data(n=900)
    X_tr, y_tr = X.iloc[:700], y[:700]
    X_va, y_va = X.iloc[700:], y[700:]

    net = MLPClassifier(hidden=(32, 16), max_epochs=60, patience=10, seed=0)
    net.fit(X_tr, y_tr, X_va, y_va)

    uniform_ll = -np.log(1 / 3)  # ≈ 1.0986
    assert net.best_val_loss_ < 0.6 * uniform_ll


def test_predict_proba_valid():
    X, y = make_data(n=300)
    net = MLPClassifier(hidden=(16,), max_epochs=15, seed=0)
    net.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    probs = net.predict_proba(X)
    assert probs.shape == (300, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_handles_nans_and_missing_columns():
    X, y = make_data(n=300)
    X_nan = X.copy()
    X_nan.iloc[::7, 2] = np.nan
    net = MLPClassifier(hidden=(16,), max_epochs=10, seed=0)
    net.fit(X_nan.iloc[:250], y[:250], X_nan.iloc[250:], y[250:])
    # prediction with an extra column and a missing column still works
    X_messy = X.drop(columns=["f3"]).assign(extra=1.0)
    probs = net.predict_proba(X_messy)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_pickle_roundtrip():
    X, y = make_data(n=300)
    net = MLPClassifier(hidden=(16,), max_epochs=10, seed=0)
    net.fit(X.iloc[:250], y[:250], X.iloc[250:], y[250:])
    restored = pickle.loads(pickle.dumps(net))
    assert np.allclose(restored.predict_proba(X), net.predict_proba(X))


def test_bagged_mlp_averages_and_improves():
    """Bagged ensemble must produce valid probs and not be worse than a single net."""
    from wm.models.neural_net import BaggedMLPClassifier, MLPClassifier
    X, y = make_data(n=900)
    X_tr, y_tr = X.iloc[:700], y[:700]
    X_va, y_va = X.iloc[700:], y[700:]

    single = MLPClassifier(hidden=(32, 16), max_epochs=60, patience=10, seed=1)
    single.fit(X_tr, y_tr, X_va, y_va)

    bag = BaggedMLPClassifier(n_models=3, base_seed=1, hidden=(32, 16),
                              max_epochs=60, patience=10)
    bag.fit(X_tr, y_tr, X_va, y_va)

    probs = bag.predict_proba(X_va)
    assert probs.shape == (200, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert len(bag.models_) == 3

    yv_oh = np.eye(3)[y_va]
    bag_ll = -np.mean(np.log((probs * yv_oh).sum(1) + 1e-12))
    single_ll = -np.mean(np.log((single.predict_proba(X_va) * yv_oh).sum(1) + 1e-12))
    assert bag_ll <= single_ll + 0.02  # bagging shouldn't hurt


def test_blend_weights_on_simplex():
    from wm.models.ensemble import fit_blend_weights
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.1)
    good[np.arange(n), y] = 0.8
    bad = np.full((n, 3), 1 / 3)
    w = fit_blend_weights([good, bad], y)
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert w[0] > 0.8  # the informative branch must dominate
