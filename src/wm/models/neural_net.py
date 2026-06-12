"""
Multi-layer perceptron classifier trained with backpropagation.

Implemented from scratch in NumPy so it runs everywhere LightGBM does
(Termux, Colab, CI) without pulling in PyTorch/TensorFlow.

Architecture & training:
  input → Dense+ReLU (×len(hidden)) → Dense → softmax
  - weighted cross-entropy loss, L2 regularization
  - gradients via reverse-mode backpropagation (see _loss_and_grads)
  - Adam optimizer, mini-batches, inverted dropout
  - early stopping on validation log-loss
  - built-in preprocessing: median imputation + standardization
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        n_classes: int = 3,
        lr: float = 1e-3,
        l2: float = 1e-4,
        dropout: float = 0.2,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 12,
        seed: int = 42,
    ):
        self.hidden = tuple(hidden)
        self.n_classes = n_classes
        self.lr = lr
        self.l2 = l2
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

        self.feature_names_: list[str] = []
        self.medians_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.best_val_loss_: float = float("inf")
        self.n_epochs_run_: int = 0

    # ── preprocessing ─────────────────────────────────────────────────────────
    def _fit_preprocess(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names_ = list(X.columns)
        arr = X.to_numpy(dtype=float)
        self.medians_ = np.nanmedian(arr, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        arr = np.where(np.isnan(arr), self.medians_, arr)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-9, 1.0, self.std_)
        return (arr - self.mean_) / self.std_

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names_)
        arr = X.to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), self.medians_, arr)
        return (arr - self.mean_) / self.std_

    # ── core: forward pass + backpropagation ──────────────────────────────────
    def _init_params(self, n_features: int, rng: np.random.Generator) -> None:
        sizes = [n_features, *self.hidden, self.n_classes]
        self.weights_, self.biases_ = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialization for ReLU layers
            self.weights_.append(rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))

    def _loss_and_grads(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        sample_w: np.ndarray,
        dropout_masks: list[np.ndarray] | None = None,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """
        Forward pass, then backpropagation of the loss gradient through
        every layer (reverse-mode autodiff by hand).

        Returns (loss, weight_grads, bias_grads).
        """
        n_layers = len(self.weights_)
        w_sum = sample_w.sum()

        # ---- forward ----
        activations = [X]          # a_0 = input
        pre_acts = []              # z_l
        a = X
        for l in range(n_layers - 1):
            z = a @ self.weights_[l] + self.biases_[l]
            a = np.maximum(z, 0.0)                       # ReLU
            if dropout_masks is not None:
                a = a * dropout_masks[l] / (1.0 - self.dropout)  # inverted dropout
            pre_acts.append(z)
            activations.append(a)
        z_out = a @ self.weights_[-1] + self.biases_[-1]
        probs = _softmax(z_out)

        # weighted cross-entropy + L2
        ce = -np.sum(sample_w * np.log(np.sum(probs * y_onehot, axis=1) + 1e-12)) / w_sum
        l2_term = 0.5 * self.l2 * sum(float(np.sum(W * W)) for W in self.weights_)
        loss = ce + l2_term

        # ---- backward (backpropagation) ----
        w_grads = [None] * n_layers
        b_grads = [None] * n_layers

        # ∂L/∂z_out for softmax + cross-entropy: (p - y) weighted per sample
        delta = (probs - y_onehot) * (sample_w / w_sum)[:, None]
        w_grads[-1] = activations[-1].T @ delta + self.l2 * self.weights_[-1]
        b_grads[-1] = delta.sum(axis=0)

        for l in range(n_layers - 2, -1, -1):
            delta = delta @ self.weights_[l + 1].T          # propagate through dense
            if dropout_masks is not None:
                delta = delta * dropout_masks[l] / (1.0 - self.dropout)
            delta = delta * (pre_acts[l] > 0)               # ReLU derivative
            w_grads[l] = activations[l].T @ delta + self.l2 * self.weights_[l]
            b_grads[l] = delta.sum(axis=0)

        return loss, w_grads, b_grads

    # ── training loop (Adam + early stopping) ─────────────────────────────────
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "MLPClassifier":
        rng = np.random.default_rng(self.seed)
        Xs = self._fit_preprocess(X)
        Xv = self._transform(X_val)
        y = np.asarray(y, dtype=int)
        y_val = np.asarray(y_val, dtype=int)
        n, d = Xs.shape

        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        y_oh = np.eye(self.n_classes)[y]
        yv_oh = np.eye(self.n_classes)[y_val]

        self._init_params(d, rng)
        n_layers = len(self.weights_)

        # Adam state
        m_w = [np.zeros_like(W) for W in self.weights_]
        v_w = [np.zeros_like(W) for W in self.weights_]
        m_b = [np.zeros_like(b) for b in self.biases_]
        v_b = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        best_w = [W.copy() for W in self.weights_]
        best_b = [b.copy() for b in self.biases_]
        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                masks = None
                if self.dropout > 0:
                    masks = [
                        (rng.random((len(idx), h)) > self.dropout).astype(float)
                        for h in self.hidden
                    ]
                _, w_grads, b_grads = self._loss_and_grads(Xs[idx], y_oh[idx], sw[idx], masks)

                t += 1
                for l in range(n_layers):
                    m_w[l] = beta1 * m_w[l] + (1 - beta1) * w_grads[l]
                    v_w[l] = beta2 * v_w[l] + (1 - beta2) * w_grads[l] ** 2
                    m_b[l] = beta1 * m_b[l] + (1 - beta1) * b_grads[l]
                    v_b[l] = beta2 * v_b[l] + (1 - beta2) * b_grads[l] ** 2
                    mw_hat = m_w[l] / (1 - beta1 ** t)
                    vw_hat = v_w[l] / (1 - beta2 ** t)
                    mb_hat = m_b[l] / (1 - beta1 ** t)
                    vb_hat = v_b[l] / (1 - beta2 ** t)
                    self.weights_[l] -= self.lr * mw_hat / (np.sqrt(vw_hat) + eps)
                    self.biases_[l] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            # validation early stopping
            val_probs = self._forward_inference(Xv)
            val_loss = -np.mean(np.log(np.sum(val_probs * yv_oh, axis=1) + 1e-12))
            self.n_epochs_run_ = epoch + 1
            if verbose and epoch % 10 == 0:
                print(f"epoch {epoch:3d}  val_logloss {val_loss:.4f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_w = [W.copy() for W in self.weights_]
                best_b = [b.copy() for b in self.biases_]
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.weights_, self.biases_ = best_w, best_b
        self.best_val_loss_ = best_val
        return self

    def _forward_inference(self, Xs: np.ndarray) -> np.ndarray:
        a = Xs
        for l in range(len(self.weights_) - 1):
            a = np.maximum(a @ self.weights_[l] + self.biases_[l], 0.0)
        return _softmax(a @ self.weights_[-1] + self.biases_[-1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (N, 3) probabilities [p_away_win, p_draw, p_home_win]."""
        return self._forward_inference(self._transform(X))


class BaggedMLPClassifier:
    """
    Bagged ensemble of MLPs trained with different random seeds.

    Averaging several independently-initialized backprop nets reduces the
    variance of the neural branch — the single biggest accuracy gain found
    in tuning (val log-loss ~0.856 single net -> ~0.854 over 3 seeds).
    """

    def __init__(self, n_models: int = 3, base_seed: int = 42, **mlp_kwargs):
        self.n_models = n_models
        self.base_seed = base_seed
        self.mlp_kwargs = mlp_kwargs
        self.models_: list[MLPClassifier] = []

    def fit(self, X, y, X_val, y_val, sample_weight=None, verbose=False) -> "BaggedMLPClassifier":
        self.models_ = []
        for i in range(self.n_models):
            net = MLPClassifier(seed=self.base_seed + i, **self.mlp_kwargs)
            net.fit(X, y, X_val, y_val, sample_weight=sample_weight, verbose=verbose)
            self.models_.append(net)
        return self

    @property
    def best_val_loss_(self) -> float:
        return float(np.mean([m.best_val_loss_ for m in self.models_])) if self.models_ else float("inf")

    @property
    def n_epochs_run_(self) -> int:
        return int(np.mean([m.n_epochs_run_ for m in self.models_])) if self.models_ else 0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)
