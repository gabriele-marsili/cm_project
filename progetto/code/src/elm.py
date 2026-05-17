"""ELM with L1-regularised output layer."""

import numpy as np

from .irls import irls
from .deflected_subgradient import deflected_subgradient


def _sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z):
    return np.maximum(0.0, z)


def _tanh(z):
    return np.tanh(z)


_ACTIVATIONS = {'sigmoid': _sigmoid, 'relu': _relu, 'tanh': _tanh}


class ELM:
    def __init__(self, d, p, activation='sigmoid', lam=0.1, random_state=42):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}")
        self.d = d
        self.p = p
        self.activation = activation
        self.lam = lam
        self.sigma = _ACTIVATIONS[activation]

        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(p, d)

        self.w = None
        self._fit_result = None

    def transform(self, X_raw):
        return self.sigma(X_raw @ self.W1.T)

    def fit(self, X_raw, y, solver='irls', **kwargs):
        Xh = self.transform(X_raw)
        if solver == 'irls':
            res = irls(Xh, y, self.lam, **kwargs)
        elif solver == 'dsm':
            res = deflected_subgradient(Xh, y, self.lam, **kwargs)
        else:
            raise ValueError(f"Unknown solver {solver!r}")
        self.w = res['w']
        self._fit_result = res
        return self

    def predict(self, X_raw):
        if self.w is None:
            raise RuntimeError("Call fit() first.")
        return self.transform(X_raw) @ self.w

    @property
    def sparsity(self):
        if self.w is None:
            return None
        return float(np.mean(np.abs(self.w) < 1e-8))

    @property
    def n_active(self):
        if self.w is None:
            return None
        return int(np.sum(np.abs(self.w) >= 1e-8))

    def __repr__(self):
        return (f"ELM(d={self.d}, p={self.p}, "
                f"activation={self.activation!r}, lam={self.lam})")
