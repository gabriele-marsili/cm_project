"""
Extreme Learning Machine wrapper around the two LASSO solvers.

The ELM trick is to fix the input-to-hidden weights ``W_1`` at random and
train only the output weights ``w``. With a non-linear activation ``sigma``,
the hidden representation ``X_hidden = sigma(X_raw W_1^T)`` is constant
during training, so fitting reduces to the convex LASSO

    min_w  (1/2) ||X_hidden w - y||^2 + lam ||w||_1.

Both IRLS (Algorithm A1) and SGPTL (A2) plug into the same fit() entry
point — pick one with ``solver='irls'`` or ``solver='dsm'``.
"""

import numpy as np

from .irls import irls
from .deflected_subgradient import deflected_subgradient


def _sigmoid(z):
    # The clip prevents exp() overflow on rare large pre-activations; the
    # 500 cutoff is well outside the region where sigmoid(z) differs from
    # 0 or 1 by more than machine epsilon.
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z):
    return np.maximum(0.0, z)


def _tanh(z):
    return np.tanh(z)


_ACTIVATIONS = {
    'sigmoid': _sigmoid,
    'relu':    _relu,
    'tanh':    _tanh,
}


class ELM:
    """ELM with an L1-regularised output layer.

    The hidden weights ``W_1`` are sampled once in ``__init__`` and never
    touched again, which is the whole point of the ELM construction.
    """

    def __init__(self, d, p, activation='sigmoid', lam=0.1, random_state=42):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Pick one of {list(_ACTIVATIONS)}."
            )
        self.d = d
        self.p = p
        self.activation = activation
        self.lam = lam
        self.sigma = _ACTIVATIONS[activation]

        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(p, d)

        # Filled in by fit().
        self.w = None
        self._fit_result = None

    def transform(self, X_raw):
        """Apply the fixed random projection: ``X_hidden = sigma(X_raw W_1^T)``."""
        return self.sigma(X_raw @ self.W1.T)

    def fit(self, X_raw, y, solver='irls', **solver_kwargs):
        """Solve the LASSO on the hidden activations.

        ``solver_kwargs`` is forwarded to the chosen optimiser, so e.g. you can
        pass ``eps_thr=1e-10`` for IRLS or ``i_max=10000`` for SGPTL.
        """
        X_hidden = self.transform(X_raw)

        if solver == 'irls':
            result = irls(X_hidden, y, self.lam, **solver_kwargs)
        elif solver == 'dsm':
            result = deflected_subgradient(X_hidden, y, self.lam, **solver_kwargs)
        else:
            raise ValueError(f"Unknown solver '{solver}'. Use 'irls' or 'dsm'.")

        self.w = result['w']
        self._fit_result = result
        return self

    def predict(self, X_raw):
        if self.w is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self.transform(X_raw) @ self.w

    @property
    def sparsity(self):
        """Fraction of output weights with |w_i| < 1e-8 (None before fit)."""
        if self.w is None:
            return None
        return float(np.mean(np.abs(self.w) < 1e-8))

    @property
    def n_active(self):
        """Count of weights with |w_i| >= 1e-8 (None before fit)."""
        if self.w is None:
            return None
        return int(np.sum(np.abs(self.w) >= 1e-8))

    def __repr__(self):
        return (f"ELM(d={self.d}, p={self.p}, "
                f"activation='{self.activation}', lam={self.lam})")
