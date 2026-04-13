"""
elm.py
------
Extreme Learning Machine (ELM) model.

Architecture:
    - Input layer:  x in R^d
    - Hidden layer: H neurons with fixed random weights W1 in R^{p x d}
                    activations: sigma(W1 x) in R^p
    - Output layer: y_hat = w^T sigma(W1 x),   w in R^p  (learned)

Learning rule:
    Collect hidden activations for all m training samples into matrix
        X = [sigma(W1 x^(1))^T; ...; sigma(W1 x^(m))^T]  in R^{m x p}
    Then find w by solving:
        min_w  ||X w - y||_2^2 + lambda ||w||_1     (LASSO)

The LASSO is solved by either IRLS (A1) or the Deflected Subgradient Method (A2).
"""

import numpy as np
from .irls import irls
from .deflected_subgradient import deflected_subgradient


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _sigmoid(z):
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


# ---------------------------------------------------------------------------
# ELM class
# ---------------------------------------------------------------------------

class ELM:
    """
    Extreme Learning Machine with L1-regularized output layer.

    Parameters
    ----------
    d          : int    -- input dimension
    p          : int    -- number of hidden neurons
    activation : str    -- activation function: 'sigmoid', 'relu', 'tanh'
    lam        : float  -- regularization parameter lambda (default: 0.1)
    random_state : int  -- seed for reproducibility
    """

    def __init__(self, d, p, activation='sigmoid', lam=0.1, random_state=42):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Choose from {list(_ACTIVATIONS)}.")
        self.d          = d
        self.p          = p
        self.activation = activation
        self.lam        = lam
        self.sigma      = _ACTIVATIONS[activation]

        # Generate fixed random hidden-layer weights W1 in R^{p x d}
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(p, d)

        # Output weights (set after fitting)
        self.w = None
        self._fit_result = None

    # ------------------------------------------------------------------
    # Transform: compute hidden activations
    # ------------------------------------------------------------------

    def transform(self, X_raw):
        """
        Compute the hidden-layer feature matrix.

        Parameters
        ----------
        X_raw : ndarray (m, d)  -- raw input data

        Returns
        -------
        X_hidden : ndarray (m, p)  -- hidden activations sigma(X_raw W1^T)
        """
        # Each row x: W1 x = W1 @ x.  For all rows: X_raw @ W1^T
        return self.sigma(X_raw @ self.W1.T)   # (m, p)

    # ------------------------------------------------------------------
    # Fit: learn output weights
    # ------------------------------------------------------------------

    def fit(self, X_raw, y, solver='irls', **solver_kwargs):
        """
        Fit the ELM by solving the LASSO problem on the hidden features.

        Parameters
        ----------
        X_raw        : ndarray (m, d)   -- raw training inputs
        y            : ndarray (m,)     -- training targets
        solver       : str              -- 'irls' or 'dsm'
        solver_kwargs: dict             -- passed to the chosen solver

        Returns
        -------
        self   (for method chaining)
        """
        X_hidden = self.transform(X_raw)   # (m, p)

        if solver == 'irls':
            result = irls(X_hidden, y, self.lam, **solver_kwargs)
        elif solver == 'dsm':
            result = deflected_subgradient(X_hidden, y, self.lam, **solver_kwargs)
        else:
            raise ValueError(f"Unknown solver '{solver}'. Use 'irls' or 'dsm'.")

        self.w = result['w']
        self._fit_result = result
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X_raw):
        """
        Predict targets for new inputs.

        Parameters
        ----------
        X_raw : ndarray (m, d)   -- raw input data

        Returns
        -------
        y_hat : ndarray (m,)     -- predictions
        """
        if self.w is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_hidden = self.transform(X_raw)   # (m, p)
        return X_hidden @ self.w

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def sparsity(self):
        """Fraction of output weights that are (approximately) zero."""
        if self.w is None:
            return None
        return float(np.mean(np.abs(self.w) < 1e-8))

    @property
    def n_active(self):
        """Number of non-zero output weights."""
        if self.w is None:
            return None
        return int(np.sum(np.abs(self.w) >= 1e-8))

    def __repr__(self):
        return (f"ELM(d={self.d}, p={self.p}, activation='{self.activation}', "
                f"lam={self.lam})")
