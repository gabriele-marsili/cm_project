"""ELM with an L1-regularised output layer.

Hidden layer W1 is random and frozen.
Only the output weights w are trained, by solving a LASSO on the hidden features (IRLS or DSM).
"""

from typing import Optional

import numpy as np

from .deflected_subgradient import deflected_subgradient
from .irls import irls

# sigmoid saturates past |z| ~ 36, so clipping at 500 before exp() only kills
# the overflow on pathological inputs and never touches the useful range
_SIGMOID_CLIP: float = 500.0

# below this a weight is "off" (used for the sparsity readout)
_SPARSITY_TOL: float = 1e-8


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


_ACTIVATIONS = {"sigmoid": _sigmoid, "relu": _relu, "tanh": _tanh}


class ELM:
    def __init__(
        self,
        d: int,
        p: int,
        activation: str = "sigmoid",
        lam: float = 0.1,
        random_state: int = 42,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}")
        self.d = d
        self.p = p
        self.activation = activation
        self.lam = lam
        self.sigma = _ACTIVATIONS[activation]

        # W1 drawn once and never updated -> this is what makes it an ELM
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(p, d)

        self.w: Optional[np.ndarray] = None
        self._fit_result: Optional[dict] = None

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        # raw inputs -> hidden features sigma(X_raw W1^T), shape (M, p)
        return self.sigma(X_raw @ self.W1.T)

    def fit(self, X_raw: np.ndarray, y: np.ndarray, solver: str = "irls", **kwargs) -> "ELM":
        """Train w on the hidden features. solver 'irls' (A1) or 'dsm' (A2),
        leftover kwargs go straight to the chosen solver"""
        Xh = self.transform(X_raw)
        if solver == "irls":
            res = irls(Xh, y, self.lam, **kwargs)
        elif solver == "dsm":
            res = deflected_subgradient(Xh, y, self.lam, **kwargs)
        else:
            raise ValueError(f"Unknown solver {solver!r}")
        self.w = res["w"]
        self._fit_result = res
        return self

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Call fit() first.")
        return self.transform(X_raw) @ self.w

    @property
    def sparsity(self) -> Optional[float]:
        # fraction of off weights
        if self.w is None:
            return None
        return float(np.mean(np.abs(self.w) < _SPARSITY_TOL))

    @property
    def n_active(self) -> Optional[int]:
        if self.w is None:
            return None
        return int(np.sum(np.abs(self.w) >= _SPARSITY_TOL))

    def __repr__(self) -> str:
        return (
            f"ELM(d={self.d}, p={self.p}, "
            f"activation={self.activation!r}, lam={self.lam})"
        )
