"""Synthetic and real data generators for the experiments."""

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.preprocessing import StandardScaler

from .elm import _ACTIVATIONS
from .lasso_utils import f_lasso

# sklearn minimises (1/(2M))*||Xw - y||^2 + alpha*||w||_1
# our f_lasso uses (1/2)*||Xw - y||^2 + lam*||w||_1. Same argmin iff alpha = lam / M.

_COL_NORM_FLOOR = 1e-12 # guard against zero-norm columns when normalising
_SK_REF_MAX_ITER = 100_000 # sklearn Lasso budget for the reference w*
_SK_REF_TOL = 1e-12 # sklearn Lasso tol for the reference w*


def make_lasso_problem(
    n: int = 100,
    m: int = 200,
    sparsity: float = 0.1,
    noise_std: float = 0.1,
    lam: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Synthetic linear LASSO problem with a planted sparse w_true.

    Steps: (1) draw N(0,1) design and normalise columns to unit l2 norm;
    (2) draw sparse w_true with floor(n*sparsity) N(0,1) entries; (3) add
    Gaussian noise; (4) compute the reference (f*, w*) with sklearn Lasso.
    Returns (X, y, w_true, f_star, w_star).
    """
    rng = np.random.RandomState(random_state)

    n_active = max(1, int(sparsity * n))
    w_true = np.zeros(n)
    active = rng.choice(n, size=n_active, replace=False)
    w_true[active] = rng.randn(n_active)

    X_raw = rng.randn(m, n)
    X = X_raw / (np.linalg.norm(X_raw, axis=0, keepdims=True) + _COL_NORM_FLOOR)

    y = X @ w_true + noise_std * rng.randn(m)

    sk = SklearnLasso(alpha=lam / m, fit_intercept=False, max_iter=_SK_REF_MAX_ITER, tol=_SK_REF_TOL)
    sk.fit(X, y)
    w_star = sk.coef_
    f_star = f_lasso(X, y, w_star, lam)

    return X, y, w_true, f_star, w_star


def make_elm_problem(
    d: int = 20,
    p: int = 100,
    m: int = 500,
    sparsity: float = 0.1,
    noise_std: float = 0.1,
    activation: str = "sigmoid",
    lam: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Synthetic ELM-transformed LASSO problem.

    Like make_lasso_problem but the design is X_hid = sigma(X_raw @ W1^T)
    with a random ELM hidden layer (W1 ~ N(0,1), shape (p, d)). Columns of
    X_hid are NOT renormalised: bounded activations already keep them in a
    fixed range. Returns (X_raw, X_hid, y, W1, w_true, f_star, w_star).
    """
    sigma = _ACTIVATIONS[activation]
    rng = np.random.RandomState(random_state)

    W1 = rng.randn(p, d)
    X_raw = rng.randn(m, d)
    X_hid = sigma(X_raw @ W1.T)

    n_active = max(1, int(sparsity * p))
    w_true = np.zeros(p)
    active = rng.choice(p, size=n_active, replace=False)
    w_true[active] = rng.randn(n_active)

    y = X_hid @ w_true + noise_std * rng.randn(m)

    sk = SklearnLasso(alpha=lam / m, fit_intercept=False, max_iter=_SK_REF_MAX_ITER, tol=_SK_REF_TOL)
    sk.fit(X_hid, y)
    w_star = sk.coef_
    f_star = f_lasso(X_hid, y, w_star, lam)

    return X_raw, X_hid, y, W1, w_true, f_star, w_star


def load_real_dataset(
    name: str = "diabetes",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a real regression dataset and split, scaling features on the train
    split only (no test-set leakage).

    Returns (X_train, X_test, y_train, y_test). Target y is not rescaled;
    callers that need y standardisation handle it downstream against the
    training-set mean/std (as the experiment scripts do).
    """
    if name == 'diabetes':
        data = load_diabetes()
    elif name == 'california':
        data = fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    X, y = data.data, data.target

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(y))
    n_test = int(test_size * len(y))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    scaler = StandardScaler().fit(X[train_idx])
    X = scaler.transform(X)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
