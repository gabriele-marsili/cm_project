"""Problem generators: synthetic LASSO, synthetic ELM-LASSO, real datasets."""

from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.preprocessing import StandardScaler

from .elm import _ACTIVATIONS
from .lasso_utils import f_lasso

# our objective is 1/2||Xw-y||^2 + lam||w||_1
# sklearn's: 1/(2M)||Xw-y||^2 + alpha||w||_1 -> same argmin when alpha = lam/M
# we use sklearn (heavily over-iterated) only as an independent f* reference
_COL_NORM_FLOOR = 1e-12  # avoid 1/0 when a design column is all zeros
_SK_REF_MAX_ITER = 100_000  # sklearn budget for the reference w*
_SK_REF_TOL = 1e-12  # sklearn tol for the reference w*


def make_lasso_problem(
    n: int = 100,
    m: int = 200,
    sparsity: float = 0.1,
    noise_std: float = 0.1,
    lam: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Plain linear LASSO with a planted sparse w_true.

    Gaussian design with unit-norm columns, floor(n*sparsity) nonzeros in
    w_true, Gaussian noise on y. 
    Returns (X, y, w_true, f_star, w_star), where (f_star, w_star) is the sklearn reference
    """
    rng = np.random.RandomState(random_state)

    n_active = max(1, int(sparsity * n))
    w_true = np.zeros(n)
    active = rng.choice(n, size=n_active, replace=False)
    w_true[active] = rng.randn(n_active)

    X_raw = rng.randn(m, n)
    X = X_raw / (np.linalg.norm(X_raw, axis=0, keepdims=True) + _COL_NORM_FLOOR)

    y = X @ w_true + noise_std * rng.randn(m)

    sk = SklearnLasso(alpha=lam / m, fit_intercept=False,
                      max_iter=_SK_REF_MAX_ITER, tol=_SK_REF_TOL)
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
    """Same as make_lasso_problem but on ELM hidden features.

    Design is X_hid = sigma(X_raw W1^T) with a random frozen W1 (shape (p, d)).
    Hidden columns are left un-normalised -> bounded activations already keep them in range. 
    Returns (X_raw, X_hid, y, W1, w_true, f_star, w_star).
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

    sk = SklearnLasso(alpha=lam / m, fit_intercept=False,
                      max_iter=_SK_REF_MAX_ITER, tol=_SK_REF_TOL)
    sk.fit(X_hid, y)
    w_star = sk.coef_
    f_star = f_lasso(X_hid, y, w_star, lam)

    return X_raw, X_hid, y, W1, w_true, f_star, w_star


def load_real_dataset(
    name: str = "diabetes",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load + split a real regression dataset.

    The scaler is fit on the train split only (no leakage). 
    y is left on its own scale.
    The experiment scripts standardise it downstream against the train mean/std when they need to. 
    Returns (X_train, X_test, y_train, y_test)
    """
    if name == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    elif name == 'california':
        X, y = fetch_california_housing(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset {name!r}")
    X = np.asarray(X)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(y))
    n_test = int(test_size * len(y))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    scaler = StandardScaler().fit(X[train_idx])
    X = scaler.transform(X)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
