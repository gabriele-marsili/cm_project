"""Synthetic and real data generators for the experiments."""

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler


# sklearn uses (1/(2M))||Xw-y||^2 + a||w||_1; pass alpha = lam / m to match ours.

def make_lasso_problem(n=100, m=200, sparsity=0.1, noise_std=0.1,
                       lam=0.1, random_state=42):
    rng = np.random.RandomState(random_state)

    n_active = max(1, int(sparsity * n))
    w_true = np.zeros(n)
    active = rng.choice(n, size=n_active, replace=False)
    w_true[active] = rng.randn(n_active)

    X_raw = rng.randn(m, n)
    X = X_raw / (np.linalg.norm(X_raw, axis=0, keepdims=True) + 1e-12)

    y = X @ w_true + noise_std * rng.randn(m)

    sk = SklearnLasso(alpha=lam / m, fit_intercept=False,
                      max_iter=100000, tol=1e-12)
    sk.fit(X, y)
    w_star = sk.coef_

    from .lasso_utils import f_lasso
    f_star = f_lasso(X, y, w_star, lam)

    return X, y, w_true, f_star, w_star


def make_elm_problem(d=20, p=100, m=500, sparsity=0.1, noise_std=0.1,
                     activation='sigmoid', lam=0.1, random_state=42):
    from .elm import _ACTIVATIONS
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
                      max_iter=100000, tol=1e-12)
    sk.fit(X_hid, y)
    w_star = sk.coef_

    from .lasso_utils import f_lasso
    f_star = f_lasso(X_hid, y, w_star, lam)

    return X_raw, X_hid, y, W1, w_true, f_star, w_star


def load_real_dataset(name='diabetes', test_size=0.2, random_state=42):
    if name == 'diabetes':
        data = load_diabetes()
    elif name == 'california':
        data = fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)

    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(y))
    n_test = int(test_size * len(y))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
