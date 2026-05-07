"""
Synthetic and real-data builders for the experiments.

For synthetic problems we sample a sparse ``w_true``, build a column-normalised
design matrix X (or its ELM-transformed counterpart), and add Gaussian noise to
form ``y = X w_true + eps``. We then call sklearn's coordinate descent on the
same design at high tolerance to obtain the reference ``f^*``, which is what
the gap plots in §5.2--§5.7 of the report measure against.

The tricky bit is the scaling. sklearn minimises ``(1/(2M)) ||Xw-y||^2 +
alpha ||w||_1``; our objective is ``(1/2) ||Xw-y||^2 + lam ||w||_1``. Multiplying
sklearn's loss by M pushes its quadratic term onto the same 1/2 factor we use,
which gives ``alpha = lam / M`` (this is exactly the mapping stated in §5.1 of
the report).

Real datasets (diabetes, California housing) ship with sklearn and are loaded
in ``load_real_dataset``.
"""

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler


def make_lasso_problem(n=100, m=200, sparsity=0.1, noise_std=0.1,
                       lam=0.1, random_state=42):
    """Synthetic LASSO instance with a known sparse ground truth.

    Returns ``(X, y, w_true, f_star, w_star)``: design + target, the sparse
    vector used to build y, and the sklearn reference optimum (both value and
    minimiser). ``f_star`` is computed with **our** objective evaluated at
    sklearn's ``w_star``, not with sklearn's internal value.
    """
    rng = np.random.RandomState(random_state)

    # Ground truth: a fraction `sparsity` of components drawn from N(0, 1),
    # the rest exactly zero.
    n_active = max(1, int(sparsity * n))
    w_true = np.zeros(n)
    active = rng.choice(n, size=n_active, replace=False)
    w_true[active] = rng.randn(n_active)

    # Random Gaussian design with unit-norm columns. Column normalisation
    # bounds cond(X^T X), which is the regime our convergence analysis
    # actually covers (report §5.1, "Synthetic data").
    X_raw = rng.randn(m, n)
    X = X_raw / (np.linalg.norm(X_raw, axis=0, keepdims=True) + 1e-12)

    y = X @ w_true + noise_std * rng.randn(m)

    # alpha = lam / M, see the module docstring for the derivation.
    sk = SklearnLasso(alpha=lam / m, fit_intercept=False,
                      max_iter=100000, tol=1e-12)
    sk.fit(X, y)
    w_star = sk.coef_

    from .lasso_utils import f_lasso
    f_star = f_lasso(X, y, w_star, lam)

    return X, y, w_true, f_star, w_star


def make_elm_problem(d=20, p=100, m=500, sparsity=0.1, noise_std=0.1,
                     activation='sigmoid', lam=0.1, random_state=42):
    """Full ELM pipeline: raw inputs -> hidden activations -> LASSO.

    Builds ``X_hidden = sigma(X_raw W_1^T)`` with a fixed random ``W_1`` and
    uses that as the design for the LASSO problem. Returns the raw inputs and
    the hidden matrix as well, in case downstream code wants to plot or
    diagnose the activation distribution.
    """
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
    """Load a real regression dataset from sklearn and split it train/test.

    Features are standardised on the **whole** dataset (not just the train
    split); ``experiment_real_data.py`` does its own train-only standardisation
    on top of this when it needs strict no-leakage guarantees.
    """
    if name == 'diabetes':
        data = load_diabetes()
    elif name == 'california':
        data = fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)

    # Manual permutation so we don't drag in sklearn.model_selection just for
    # one shuffle.
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(y))
    n_test = int(test_size * len(y))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
