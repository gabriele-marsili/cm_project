"""
data_generation.py
------------------
Synthetic and real dataset utilities for Project 25 experiments.

Synthetic data generation:
  - Generate sparse true weights w* in R^n
  - Build random ELM feature matrix X in R^{m x n}
  - Set y = X w* + noise
  - Compute reference f* using sklearn.linear_model.Lasso

This allows gap plots  f(w_k) - f*  for both algorithms.
"""

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Synthetic LASSO data (with known optimal f*)
# ---------------------------------------------------------------------------

def make_lasso_problem(n=100, m=200, sparsity=0.1, noise_std=0.1, lam=0.1, random_state=42):
    """
    Generate a synthetic LASSO problem with known sparse ground truth.

    Parameters
    ----------
    n          : int    -- number of features (hidden neurons p)
    m          : int    -- number of samples
    sparsity   : float  -- fraction of non-zero components in w*
    noise_std  : float  -- standard deviation of additive Gaussian noise
    lam        : float  -- regularization parameter (used to compute f*)
    random_state : int  -- random seed

    Returns
    -------
    X      : ndarray (m, n)   -- feature matrix
    y      : ndarray (m,)     -- targets
    w_true : ndarray (n,)     -- ground-truth sparse weights
    f_star : float            -- optimal LASSO value (from sklearn)
    w_star : ndarray (n,)     -- optimal LASSO solution (from sklearn)
    """
    rng = np.random.RandomState(random_state)

    # sparse ground truth
    k = max(1, int(sparsity * n))
    w_true = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    w_true[idx] = rng.randn(k)

    # feature matrix (normalized columns improve conditioning)
    X_raw = rng.randn(m, n)
    X = X_raw / (np.linalg.norm(X_raw, axis=0, keepdims=True) + 1e-12)

    # targets
    y = X @ w_true + noise_std * rng.randn(m)

    # reference solution via sklearn (validation only, not for algorithm)
    # sklearn Lasso minimizes (1/(2m)) ||Xw-y||^2 + alpha ||w||_1
    # our f = ||Xw-y||^2 + lam ||w||_1
    # dividing ours by 2m: (1/(2m))||Xw-y||^2 + (lam/(2m))||w||_1
    # so alpha = lam / (2m)
    alpha_sk = lam / (2.0 * m)
    sk = SklearnLasso(alpha=alpha_sk, fit_intercept=False, max_iter=100000, tol=1e-12)
    sk.fit(X, y)
    w_star = sk.coef_

    # compute f* with OUR objective
    from .lasso_utils import f_lasso
    f_star = f_lasso(X, y, w_star, lam)

    return X, y, w_true, f_star, w_star


# ---------------------------------------------------------------------------
# ELM-specific synthetic data
# ---------------------------------------------------------------------------

def make_elm_problem(d=20, p=100, m=500, sparsity=0.1, noise_std=0.1, activation='sigmoid', lam=0.1, random_state=42):
    """
    Generate synthetic data for the full ELM pipeline.

    Creates:
      1. Raw input X_raw in R^{m x d}
      2. Random ELM weights W1 in R^{p x d}  (fixed)
      3. Hidden features X = sigma(X_raw W1^T) in R^{m x p}
      4. Sparse w* in R^p
      5. Targets y = X w* + noise

    Parameters
    ----------
    d          : int    -- raw input dimension
    p          : int    -- number of hidden neurons
    m          : int    -- number of training samples
    sparsity   : float  -- fraction of non-zero components in w*
    noise_std  : float  -- noise standard deviation
    activation : str    -- 'sigmoid', 'relu', or 'tanh'
    lam        : float  -- regularization parameter
    random_state : int

    Returns
    -------
    X_raw  : ndarray (m, d)   -- raw inputs
    X_hid  : ndarray (m, p)   -- hidden activations
    y      : ndarray (m,)     -- targets
    W1     : ndarray (p, d)   -- fixed hidden weights
    w_true : ndarray (p,)     -- ground-truth output weights
    f_star : float            -- optimal LASSO value (from sklearn)
    w_star : ndarray (p,)     -- optimal LASSO solution (from sklearn)
    """
    from .elm import _ACTIVATIONS
    sigma = _ACTIVATIONS[activation]

    rng = np.random.RandomState(random_state)

    # ELM architecture
    W1    = rng.randn(p, d)
    X_raw = rng.randn(m, d)
    X_hid = sigma(X_raw @ W1.T)   # (m, p)

    # sparse ground-truth output weights
    k = max(1, int(sparsity * p))
    w_true = np.zeros(p)
    idx = rng.choice(p, size=k, replace=False)
    w_true[idx] = rng.randn(k)

    # targets
    y = X_hid @ w_true + noise_std * rng.randn(m)

    # reference optimal value (sklearn)
    alpha_sk = lam * m / 2.0 # ??????????
    sk = SklearnLasso(alpha=alpha_sk, fit_intercept=False, max_iter=100000, tol=1e-12)
    sk.fit(X_hid, y)
    w_star = sk.coef_

    from .lasso_utils import f_lasso
    f_star = f_lasso(X_hid, y, w_star, lam)

    return X_raw, X_hid, y, W1, w_true, f_star, w_star


# ---------------------------------------------------------------------------
# Real datasets (from sklearn)
# ---------------------------------------------------------------------------

def load_real_dataset(name='diabetes', test_size=0.2, random_state=42):
    """
    Load and preprocess a real regression dataset.

    Parameters
    ----------
    name        : str   -- 'diabetes' or 'california'
    test_size   : float -- fraction of data for test set
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
    """
    if name == 'diabetes':
        data = load_diabetes()
    elif name == 'california':
        data = fetch_california_housing()
    else:
        raise ValueError(f"Unknown dataset '{name}'.")

    X, y = data.data, data.target

    # standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train-test split (manual, no sklearn split to avoid dependency)
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    n_test = int(test_size * len(y))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
