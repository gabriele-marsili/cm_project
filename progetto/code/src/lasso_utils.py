"""
lasso_utils.py
--------------
Common utilities for the LASSO problem:
    min_{w in R^n}  f(w) = ||Xw - y||_2^2 + lambda * ||w||_1

Project 25: Extreme Learning Machine with L1 regularization.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Objective function and derivatives
# ---------------------------------------------------------------------------

def f_lasso(X, y, w, lam):
    """
    Evaluate the LASSO objective f(w) = ||Xw - y||^2 + lambda * ||w||_1.

    Parameters
    ----------
    X   : ndarray, shape (m, n)   -- feature matrix
    y   : ndarray, shape (m,)     -- target vector
    w   : ndarray, shape (n,)     -- weight vector
    lam : float                   -- regularization parameter (>0)

    Returns
    -------
    float : f(w)
    """
    residual = X @ w - y
    return float(0.5 * np.dot(residual, residual) + lam * np.sum(np.abs(w)))


def grad_smooth(X, y, w):
    """
    Gradient of the smooth part g(w) = ||Xw - y||^2:
        nabla g(w) = 2 X^T (Xw - y).

    Parameters
    ----------
    X : ndarray (m, n)
    y : ndarray (m,)
    w : ndarray (n,)

    Returns
    -------
    ndarray (n,)
    """
    return X.T @ (X @ w - y)


def subgradient_f(X, y, w, lam):
    """
    Compute a subgradient of f at w:
        g = 2 X^T (Xw - y) + lambda * s,
    where s_i = sign(w_i) if w_i != 0, and s_i = 0 if w_i = 0
    (minimum-norm element of the subdifferential of |w_i|).

    Parameters
    ----------
    X   : ndarray (m, n)
    y   : ndarray (m,)
    w   : ndarray (n,)
    lam : float

    Returns
    -------
    ndarray (n,)
    """
    g_smooth = grad_smooth(X, y, w)
    s = np.sign(w) # np.sign(0) == 0  (minimum-norm subgradient)
    return g_smooth + lam * s


def optimality_gap(X, y, w, lam, f_star):
    """
    Gap to the (known) optimum: f(w) - f*.

    Parameters
    ----------
    f_star : float  -- optimal value (e.g. from sklearn.linear_model.Lasso)

    Returns
    -------
    float >= 0
    """
    return f_lasso(X, y, w, lam) - f_star


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_optimality(X, y, w, lam, tol=1e-6):
    """
    Check the KKT optimality condition  0 in partial f(w*):
        |[2 X^T(Xw-y)]_i| <= lambda   if w_i == 0
        [2 X^T(Xw-y)]_i = -lambda s_i  if w_i != 0

    Returns the maximum violation (0 means exactly optimal).
    """
    g_smooth = grad_smooth(X, y, w)
    violation = 0.0
    for i in range(len(w)):
        if abs(w[i]) < tol:
            # w_i = 0: need |g_smooth_i| <= lambda
            viol_i = max(0.0, abs(g_smooth[i]) - lam)
        else:
            # w_i != 0: need g_smooth_i + lambda * sign(w_i) = 0
            viol_i = abs(g_smooth[i] + lam * np.sign(w[i]))
        violation = max(violation, viol_i)
    return violation
