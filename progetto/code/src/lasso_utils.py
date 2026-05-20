"""LASSO objective f(w) = (1/2)||Xw-y||^2 + lam ||w||_1 and helpers."""

import numpy as np


def f_lasso(X, y, w, lam):
    """Compute f(w) = (1/2)·||Xw - y||² + λ·||w||₁."""
    r = X @ w - y
    return float(0.5 * r @ r + lam * np.sum(np.abs(w)))


def grad_smooth(X, y, w):
    """Gradient of the smooth part: X^T(Xw - y)."""
    return X.T @ (X @ w - y)


def subgradient_f(X, y, w, lam):
    """A subgradient of f at w (picks sign(0) = 0 on null coordinates)."""
    return grad_smooth(X, y, w) + lam * np.sign(w)


def optimality_gap(X, y, w, lam, f_star):
    """f(w) - f*."""
    return f_lasso(X, y, w, lam) - f_star


def support_metrics(w_true, w_hat, tol=1e-3):
    """Precision/recall/F1 of the support of w_hat against w_true at threshold tol."""
    s_true = np.abs(w_true) >= tol
    s_hat  = np.abs(w_hat)  >= tol
    tp = int(np.sum(s_true & s_hat))
    fp = int(np.sum(~s_true & s_hat))
    fn = int(np.sum(s_true & ~s_hat))
    # precision undefined when nothing is predicted positive; return 0
    # (no positives predicted = no correct positives).
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tol":       tol,
        "sparsity":  float(np.mean(~s_hat)),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def check_optimality(X, y, w, lam, zero_tol=1e-6):
    """Max KKT-residual over coordinates.

    A coordinate w_i is treated as zero when |w_i| < zero_tol.
    Returns 0 iff w satisfies the LASSO optimality system up to the tolerance.
    """
    g = grad_smooth(X, y, w)
    v = 0.0
    for i in range(len(w)):
        if abs(w[i]) < zero_tol:
            vi = max(0.0, abs(g[i]) - lam)
        else:
            vi = abs(g[i] + lam * np.sign(w[i]))
        v = max(v, vi)
    return v
