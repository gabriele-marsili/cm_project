"""LASSO objective and the small primitives the two solvers share.

f(w) = 1/2 ||Xw - y||^2 + lam ||w||_1
"""

import numpy as np


def f_lasso(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    r = X @ w - y
    return float(0.5 * r @ r + lam * np.sum(np.abs(w)))


def grad_smooth(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # gradient of the smooth half only -> X^T(Xw - y)
    return X.T @ (X @ w - y)


def subgradient_f(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float,
) -> np.ndarray:
    # one subgradient of f: on w_i = 0 we take sign(0) = 0 (the 0 of the
    # subdifferential), which is what np.sign already returns
    return grad_smooth(X, y, w) + lam * np.sign(w)


def optimality_gap(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, f_star: float,
) -> float:
    return f_lasso(X, y, w, lam) - f_star


def support_metrics(
    w_true: np.ndarray, w_hat: np.ndarray, tol: float = 1e-3,
) -> dict:
    """Precision / recall / F1 of supp(w_hat) vs supp(w_true), threshold tol

    Empty-denominator conventions:
        nothing predicted active -> precision = 0
        true support empty -> recall = 1 (nothing was there to miss)

    The recall case disagrees with sklearn (which returns 0 + a warning), so
    anyone cross-checking on an empty true support has to convert.
    """
    s_true = np.abs(w_true) >= tol
    s_hat = np.abs(w_hat) >= tol
    tp = int(np.sum(s_true & s_hat))
    fp = int(np.sum(~s_true & s_hat))
    fn = int(np.sum(s_true & ~s_hat))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "tol": tol,
        "sparsity": float(np.mean(~s_hat)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def check_optimality(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float,
    zero_tol: float = 1e-6,
) -> float:
    """Worst KKT residual over the coordinates (0 => w is optimal up to tol).

    Coordinate split, with g = X^T(Xw - y):
        |w_i| < zero_tol  ->  must have |g_i| <= lam   (residual = max(0, |g_i| - lam))
        otherwise         ->  must have g_i = -lam sign(w_i)
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
