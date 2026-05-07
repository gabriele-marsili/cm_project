"""
Shared helpers for the LASSO objective

    f(w) = (1/2) ||Xw - y||^2 + lam ||w||_1

(Eq. 1.3 of the report). The 1/2 factor is what lets ∇g(w) = X^T(Xw - y)
appear without a leading 2 throughout the rest of the code.
"""

import numpy as np


def f_lasso(X, y, w, lam):
    """LASSO objective (1/2) ||Xw - y||^2 + lam ||w||_1."""
    r = X @ w - y
    return float(0.5 * np.dot(r, r) + lam * np.sum(np.abs(w)))


def grad_smooth(X, y, w):
    """Gradient of the smooth part g(w) = (1/2) ||Xw - y||^2: ∇g = X^T(Xw - y)."""
    return X.T @ (X @ w - y)


def subgradient_f(X, y, w, lam):
    """A subgradient of f at w (report Eq 3.9).

    Component-wise we have g_i = [X^T(Xw-y)]_i + lam * s_i, where s_i = sign(w_i)
    if w_i != 0 and s_i = 0 if w_i = 0. The choice s_i = 0 at the kink is the
    minimum-norm element of ∂|w_i|, which is what minimises ||g||^2 and
    therefore maximises the achievable progress under the Polyak step.

    np.sign already returns 0 at exactly zero, so no special-casing is needed.
    """
    return grad_smooth(X, y, w) + lam * np.sign(w)


def optimality_gap(X, y, w, lam, f_star):
    """Convenience wrapper: f(w) - f_star (not clipped)."""
    return f_lasso(X, y, w, lam) - f_star


def support_metrics(w_true, w_hat, tol=1e-3):
    """Support-recovery metrics at a hard threshold ``tol``.

    Both vectors are thresholded with the *same* rule ``|w_i| >= tol`` to
    define the "selected" support; this makes IRLS and SGPTL comparable on
    sparsity even though IRLS shrinks small components down to ~eps_thr while
    SGPTL leaves them at the residual sublinear-rate level (~1e-3..1e-4).

    Returns a dict with selected-fraction, true-positive / false-positive
    counts on the support, and precision/recall/F1 with the convention that
    precision = recall = 1 when the true support is empty and so is the
    estimated one (and 0 in any other corner case).
    """
    s_true = np.abs(w_true) >= tol
    s_hat  = np.abs(w_hat)  >= tol
    tp = int(np.sum(s_true & s_hat))
    fp = int(np.sum(~s_true & s_hat))
    fn = int(np.sum(s_true & ~s_hat))
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not s_true.any() else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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


def check_optimality(X, y, w, lam, tol=1e-6):
    """Maximum violation of the KKT condition 0 ∈ ∂f(w) (report Eq 1.4).

    For each component i, the condition reads
        |[X^T(Xw - y)]_i| <= lam              if |w_i| < tol  (active set),
        [X^T(Xw - y)]_i + lam * sign(w_i) = 0 otherwise.

    The tol parameter only decides which components we treat as "exactly zero"
    when checking the inequality branch.
    """
    g_smooth = grad_smooth(X, y, w)
    violation = 0.0
    for i in range(len(w)):
        if abs(w[i]) < tol:
            v_i = max(0.0, abs(g_smooth[i]) - lam)
        else:
            v_i = abs(g_smooth[i] + lam * np.sign(w[i]))
        violation = max(violation, v_i)
    return violation
