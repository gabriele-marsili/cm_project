"""Algorithm A1: IRLS for the LASSO."""

import time
import numpy as np

from .linear_solvers import solve_spd
from .lasso_utils import f_lasso


def irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-8, k_max=200,
         solver='cholesky', w0=None, f_star=None, verbose=False):
    """Iteratively Reweighted Least Squares for LASSO (Algorithm A1).

    Solves min (1/2)||Xw - y||² + λ||w||₁ by repeatedly minimising the smooth
    quadratic surrogate of Section 2 of the report. Each iteration solves the
    SPD system Q_k w_{k+1} = X^T y with Q_k = X^T X + λ · diag(1/max(|w|,ε_thr)).

    Args:
        eps_thr: safety threshold for the diagonal weights.
        eps_stop: tolerance on the scale-invariant change ||Δw||/max(1,||w_k||).
        k_max: max outer iterations.
        solver: 'cholesky' or 'cg' (forwarded to solve_spd).
        w0: optional warm-start; defaults to a ridge-regularised OLS.
        f_star: if given, optimality gaps f(w_k) - f* are stored in result['gaps'].

    Returns dict with keys: w, f_vals, gaps, times, n_iter, converged.
    """
    _, H = X.shape

    A = X.T @ X
    b = X.T @ y

    if w0 is None:
        # OLS warm start with a tiny ridge to guard against rank-deficient A.
        try:
            w = solve_spd(A + 1e-12 * np.eye(H), b, method=solver)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_vals, gaps, times = [], [], []
    t0 = time.perf_counter()

    f_curr = f_lasso(X, y, w, lam)
    f_vals.append(f_curr)
    # Floor at 0 for log-scale plots; f_curr < f_star (when f* is a sklearn
    # surrogate, not the true optimum) becomes a missing point rather than
    # a negative value that breaks log axes.
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False
    k = -1

    for k in range(k_max):
        w_old = w.copy()

        # Weighted normal equations Q_k w = b with
        #   Q_k = A + 2 λ_IRLS W_k^T W_k,  λ_IRLS = λ_LASSO / 2
        # ⇒ 2 λ_IRLS · (W_k^T W_k)_{ii} = λ_LASSO / max(|w_i|, ε_thr).
        # The expression below is the same matrix, written without the 1/2.
        D = 1.0 / np.maximum(np.abs(w), eps_thr)

        Q = A.copy()
        Q[np.arange(H), np.arange(H)] += lam * D

        w = solve_spd(Q, b, method=solver, tol=1e-12, max_iter=10 * H)

        f_curr = f_lasso(X, y, w, lam)
        f_vals.append(f_curr)
        times.append(time.perf_counter() - t0)
        if f_star is not None:
            gaps.append(max(0.0, f_curr - f_star))

        
        rel = np.linalg.norm(w - w_old) / max(1.0, np.linalg.norm(w_old))

        if verbose:
            gs = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  IRLS iter {k+1:4d}:  f={f_curr:.6e}  dw={rel:.2e}{gs}")

        if rel < eps_stop:
            converged = True
            break

    return {
        'w':         w,
        'f_vals':    f_vals,
        'gaps':      gaps,
        'times':     times,
        'n_iter':    k + 1,
        'converged': converged,
    }
