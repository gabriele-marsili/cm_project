"""Algorithm A1: IRLS for the LASSO."""

import time
import numpy as np

from .linear_solvers import solve_spd
from .lasso_utils import f_lasso


def irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-8, k_max=200,
         solver='cholesky', w0=None, f_star=None, verbose=False):
    _, H = X.shape

    A = X.T @ X
    b = X.T @ y

    if w0 is None:
        # OLS warm start; tiny ridge guards against rank-deficient X^T X.
        try:
            w = solve_spd(A + 1e-12 * np.eye(H), b, method=solver)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_vals, gaps, times = [], [], []
    t0 = time.perf_counter()

    f_curr = f_lasso(X, y, w, lam)
    f_vals.append(f_curr)
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False
    k = -1

    for k in range(k_max):
        w_old = w.copy()

        D = 1.0 / np.maximum(np.abs(w), eps_thr)

        Q = A.copy()
        Q[np.arange(H), np.arange(H)] += lam * D

        w = solve_spd(Q, b, method=solver, tol=1e-12, max_iter=10 * H)

        f_curr = f_lasso(X, y, w, lam)
        f_vals.append(f_curr)
        times.append(time.perf_counter() - t0)
        if f_star is not None:
            gaps.append(max(0.0, f_curr - f_star))

        nw = np.linalg.norm(w_old)
        rel = np.linalg.norm(w - w_old) / nw if nw > 1e-12 else np.linalg.norm(w - w_old)

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
