"""Algorithm A1 -- IRLS for the LASSO.

The |w_i| in the L1 term is replaced by w_i^2 / |w_i|, which turns the
non-smooth penalty into a reweighted quadratic. 
Each outer step then solves one SPD system and re-reads the weights from the new iterate.
"""

from typing import Optional

import time

import numpy as np

from .lasso_utils import f_lasso
from .linear_solvers import solve_spd

# tiny ridge on X^T X for the OLS warm start: keeps Cholesky alive when X is
# rank-deficient, and at 1e-12 it does not move the solution in double precision
_OLS_RIDGE: float = 1e-12

# CG knobs (solver='cg'). tol kept well under eps_stop so the inner solve is
# never the bottleneck on the outer rate, iteration cap = 10*H
_CG_TOL: float = 1e-12
_CG_MAX_ITER_FACTOR: int = 10


def irls(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    eps_thr: float = 1e-8,
    eps_stop: float = 1e-8,
    k_max: int = 200,
    solver: str = "cholesky",
    w0: Optional[np.ndarray] = None,
    f_star: Optional[float] = None,
    verbose: bool = False,
) -> dict:
    """IRLS for min 1/2||Xw - y||^2 + lam||w||_1

    Outer step k solves Q_k w = X^T y with
        Q_k = X^T X + lam diag( 1 / max(|w_i|, eps_thr) )

    Args:
        eps_thr: floor on |w_i| inside the diagonal -> caps the weights and
            stops Q_k blowing up as coordinates go to zero
        eps_stop: stop when ||dw|| / max(1, ||w||) drops below this
        solver: 'cholesky' or 'cg' (see solve_spd)
        w0: warm start, default is ridge-OLS
        f_star: if set, the gap f(w_k) - f* is logged in result['gaps']

    Returns dict: w, f_vals, gaps, times, n_iter, converged
    """
    _, H = X.shape

    A = X.T @ X
    b = X.T @ y

    if w0 is None:
        # OLS warm start. Cholesky throws if X^T X is not SPD, CG just returns its last iterate
        # Either way, check the residual and drop to lstsq if the solve came back garbage
        A_reg = A + _OLS_RIDGE * np.eye(H)
        try:
            w = solve_spd(A_reg, b, method=solver)
            resid = np.linalg.norm(A_reg @ w - b)
            if not np.all(np.isfinite(w)) or resid > 1e-6 * max(1.0, np.linalg.norm(b)):
                raise np.linalg.LinAlgError("warm-start solve did not converge")
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_vals: list = []
    gaps: list = []
    times: list = []
    t0 = time.perf_counter()

    f_curr = f_lasso(X, y, w, lam)
    f_vals.append(f_curr)
    # clamp at 0: with an inexact f* proxy, f_curr can sit a hair below it and
    # the log-scale plots would choke on a negative gap
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False
    k = -1

    for k in range(k_max):
        w_old = w.copy()

        # rebuild the diagonal from the current w, add it onto A in place
        D = 1.0 / np.maximum(np.abs(w), eps_thr)
        Q = A.copy()
        Q[np.arange(H), np.arange(H)] += lam * D

        w = solve_spd(
            Q, b, method=solver,
            tol=_CG_TOL, max_iter=_CG_MAX_ITER_FACTOR * H,
        )

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
        "w": w,
        "f_vals": f_vals,
        "gaps": gaps,
        "times": times,
        "n_iter": k + 1,
        "converged": converged,
    }
