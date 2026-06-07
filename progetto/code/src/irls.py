"""Algorithm A1: IRLS for the LASSO."""

from typing import Optional

import time

import numpy as np

from .lasso_utils import f_lasso
from .linear_solvers import solve_spd

# Ridge added to A = X^T X when computing the default OLS warm start, to keep
# the Cholesky factorisation well-defined on rank-deficient designs. Small
# enough to perturb the OLS solution below floating-point noise.
_OLS_RIDGE: float = 1e-12

# CG inner-solver settings (only relevant when solver='cg'). Tolerance is
# tight enough that the WLS inner subproblem error does not pollute the outer
# MM rate; max_iter is generous (in exact arithmetic CG converges in <= H
# steps, the cap accounts for ill-conditioned Q_k near the eps_thr floor).
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
    """Iteratively Reweighted Least Squares for LASSO (Algorithm A1).

    Solves min (1/2)||Xw - y||^2 + lam*||w||_1 by repeatedly minimising the
    smooth quadratic surrogate of Section 2 of the report. Each iteration
    solves Q_k w_{k+1} = X^T y with Q_k = X^T X + lam*diag(1/max(|w|, eps_thr)).

    Args:
        eps_thr: safety threshold for the diagonal weights.
        eps_stop: tolerance on the scale-invariant change ||dw||/max(1, ||w_k||).
        k_max: max outer iterations.
        solver: 'cholesky' or 'cg' (forwarded to solve_spd).
        w0: optional warm-start; defaults to a ridge-regularised OLS.
        f_star: if given, gaps f(w_k) - f* are stored in result['gaps'].

    Returns dict with keys: w, f_vals, gaps, times, n_iter, converged.
    """
    _, H = X.shape

    A = X.T @ X
    b = X.T @ y

    if w0 is None:
        # OLS warm start with the chosen solver. Cholesky raises
        # np.linalg.LinAlgError on a non-SPD X^T X; CG instead returns its last
        # iterate on breakdown. We validate the normal-equations residual so a
        # bad solve from *either* back-end falls back to a minimum-norm lstsq
        # solution rather than seeding IRLS with a garbage warm start.
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
    # Floor gap at 0 for log-scale plots: when f* is an inexact proxy
    # (e.g. sklearn), f_curr can fall below it.
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False
    k = -1

    for k in range(k_max):
        w_old = w.copy()

        # Weighted normal equations Q_k w = b with
        #   Q_k = A + 2 lam_irls W_k^T W_k,  lam_irls = lam_lasso / 2
        # so 2 lam_irls (W_k^T W_k)_{ii} = lam_lasso / max(|w_i|, eps_thr).
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
