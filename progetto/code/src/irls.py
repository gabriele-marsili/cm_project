"""
irls.py
-------
Algorithm A1: Iteratively Reweighted Least Squares (IRLS) for LASSO.

Solves:
    min_{w in R^n}  f(w) = (1/2) ||Xw - y||_2^2 + lambda * ||w||_1

Core idea (report Eq 2.6): build a smooth quadratic surrogate
    Q(w, w_k) = (1/2) ||Xw - y||^2 + (lambda/2) w^T W_k^T W_k w + (lambda/2) ||w_k||_1
with  (W_k)_{ii} = max(|w_{k,i}|, eps_thr)^{-1/2}, so that (W_k^T W_k)_{ii} = 1/|w_{k,i}|.

Setting nabla_w Q = 0 gives the normal equations
    (X^T X + lambda W_k^T W_k) w_{k+1} = X^T y.

Reference: Report Chapter 2 (approved theory).
"""

import time
import numpy as np
from .linear_solvers import solve_spd
from .lasso_utils import f_lasso


def irls(X, y, lam,
         eps_thr=1e-8,
         eps_stop=1e-8,
         k_max=200,
         solver='cholesky',
         w0=None,
         f_star=None,
         verbose=False):
    """
    Iteratively Reweighted Least Squares for LASSO.

    Parameters
    ----------
    X        : ndarray (m, n)   -- feature matrix (hidden-layer activations)
    y        : ndarray (m,)     -- target vector
    lam      : float            -- regularization parameter lambda > 0
    eps_thr  : float            -- safety threshold to avoid division by zero
                                   (typical range: 1e-6 to 1e-10)
    eps_stop : float            -- relative change stopping criterion
    k_max    : int              -- maximum number of iterations
    solver   : str              -- 'cholesky' or 'cg'
    w0       : ndarray (n,)     -- initial iterate (default: ordinary LS solution)
    f_star   : float or None    -- known optimal value for gap tracking
    verbose  : bool             -- print iteration info

    Returns
    -------
    result : dict with keys
        'w'         : ndarray (n,)   -- solution found
        'f_vals'    : list of float  -- f(w_k) at each iteration
        'gaps'      : list of float  -- f(w_k)-f* (only if f_star given)
        'times'     : list of float  -- cumulative CPU time per iteration
        'n_iter'    : int            -- iterations performed
        'converged' : bool
    """
    m, n = X.shape

    # ------------------------------------------------------------------
    # Pre-compute A = X^T X and b = X^T y  (done ONCE, O(m n^2))
    # ------------------------------------------------------------------
    A = X.T @ X          # (n, n)
    b = X.T @ y          # (n,)

    # ------------------------------------------------------------------
    # Initialization: w0 = ordinary least-squares solution A^{-1} b
    # ------------------------------------------------------------------
    if w0 is None:
        try:
            w = solve_spd(A + 1e-12 * np.eye(n), b, method=solver)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    f_vals = []
    gaps   = []
    times  = []
    t_start = time.perf_counter()

    f_curr = f_lasso(X, y, w, lam)
    f_vals.append(f_curr)
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for k in range(k_max):
        w_old = w.copy()

        # Step 1: compute diagonal weights  D_k = diag(max(|w_i|, eps_thr)^{-1})
        diag_D = 1.0 / np.maximum(np.abs(w), eps_thr)   # shape (n,)

        # Step 2: assemble Q_k = A + lambda * D_k
        #   For f = (1/2)||Xw-y||^2 + lambda||w||_1 (report Eq 1.3), the
        #   normal equation derived from Eq (2.6) is
        #     (X^T X + lambda W_k^T W_k) w_{k+1} = X^T y,
        #   with (W_k^T W_k)_{ii} = 1/max(|w_i|, eps_thr) = diag_D[i].
        #   Only the diagonal of A changes each iteration => O(n).
        Q = A.copy()
        Q[np.arange(n), np.arange(n)] += lam * diag_D

        # Step 3: solve Q_k w_{k+1} = b
        w = solve_spd(Q, b, method=solver)

        # Record
        t_elapsed = time.perf_counter() - t_start
        f_curr = f_lasso(X, y, w, lam)
        f_vals.append(f_curr)
        times.append(t_elapsed)
        if f_star is not None:
            gaps.append(max(0.0, f_curr - f_star))

        # Step 4: stopping criterion (relative iterate change)
        delta_w = np.linalg.norm(w - w_old) / max(1.0, np.linalg.norm(w_old))

        if verbose:
            gap_str = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  IRLS iter {k+1:4d}:  f={f_curr:.6e}  dw={delta_w:.2e}{gap_str}")

        if delta_w < eps_stop:
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
