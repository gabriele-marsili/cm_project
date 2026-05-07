"""
irls.py
-------
Algorithm A1: Iteratively Reweighted Least Squares (IRLS) for LASSO.

Solves:
    min_{w in R^n}  f(w) = ||Xw - y||_2^2 + lambda * ||w||_1

Core idea: approximate ||w||_1 with the quadratic surrogate w^T W_k^T W_k w,
where W_k is diagonal with  (W_k)_{ii} = max(|w_{k,i}|, eps_thr)^{-1/2}.

This turns the non-smooth problem into a sequence of weighted least-squares
(Ridge) problems, each solved via the normal equations:
    (X^T X + 2 lambda W_k^T W_k) w_{k+1} = X^T y

Reference: Algorithm 1, guida_teorica.pdf (Project 25).
"""

import time
import numpy as np
from .linear_solvers import solve_spd
from .lasso_utils import f_lasso


def irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-8, k_max=200, solver='cholesky', w0=None, f_star=None, verbose=False):
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

    # pre-compute A = X^T X and b = X^T y
    A = X.T @ X # (n, n)
    b = X.T @ y # (n,)

    # initialization: w0 = ordinary least-squares solution A^{-1} b
    if w0 is None:
        try:
            w = solve_spd(A + 1e-12 * np.eye(n), b, method=solver) # solves w = A^-1 b
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    # ------------------------------------------------------------------
    # storage
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
    # main loop
    # ------------------------------------------------------------------
    for k in range(k_max):
        w_old = w.copy()

        # step 1: compute diagonal weights  D_k = diag(max(|w_i|, eps_thr)^{-1})
        diag_D = 1.0 / np.maximum(np.abs(w), eps_thr) # shape (n,)

        # step 2: assemble Q_k = A + lambda * D_k  (= A + 2*lambda_IRLS * Wk^T Wk)
        #   For f = 0.5*||Xw-y||^2 + lambda*||w||_1, the surrogate gradient gives:
        #   (X^TX + lambda * Wk^TWk) * w = X^Ty,  where (Wk^TWk)_ii = 1/|w_i|.
        Q = A.copy()
        Q[np.arange(n), np.arange(n)] += lam * diag_D # lam is exactly \lambda_{LASSO}

        # step 3: solve Q_k w_{k+1} = b
        w = solve_spd(Q, b, method=solver)

        # record
        t_elapsed = time.perf_counter() - t_start
        f_curr = f_lasso(X, y, w, lam)
        f_vals.append(f_curr)
        times.append(t_elapsed)
        if f_star is not None:
            gaps.append(max(0.0, f_curr - f_star))

        # step 4: stopping criterion (relative iterate change)
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
