"""
linear_solvers.py
-----------------
Linear system solvers for SPD systems arising in IRLS:

    Q w = b,   Q = X^T X + 2 lambda W_k^T W_k   (symmetric positive definite)

Two methods are provided:
  - Cholesky factorization  (direct, O(n^3/3) + O(n^2) per solve)
  - Conjugate Gradient      (iterative, O(k n^2) for k steps)

Note: scipy.linalg.cho_factor / cho_solve are used for Cholesky
      (basic linear algebra calls are allowed).
"""

import numpy as np
import scipy.linalg as la


# ---------------------------------------------------------------------------
# cholesky solver
# ---------------------------------------------------------------------------
def cholesky_solve(Q, b):
    """
    Solve the SPD system Q w = b via Cholesky factorization.

    Q = L L^T  (Cholesky factor computed once per call)
    Then:
      1. Forward substitution:  L v = b
      2. Back substitution:     L^T w = v

    Parameters
    ----------
    Q : ndarray (n, n)  -- symmetric positive definite matrix
    b : ndarray (n,)    -- right-hand side

    Returns
    -------
    w : ndarray (n,)    -- solution
    """
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


# ---------------------------------------------------------------------------
# conjugate gradient solver
# ---------------------------------------------------------------------------
def conjugate_gradient(Q, b, x0=None, tol=1e-10, max_iter=None):
    """
    Solve the SPD system Q w = b via the Conjugate Gradient method.

    Converges in at most n iterations in exact arithmetic.
    Preferred for large n where Cholesky O(n^3) is too expensive.

    Parameters
    ----------
    Q        : ndarray (n, n)  -- SPD matrix
    b        : ndarray (n,)    -- right-hand side
    x0       : ndarray (n,)    -- initial guess (default: zeros)
    tol      : float           -- relative residual tolerance
    max_iter : int             -- maximum iterations (default: n)

    Returns
    -------
    x        : ndarray (n,)    -- approximate solution
    n_iter   : int             -- iterations performed
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - Q @ x
    p = r.copy()
    rr = np.dot(r, r)
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return x, 0

    for k in range(max_iter):
        if np.sqrt(rr) <= tol * b_norm:
            return x, k
        Qp = Q @ p
        alpha = rr / np.dot(p, Qp)
        x = x + alpha * p
        r = r - alpha * Qp
        rr_new = np.dot(r, r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x, max_iter


# ---------------------------------------------------------------------------
# unified interface
# ---------------------------------------------------------------------------
def solve_spd(Q, b, method='cholesky', **kwargs):
    """
    Solve SPD system Q w = b.

    Parameters
    ----------
    Q      : ndarray (n, n)   -- SPD coefficient matrix
    b      : ndarray (n,)     -- right-hand side
    method : str              -- 'cholesky' or 'cg'
    kwargs : passed to CG solver if method='cg'

    Returns
    -------
    w : ndarray (n,)
    """
    if method == 'cholesky':
        return cholesky_solve(Q, b)
    elif method == 'cg':
        w, _ = conjugate_gradient(Q, b, **kwargs)
        return w
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'cholesky' or 'cg'.")
