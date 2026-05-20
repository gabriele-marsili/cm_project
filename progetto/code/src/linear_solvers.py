"""SPD solvers used by IRLS: dense Cholesky and Jacobi-preconditioned CG."""

import numpy as np
import scipy.linalg as la


def cholesky_solve(Q, b):
    """Solve Q x = b for SPD Q via Cholesky factorization (Q = L L^T)."""
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


def conjugate_gradient(Q, b, x0=None, tol=1e-10, max_iter=None, precond=None):
    """Preconditioned CG for SPD Q x = b. Returns (x, iters_done).

    precond: vector applied as M^{-1} = diag(precond). If None, uses Jacobi
    (precond_i = 1 / Q_ii), which requires strictly positive diag(Q).
    Stops on relative residual ||r|| <= tol * ||b||. In exact arithmetic,
    converges in at most n iterations.
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - Q @ x

    if precond is None:
        precond = 1.0 / np.diag(Q)   # Jacobi preconditioner (assumes diag(Q) > 0)
    z = precond * r
    p = z.copy()
    rz = r @ z

    bn = np.linalg.norm(b)
    if bn == 0:
        return x, 0

    for k in range(max_iter):
        if np.linalg.norm(r) <= tol * bn:
            return x, k
        Qp = Q @ p
        alpha = rz / (p @ Qp)
        x += alpha * p
        r -= alpha * Qp
        z = precond * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new

    return x, max_iter


def solve_spd(Q, b, method='cholesky', return_info=False, **kwargs):
    """Solve SPD system Q x = b. method ∈ {'cholesky', 'cg'}.

    return_info=False (default): returns x.
    return_info=True: returns (x, info) where info is None for Cholesky and
    the number of CG iterations performed for CG. Useful for diagnostics.
    """
    if method == 'cholesky':
        x = cholesky_solve(Q, b)
        return (x, None) if return_info else x
    if method == 'cg':
        x, n_iter = conjugate_gradient(Q, b, **kwargs)
        return (x, n_iter) if return_info else x
    raise ValueError(f"Unknown method: {method!r}")
