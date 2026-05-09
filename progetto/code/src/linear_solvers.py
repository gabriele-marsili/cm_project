"""SPD solvers used by IRLS: dense Cholesky and Jacobi-preconditioned CG."""

import numpy as np
import scipy.linalg as la


def cholesky_solve(Q, b):
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


def conjugate_gradient(Q, b, x0=None, tol=1e-10, max_iter=None, precond=None):
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - Q @ x

    if precond is None:
        precond = 1.0 / np.diag(Q)   # Jacobi
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


def solve_spd(Q, b, method='cholesky', **kwargs):
    if method == 'cholesky':
        return cholesky_solve(Q, b)
    if method == 'cg':
        w, _ = conjugate_gradient(Q, b, **kwargs)
        return w
    raise ValueError(f"Unknown method: {method!r}")
