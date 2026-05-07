"""
SPD linear solvers used by IRLS.

The IRLS subproblem at iteration k is

    Q_k w = b,    Q_k = X^T X + lam W_k^T W_k    (SPD; report eq. 2.10),

so we only ever need to solve symmetric positive definite systems. We provide
two solvers, both backing the ``solve_spd`` dispatcher:

* a direct Cholesky path, which delegates the L L^T factorisation and the two
  triangular solves to ``scipy.linalg.cho_factor`` / ``cho_solve`` --- as
  allowed by §4.4 of the project comando, basic linear algebra calls go to
  the library;

* a Conjugate Gradient path implemented from scratch (§4.4 again: the linear
  solver is a main task here, so it cannot be a single library call).

Cholesky is the default for moderate H; CG is preferable when H is large and
Q_k is well conditioned.
"""

import numpy as np
import scipy.linalg as la


def cholesky_solve(Q, b):
    """Direct solver Q w = b via L L^T factorisation + two triangular solves.

    ``check_finite=False`` skips the input validation pass; we already know Q
    is finite because we just assembled it from finite operands.
    """
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


def conjugate_gradient(Q, b, x0=None, tol=1e-10, max_iter=None):
    """Standard CG for SPD systems.

    Terminates when the relative residual falls below ``tol`` or after
    ``max_iter`` iterations (default ``len(b)``). Returns the iterate and
    the number of iterations performed.
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
        x += alpha * p
        r -= alpha * Qp
        rr_new = np.dot(r, r)
        p = r + (rr_new / rr) * p
        rr = rr_new

    return x, max_iter


def solve_spd(Q, b, method='cholesky', **kwargs):
    """Dispatcher: pick Cholesky or CG. Extra kwargs are forwarded to CG."""
    if method == 'cholesky':
        return cholesky_solve(Q, b)
    if method == 'cg':
        w, _ = conjugate_gradient(Q, b, **kwargs)
        return w
    raise ValueError(f"Unknown method: '{method}'. Use 'cholesky' or 'cg'.")
