"""SPD solvers used by IRLS: dense Cholesky and Jacobi-preconditioned CG."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as la


def cholesky_solve(Q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Q x = b for SPD Q via Cholesky factorization (Q = L L^T)."""
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


# inner products (used as squared norms) below this are treated as zero: either CG breakdown on a non-SPD matrix or a reached fixed point
_NUMERICAL_FLOOR = 1e-30


def conjugate_gradient(
    Q: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
    precond: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Preconditioned CG for SPD Q x = b. Returns (x, iters_done).

    precond: M^{-1} = diag(precond). Default: Jacobi precond_i = 1 / Q_ii (requires strictly positive diag(Q)). Stops on ||r|| <= tol * ||b|| or on breakdown (p @ Qp <= 0, i.e. loss of SPD), returning the current iterate. the caller should check the residual before using it.
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - Q @ x

    if precond is None:
        precond = 1.0 / np.diag(Q)
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
        pQp = p @ Qp
        if pQp <= _NUMERICAL_FLOOR:
            # SPD lost to round-off, return the current iterate
            return x, k
        alpha = rz / pQp
        x += alpha * p
        r -= alpha * Qp
        z = precond * r
        rz_new = r @ z
        if abs(rz) <= _NUMERICAL_FLOOR:
            return x, k
        p = z + (rz_new / rz) * p
        rz = rz_new

    return x, max_iter


def solve_spd(
    Q: np.ndarray,
    b: np.ndarray,
    method: str = "cholesky",
    return_info: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[int]]]:
    """Solve SPD system Q x = b. method in {'cholesky', 'cg'}.

    return_info=True returns (x, info). info is None for cholesky and the CG iteration count for CG.
    """
    if method == "cholesky":
        x = cholesky_solve(Q, b)
        return (x, None) if return_info else x
    if method == "cg":
        x, n_iter = conjugate_gradient(Q, b, **kwargs)
        return (x, n_iter) if return_info else x
    raise ValueError(f"Unknown method: {method!r}")
