"""SPD solvers behind the IRLS inner step: dense Cholesky and Jacobi-PCG."""

from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as la


def cholesky_solve(Q: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Q = L L^T then two triangular solves, check_finite off since Q is built clean
    c, low = la.cho_factor(Q, lower=True, check_finite=False)
    return la.cho_solve((c, low), b, check_finite=False)


# squared quantities below this count as zero: CG breakdown on a (numerically)
# non-SPD matrix, or an exact fixed point
_NUMERICAL_FLOOR = 1e-30


def conjugate_gradient(
    Q: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
    precond: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Preconditioned CG for SPD Qx = b -> (x, iters)

    precond holds the diagonal of M^{-1}, default is Jacobi (1/Q_ii), which
    needs a strictly positive diagonal. 
    Two exits:
        ||r|| <= tol ||b|| -> converged
        p^T Q p <= 0 -> SPD lost to round-off, bail with current x
    On the second exit the iterate is not trustworthy, so the caller checks the
    residual before using it.
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
        if pQp <= _NUMERICAL_FLOOR:  # not SPD anymore -> stop here
            return x, k
        alpha = rz / pQp
        x += alpha * p
        r -= alpha * Qp
        z = precond * r
        rz_new = r @ z
        if abs(rz) <= _NUMERICAL_FLOOR:
            return x, k
        p = z + (rz_new / rz) * p  # Fletcher-Reeves beta
        rz = rz_new

    return x, max_iter


def solve_spd(
    Q: np.ndarray,
    b: np.ndarray,
    method: str = "cholesky",
    return_info: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[int]]]:
    """Dispatch to a single SPD solve. method in {'cholesky', 'cg'}

    With return_info, also hand back the CG iteration count (None for Cholesky)
    """
    if method == "cholesky":
        x = cholesky_solve(Q, b)
        return (x, None) if return_info else x
    if method == "cg":
        x, n_iter = conjugate_gradient(Q, b, **kwargs)
        return (x, n_iter) if return_info else x
    raise ValueError(f"Unknown method: {method!r}")
