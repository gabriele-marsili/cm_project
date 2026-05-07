"""
IRLS for the LASSO objective f(w) = (1/2)||Xw - y||^2 + lam * ||w||_1.

We replace the non-smooth L1 term with the AM-GM majorisation
``||w||_1 <= (1/2) w^T W_k^T W_k w + (1/2) ||w_k||_1`` (report Eq 2.3),
where (W_k)_{ii} = max(|w_{k,i}|, eps_thr)^{-1/2}. The resulting surrogate

    Q(w; w_k) = (1/2)||Xw-y||^2 + (lam/2) w^T W_k^T W_k w + const

is a strongly convex quadratic in w, minimised by the weighted normal
equations (eq. 2.9, with lam_IRLS = lam_LASSO/2 substituted):

    (X^T X + lam * W_k^T W_k) w_{k+1} = X^T y.

The eps_thr clipping in W_k is what keeps the system non-singular when a
component drifts to zero; the report discusses the trade-off in §2.7.
"""

import time
import numpy as np

from .linear_solvers import solve_spd
from .lasso_utils import f_lasso


def irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-8, k_max=200,
         solver='cholesky', w0=None, f_star=None, verbose=False):
    """Iteratively Reweighted Least Squares for the LASSO.

    Parameters
    ----------
    X, y       Design matrix (M, H) and target (M,).
    lam        L1 regularisation strength, > 0.
    eps_thr    Floor on |w_{k,i}| inside the weight update; defaults to 1e-8,
               which §5.3.1 of the report shows is the empirical sweet spot.
    eps_stop   Relative iterate-change tolerance for early termination.
    k_max      Iteration cap.
    solver     Inner linear solver: 'cholesky' (direct, recommended) or 'cg'.
    w0         Optional warm start; the OLS solution A^{-1} b is used by default.
    f_star     Reference optimum for gap tracking. If None, gaps is left empty.
    verbose    Print one line per iteration.

    Returns a dict with the iterate, the trajectory (f_vals, gaps, times),
    the iteration count, and a boolean convergence flag.
    """
    _, H = X.shape

    # We keep A and b fixed across iterations: only the diagonal piece of Q
    # changes with k, so this precomputation is amortised over the whole run.
    A = X.T @ X
    b = X.T @ y

    if w0 is None:
        # Tikhonov-regularised normal equations as a numerically safe OLS:
        # the 1e-12 ridge keeps A non-singular when X^T X is barely rank-deficient.
        try:
            w = solve_spd(A + 1e-12 * np.eye(H), b, method=solver)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_vals, gaps, times = [], [], []
    t0 = time.perf_counter()

    f_curr = f_lasso(X, y, w, lam)
    f_vals.append(f_curr)
    if f_star is not None:
        gaps.append(max(0.0, f_curr - f_star))
    times.append(0.0)

    converged = False
    k = -1   # ensures n_iter = 0 when k_max = 0 and the loop never runs

    for k in range(k_max):
        w_old = w.copy()

        # Diagonal of the reweighting D_k = W_k^T W_k = diag(1 / max(|w_{k,i}|, eps_thr)).
        D_diag = 1.0 / np.maximum(np.abs(w), eps_thr)

        # Q_k = A + lam D_k — only the diagonal of Q changes between iterations.
        Q = A.copy()
        Q[np.arange(H), np.arange(H)] += lam * D_diag

        w = solve_spd(Q, b, method=solver)

        f_curr = f_lasso(X, y, w, lam)
        f_vals.append(f_curr)
        times.append(time.perf_counter() - t0)
        if f_star is not None:
            gaps.append(max(0.0, f_curr - f_star))

        # Scale-invariant stopping rule from §2.4. The max(1, ||w_old||) avoids
        # spurious early termination when w drifts close to the origin.
        rel_change = np.linalg.norm(w - w_old) / max(1.0, np.linalg.norm(w_old))

        if verbose:
            gap_str = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  IRLS iter {k+1:4d}:  f={f_curr:.6e}  dw={rel_change:.2e}{gap_str}")

        if rel_change < eps_stop:
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
