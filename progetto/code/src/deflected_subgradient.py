"""
deflected_subgradient.py
------------------------
Algorithm A2: Deflected Subgradient Method with Target Level (SGPTL).

Solves:
    min_{w in R^n}  f(w) = ||Xw - y||_2^2 + lambda * ||w||_1

Key ingredients:
  - Deflection: d^i = gamma^i g^i + (1-gamma^i) d^{i-1}
      where gamma^i is chosen to minimise ||d^i||^2 (optimal deflection).
  - Stepsize-restricted Polyak rule with target level:
      alpha^i = beta * (f(w^i) - (f_ref - delta)) / ||d^i||^2,   beta <= gamma^i
  - Target-level mechanism: maintains f_ref - delta as an estimate of f*;
      reduces delta by rho when progress stalls (travel distance > R).

Reference: Algorithm 2, guida_teorica.pdf (Project 25).
"""

import time
import numpy as np
from .lasso_utils import f_lasso, subgradient_f


# ---------------------------------------------------------------------------
# optimal deflection parameter (closed-form)
# ---------------------------------------------------------------------------

def _optimal_gamma(g, d_prev):
    """
    Compute gamma* = argmin_{gamma in [0,1]} ||gamma g + (1-gamma) d_prev||^2.

    Closed-form solution:
        gamma* = (||d_prev||^2 - <g, d_prev>) / ||g - d_prev||^2,
    projected onto [0, 1].

    Parameters
    ----------
    g      : ndarray (n,) -- current subgradient
    d_prev : ndarray (n,) -- previous deflected direction

    Returns
    -------
    float in [0, 1]
    """
    diff = g - d_prev
    denom = np.dot(diff, diff)
    if denom < 1e-30:
        return 1.0 # g == d_prev: use pure subgradient
    gamma_star = (np.dot(d_prev, d_prev) - np.dot(g, d_prev)) / denom
    return float(np.clip(gamma_star, 0.0, 1.0))


# ---------------------------------------------------------------------------
# main algorithm
# ---------------------------------------------------------------------------

def deflected_subgradient(X, y, lam, w0=None, i_max=5000, beta=1.0, delta0=None, R=None, rho=0.95, f_star=None, verbose=False, verbose_freq=500):
    """
    Deflected Subgradient Method with Target Level for LASSO.

    Parameters
    ----------
    X         : ndarray (m, n)   -- feature matrix
    y         : ndarray (m,)     -- target vector
    lam       : float            -- regularization parameter lambda > 0
    w0        : ndarray (n,)     -- initial iterate (default: zeros)
    i_max     : int              -- maximum number of iterations
    beta      : float in (0,2)   -- Polyak step modulation (natural: 1.0)
    delta0    : float or None    -- initial gap estimate;
                                    default = 0.1 * f(w0)
    R         : float or None    -- patience threshold for delta reduction;
                                    default = sqrt(i_max) * 10
    rho       : float in (0,1)   -- target reduction rate (typical: 0.95)
    f_star    : float or None    -- known optimal value for gap tracking
    verbose   : bool
    verbose_freq : int           -- print every verbose_freq iterations

    Returns
    -------
    result : dict with keys
        'w'          : ndarray (n,)   -- best iterate found (at record value)
        'f_vals'     : list of float  -- f(w_i) at each iteration (current value)
        'f_bar'      : list of float  -- record values bar{f}^i
        'gaps'       : list of float  -- bar{f}^i - f* (only if f_star given)
        'times'      : list of float  -- cumulative CPU time
        'n_iter'     : int            -- iterations performed
        'delta_hist' : list of float  -- evolution of delta
    """
    m, n = X.shape

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    if w0 is None:
        # OLS warm start, same as IRLS
        A = X.T @ X
        b = X.T @ y
        try:
            from .linear_solvers import solve_spd
            w = solve_spd(A + 1e-12 * np.eye(X.shape[1]), b)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    if delta0 is None:
        delta0 = max(0.1 * f_curr, 1e-4)
    if R is None:
        R = 10.0 * np.sqrt(i_max) # empirical formula

    # algorithm state
    r = 0.0 # accumulated travel without improvement
    delta = delta0
    f_ref = f_curr # best reference value seen
    f_bar = f_curr # record value (best f found so far)
    w_best = w.copy() # iterate achieving f_bar
    d_prev = np.zeros(n) # d_{-1} = 0

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------
    f_vals = [f_curr]
    f_bar_list = [f_bar]
    gaps = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times = [0.0]
    delta_hist = [delta]
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    for i in range(i_max):
        # step 1: compute a subgradient g in partial f(w_i)
        g = subgradient_f(X, y, w, lam)

        # step 2: optimal deflection parameter gamma_i
        if i == 0:
            gamma = 1.0 # first iteration: pure subgradient
        else:
            gamma = _optimal_gamma(g, d_prev)

        # step 3: deflected direction d_i = gamma g + (1-gamma) d_{i-1}
        d = gamma * g + (1.0 - gamma) * d_prev

        d_norm_sq = np.dot(d, d)
        if d_norm_sq < 1e-30:
            # direction is zero — at optimum or numerical issue
            break

        # step 4: Polyak stepsize with target level
        # alpha_i = beta * (f(w_i) - (f_ref - delta)) / ||d_i||^2
        beta_i = min(beta, gamma) # clipping of \beta_i (ref to 3.2 of the report)
        target = f_ref - delta
        numerator = beta_i * (f_curr - target)

        if numerator <= 0.0:
            # f_curr <= target: target is too aggressive or we're below it
            # reduce delta immediately and skip step
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            # record same point
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t_start)
            continue

        alpha = numerator / d_norm_sq

        # step 5: update iterate
        w_new = w - alpha * d

        # safety: reject step if it produces NaN/inf (can happen when delta0 is initialised larger than f(w0)-f*, pushing the target below f* and causing alpha -> inf near the optimum)
        if not np.all(np.isfinite(w_new)):
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t_start)
            continue

        # step 6: evaluate f at new iterate
        f_new = f_lasso(X, y, w_new, lam)

        # step 7: update record value before the if block
        if f_new < f_bar:
            f_bar  = f_new
            w_best = w_new.copy()

        # step 8: target-level logic
        if f_new <= f_ref - delta / 2.0:
            # significant improvement: reset reference and counter
            f_ref = f_bar
            r = 0.0
        elif r > R:
            # stalled: reduce target (lower expectations)
            delta *= rho
            r = 0.0
        else:
            # accumulate travel distance
            r += alpha * np.sqrt(d_norm_sq)

        # move to next iterate
        w = w_new
        f_curr = f_new
        d_prev = d

        # record
        t_elapsed = time.perf_counter() - t_start
        f_vals.append(f_curr)
        f_bar_list.append(f_bar)
        delta_hist.append(delta)
        times.append(t_elapsed)
        if f_star is not None:
            gaps.append(max(0.0, f_bar - f_star))

        if verbose and (i + 1) % verbose_freq == 0:
            gap_str = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  DSM iter {i+1:6d}:  f={f_curr:.6e}  f_bar={f_bar:.6e}"
                  f"  delta={delta:.2e}{gap_str}")

    return {
        'w':          w_best,
        'f_vals':     f_vals,
        'f_bar':      f_bar_list,
        'gaps':       gaps,
        'times':      times,
        'n_iter':     i + 1,
        'delta_hist': delta_hist,
    }
