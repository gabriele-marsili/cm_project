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
from .linear_solvers import solve_spd


# ---------------------------------------------------------------------------
# Optimal deflection parameter (closed-form)
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
        return 1.0                          # g == d_prev: use pure subgradient
    gamma_star = (np.dot(d_prev, d_prev) - np.dot(g, d_prev)) / denom
    return float(np.clip(gamma_star, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def deflected_subgradient(X, y, lam,
                           w0=None,
                           i_max=5000,
                           beta=1.0,
                           delta0=None,
                           R=None,
                           rho=0.95,
                           f_star=None,
                           verbose=False,
                           verbose_freq=500):
    """
    Deflected Subgradient Method with Target Level for LASSO.

    Parameters
    ----------
    X         : ndarray (m, n)   -- feature matrix
    y         : ndarray (m,)     -- target vector
    lam       : float            -- regularization parameter lambda > 0
    w0        : ndarray (n,)     -- initial iterate
                                    (default: OLS solution (X^T X)^{-1} X^T y,
                                    same warm start as IRLS — report § 3.4)
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
    # Initialization: OLS warm start (matches IRLS, report § 3.4)
    #     w_0 = (X^T X)^{-1} X^T y
    # The tiny ridge term protects against rank-deficient X^T X without
    # affecting the solution when X has full column rank (the regime
    # the report assumes throughout).
    # ------------------------------------------------------------------
    if w0 is None:
        try:
            A0 = X.T @ X + 1e-12 * np.eye(n)
            w = solve_spd(A0, X.T @ y, method='cholesky')
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    if delta0 is None:
        delta0 = max(0.1 * f_curr, 1e-4)
    if R is None:
        R = 10.0 * np.sqrt(i_max)

    # Algorithm state
    r       = 0.0           # accumulated travel without improvement (report Alg 2)
    no_imp  = 0             # consecutive iterations without sufficient decrease;
                            # safeguard against the gamma_i -> 0 deadlock that
                            # freezes the travel-based patience mechanism when
                            # the deflection greedily aligns d_i with d_{i-1}
                            # (then alpha_i -> 0 and r stops growing).
    R_iter  = max(int(i_max / 100), 50)  # patience in iteration count
    delta   = delta0
    f_ref   = f_curr        # best reference value seen
    f_bar   = f_curr        # record value (best f found so far)
    w_best  = w.copy()      # iterate achieving f_bar
    d_prev  = np.zeros(n)   # d_{-1} = 0

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    f_vals      = [f_curr]
    f_bar_list  = [f_bar]
    gaps        = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times       = [0.0]
    delta_hist  = [delta]
    t_start     = time.perf_counter()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    i = -1   # so that n_iter = i + 1 == 0 if the loop body never runs
    for i in range(i_max):
        # Step 1: compute a subgradient g in partial f(w_i)
        g = subgradient_f(X, y, w, lam)

        # Step 2: optimal deflection parameter gamma_i.
        # When d_prev is the zero vector (first iteration, or after a
        # patience reset that cleared the memory) the unconstrained
        # minimiser of ||gamma g + (1 - gamma) d_prev||^2 is gamma = 0,
        # which would freeze the algorithm. We use the same convention
        # the report adopts at i = 0: take a pure subgradient step
        # (gamma = 1).
        if i == 0 or np.dot(d_prev, d_prev) < 1e-30:
            gamma = 1.0
        else:
            gamma = _optimal_gamma(g, d_prev)

        # Step 3: deflected direction d_i = gamma g + (1-gamma) d_{i-1}
        d = gamma * g + (1.0 - gamma) * d_prev

        d_norm_sq = np.dot(d, d)
        if d_norm_sq < 1e-30:
            # direction is zero — at optimum or numerical issue
            break

        # Step 4: Polyak stepsize with target level (report Alg 2 lines 11-12)
        #   beta_i = min(beta, gamma_i)        # stepsize-restricted condition
        #   alpha_i = beta_i * (f(w_i) - (f_ref - delta)) / ||d_i||^2
        # The clipping enforces beta_i <= gamma_i (report § 3.2), which is
        # what the convergence proof of Theorem 3.1 (Eq 3.14) relies on.
        # We clip a negative numerator to zero rather than skipping the
        # iteration: a zero step is harmless (w_new = w, f_new = f_curr)
        # and lets the patience mechanism count this iteration as a
        # non-improvement, eventually triggering a delta contraction. The
        # original "numerator <= 0 -> continue" branch bypassed the
        # patience update and could deadlock when beta_i = gamma_i -> 0.
        beta_i = min(beta, gamma)
        target = f_ref - delta
        numerator = max(0.0, beta_i * (f_curr - target))
        alpha = numerator / d_norm_sq

        # Step 5: update iterate
        w_new = w - alpha * d

        # Safety: a non-finite w_new should not happen with the clipped
        # numerator and a finite d, but guard against numerical surprises:
        # treat it as a zero step.
        if not np.all(np.isfinite(w_new)):
            w_new = w.copy()

        # Step 6: evaluate f at new iterate
        f_new = f_lasso(X, y, w_new, lam)

        # Step 7: update record value BEFORE the if-elseif-else block
        if f_new < f_bar:
            f_bar  = f_new
            w_best = w_new.copy()

        # Step 8: target-level logic (report Alg 2 lines 14-21) plus a
        # consecutive-no-improvement safeguard. The travel-based patience
        # `r > R` from the report can fail to trigger when alpha_i -> 0
        # (deflection greedy gamma -> 0 deadlock); the iteration-count
        # variant catches this case. When either trigger fires we also
        # reset d_prev to zero so the next iterate uses a pure subgradient
        # step (gamma = 1 by convention), which is guaranteed to make
        # progress while still satisfying the per-step bound (3.14) since
        # the proof works iterate-by-iterate without requiring persistent
        # memory.
        if f_new <= f_ref - delta / 2.0:
            f_ref  = f_bar
            r      = 0.0
            no_imp = 0
            d_prev = d
        elif r > R or no_imp > R_iter:
            delta *= rho
            r      = 0.0
            no_imp = 0
            d_prev = np.zeros(n)   # reset memory to escape stagnation
        else:
            r      += alpha * np.sqrt(d_norm_sq)
            no_imp += 1
            d_prev = d

        # Move to next iterate
        w      = w_new
        f_curr = f_new

        # Record
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
