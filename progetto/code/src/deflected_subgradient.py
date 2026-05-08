"""
Deflected Subgradient with Target Level (SGPTL) for the LASSO objective
``f(w) = (1/2) ||Xw - y||^2 + lam ||w||_1``.

The three moving parts (report §3.1.1, §3.2, §3.3):

* deflection — the search direction blends the current subgradient with the
  previous direction, ``d_i = gamma_i g_i + (1 - gamma_i) d_{i-1}``. We pick
  gamma greedily to minimise ||d_i|| (closed form, eq. 3.5), but we project
  the result on ``[gamma_min, 1]`` instead of ``[0, 1]`` --- the floor is
  required to keep the stepsize-restricted condition beta_i <= gamma_i
  meaningful (see point 2 below) and is set to gamma_min = 0.05 by default;

* stepsize-restricted Polyak step --- ``alpha_i = beta_i (f(w_i) - target)
  / ||d_i||^2`` with ``beta_i = min(beta, gamma_i)``. The clip enforces
  beta_i <= gamma_i so that the per-step iterate-distance bound from
  Theorem 3.2 of d'Antonio-Frangioni applies at every iteration (report
  Theorem 3.1). Without the gamma_min floor the greedy choice would push
  gamma_i to 0 in monotone-descent regions, the clip would propagate to
  alpha_i, and the iterate would freeze; gamma_min keeps both quantities
  bounded away from 0 and the algorithm converges to gaps in the 1e-7..1e-8
  range on the convergence test (vs ~4e-4 for the gamma_min = 0 variant);

* target level — ``target = f_ref - delta`` is an adaptive estimate of f*;
  the gap ``delta`` is contracted by rho whenever progress stalls.

In addition to the travel-distance patience ``r > R`` from the source paper,
we run an iteration-count fallback (report §5.1): if the record value has
not improved for R_iter consecutive iterations we contract delta and reset
``d_{i-1} <- 0``, which forces gamma = 1 at the next step. The fallback is
a backstop against rare cases where the gamma_min-floored direction still
makes no progress (e.g. when target underestimates f^* across many steps).
"""

import time
import numpy as np

from .lasso_utils import f_lasso, subgradient_f


def _optimal_gamma(g, d_prev, gamma_min=0.0):
    """Closed-form deflection parameter (report eq. 3.5).

    Solves ``argmin_{gamma in [gamma_min, 1]} ||gamma g + (1-gamma) d_prev||^2``
    by expanding the parabola: the unconstrained minimiser is

        gamma* = (||d_prev||^2 - <g, d_prev>) / ||g - d_prev||^2,

    then projected onto ``[gamma_min, 1]``. The floor gamma_min > 0 is what
    makes the stepsize-restricted condition beta_i = min(beta, gamma_i) ≤ gamma_i
    operational on ELM LASSO: without it the greedy minimum would collapse
    to 0 in monotone-descent regions (consecutive subgradients align ⇒
    new-subgradient component dominates the parabola coefficients ⇒
    minimiser → 0), which would propagate to beta_i and alpha_i and freeze
    the iterate. With gamma_min > 0 the deflection step still favours the
    memory direction, but never enough to kill the new subgradient
    completely, and the per-iteration iterate-distance bound applies as
    proved in d'Antonio-Frangioni (Theorem 3.2 in the cited paper).

    Two corner cases need special handling:

    * d_prev = 0 (first step or right after the iteration-count fallback): the
      parabola degenerates to ``gamma^2 ||g||^2``, whose minimiser at gamma = 0
      would set d = 0 and stall the algorithm. We return 1 instead, matching
      the i = 0 convention used in Algorithm 2 of the report.
    * g = d_prev: the direction is invariant in gamma, so we pick gamma = 1.
    """
    d_sq = np.dot(d_prev, d_prev)
    if d_sq < 1e-30:
        return 1.0
    diff = g - d_prev
    diff_sq = np.dot(diff, diff)
    if diff_sq < 1e-30:
        return 1.0
    gamma_star = (d_sq - np.dot(g, d_prev)) / diff_sq
    return float(np.clip(gamma_star, gamma_min, 1.0))


def deflected_subgradient(X, y, lam, w0=None, i_max=5000, beta=1.0,
                          delta0=None, R=None, rho=0.95,
                          f_star=None, verbose=False, verbose_freq=500,
                          R_iter=None, gamma_min=0.05):
    """SGPTL applied to ELM LASSO; see the module docstring.

    Parameters
    ----------
    X, y         Design matrix (M, H) and target (M,).
    lam          L1 regularisation strength, > 0.
    w0           Initial iterate; defaults to the OLS solution (X^T X)^{-1} X^T y,
                 which is what IRLS uses as well.
    i_max        Iteration cap.
    beta         Polyak-step reference multiplier in (0, 1]; the actual
                 multiplier used at iteration i is beta_i = min(beta, gamma_i),
                 enforcing the stepsize-restricted condition beta_i <= gamma_i
                 of d'Antonio-Frangioni's Theorem 3.2 verbatim. The natural
                 choice is 1.
    delta0       Initial gap estimate; defaults to max(0.1 f(w_0), 1e-4).
    R            Travel-distance patience for the SGPTL contraction;
                 defaults to 10 sqrt(i_max).
    rho          Contraction factor for delta, in (0, 1).
    f_star       Reference optimum for gap tracking. If None, gaps stays empty.
    R_iter       Stall-window for the iteration-count fallback that resets
                 d_{i-1} <- 0 when f_bar fails to improve over R_iter
                 consecutive steps; defaults to max(i_max/100, 50). Set to a
                 very large value (e.g. i_max+1) to disable the fallback.
    gamma_min    Floor on the greedy deflection coefficient gamma_i: the
                 closed-form minimiser is projected on [gamma_min, 1] instead
                 of [0, 1]. The default 0.05 keeps the per-iteration bound
                 of Theorem 3.2 operational without giving up much of the
                 ||d_i||-minimising property of the greedy choice (see
                 module docstring and §3.5 of the report).

    Returns a dict with the best iterate w_best, the trajectories (f_vals,
    f_bar, gaps, times, delta_hist) and the iteration count.
    """
    _, H = X.shape

    if w0 is None:
        # Same OLS warm start as IRLS, reproduced here so the function stays
        # self-contained when called without sharing the precomputed A, b.
        from .linear_solvers import solve_spd
        try:
            w = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y)
        except Exception:
            w = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    if delta0 is None:
        delta0 = max(0.1 * f_curr, 1e-4)
    if R is None:
        R = 10.0 * np.sqrt(i_max)

    # Algorithm state.
    delta = delta0
    f_ref = f_curr           # checkpoint reference value, updated on improvement
    f_bar = f_curr           # running record min_{h <= i} f(w_h)
    w_best = w.copy()        # iterate that attained f_bar
    d_prev = np.zeros(H)     # d_{-1} = 0 by convention
    r = 0.0                  # accumulated travel since last delta-contraction

    # Iteration-count fallback (report §5.1): contract delta and reset d_prev
    # when f_bar has not improved for R_iter consecutive iterations. Setting
    # R_iter = inf disables the fallback (used in the warm-vs-cold diagnostic
    # of §5.3 to expose the unmasked deflection-collapse behaviour).
    if R_iter is None:
        R_iter = max(i_max // 100, 50)
    stalled_for = 0
    f_bar_marker = f_bar

    f_vals     = [f_curr]
    f_bar_list = [f_bar]
    gaps       = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times      = [0.0]
    delta_hist = [delta]
    gamma_hist = []   # per-iteration deflection coefficient gamma_i
    t0 = time.perf_counter()
    i = -1   # ensures n_iter = 0 when i_max = 0 and the loop never runs

    for i in range(i_max):
        # Iteration-count fallback: refresh the stall counter at every iteration,
        # including the ones that hit the early-continue branches below.
        if f_bar < f_bar_marker - 1e-14:
            stalled_for = 0
            f_bar_marker = f_bar
        else:
            stalled_for += 1
        if stalled_for >= R_iter:
            delta *= rho
            d_prev = np.zeros(H)   # _optimal_gamma will return 1 next step
            stalled_for = 0
            f_bar_marker = f_bar

        g = subgradient_f(X, y, w, lam)
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        gamma_hist.append(gamma)
        d = gamma * g + (1.0 - gamma) * d_prev

        d_sq = np.dot(d, d)
        if d_sq < 1e-30:
            # ||d|| has collapsed: either we are at a stationary point or the
            # last reset has not yet propagated. Break rather than divide by 0.
            break

        # Stepsize-restricted Polyak step (literal d'Antonio-Frangioni clip):
        # beta_i = min(beta, gamma_i) ≤ gamma_i so that the per-iteration
        # iterate-distance bound of Theorem 3.2 applies verbatim. The clip
        # is operational on ELM LASSO only because gamma_i is floored at
        # gamma_min > 0 in _optimal_gamma; without that floor the greedy
        # minimum collapses to 0 in monotone-descent regions and freezes
        # alpha_i. See module docstring and §3.5 of the report.
        beta_i = min(beta, gamma)
        target = f_ref - delta
        numerator = beta_i * (f_curr - target)

        if numerator <= 0.0:
            # f_curr already at or below the target — the gap estimate is too
            # aggressive. Contract delta and skip the step rather than make a
            # zero/backward move.
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        alpha = numerator / d_sq
        w_new = w - alpha * d

        # Defensive: very small ||d|| paired with a still-large numerator can
        # produce alpha large enough to overflow w_new. Reject the step rather
        # than poison the rest of the run.
        if not np.all(np.isfinite(w_new)):
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        f_new = f_lasso(X, y, w_new, lam)

        # Overshoot rescue. When delta sits well above f_ref - f^* the target
        # f_ref - delta is below f^*, the Polyak numerator stays at order
        # delta while ||d|| collapses near a stationary region, and alpha
        # blows up: a single iteration can send w to a region where f is
        # 50-70% above the record. Bound (3.14) still holds in iterate
        # distance but the f-trace shoots through the roof. When that
        # happens we discard w_new, snap back to w_best, contract delta and
        # zero d_prev so gamma = 1 next step. The convergence guarantee is
        # preserved (we only ever return w_best, and delta is contracted in
        # the same way the patience mechanism would have done it eventually).
        if f_new > 1.2 * f_bar and f_new > f_bar + delta:
            w = w_best.copy()
            f_curr = f_bar
            d_prev = np.zeros(H)
            delta *= rho
            delta_hist.append(delta)
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        if f_new < f_bar:
            f_bar = f_new
            w_best = w_new.copy()

        # Target-level bookkeeping (report §3.3, eq. on p.17):
        #   - significant progress -> move the checkpoint to the new record;
        #   - patience exhausted   -> contract delta;
        #   - otherwise            -> accumulate travel.
        if f_new <= f_ref - delta / 2.0:
            f_ref = f_bar
            r = 0.0
        elif r > R:
            delta *= rho
            r = 0.0
        else:
            r += alpha * np.sqrt(d_sq)

        w = w_new
        f_curr = f_new
        d_prev = d

        f_vals.append(f_curr)
        f_bar_list.append(f_bar)
        delta_hist.append(delta)
        times.append(time.perf_counter() - t0)
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
        'gamma_hist': gamma_hist,
        'times':      times,
        'n_iter':     i + 1,
        'delta_hist': delta_hist,
    }
