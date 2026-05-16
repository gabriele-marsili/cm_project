"""Algorithm A2: deflected subgradient with Polyak target level (SGPTL).

Iteration (cf. SGPTL pseudocode in the report, Algorithm 2, exactly as in
slide 14 of Lessons Optimization and Lemma 3.8 of d'Antonio-Frangioni 2009):
    g_i = subgradient of f at w_i   (min-norm element at w_i = 0)
    gamma_i = argmin_{gamma in [gamma_min, 1]} ||gamma g_i + (1 - gamma) d_{i-1}||^2
    d_i = gamma_i g_i + (1 - gamma_i) d_{i-1}
    beta_i = min(beta, gamma_i)                     # stepsize-restricted: beta_i <= gamma_i
    alpha_i = beta_i (f(w_i) - (f_ref - delta)) / ||d_i||^2     # Polyak with target level
    w_{i+1} = w_i - alpha_i d_i
    update f_ref, delta, r per the patience rules of slide 14

Parameters (all from the slide):
    beta      in (0, 2]   step modulation; we fix beta = 1
    delta0    > 0         initial target gap (default: 0.1 * f(w_0))
    R         > 0         travel-distance patience (default: 10*sqrt(i_max))
    rho       in (0, 1)   delta contraction factor (default: 0.95;
                          the report adopts 0.7 in all experiments)
    gamma_min in (0, 1]   deflection floor; required to make condition (3.5)
                          of d'Antonio-Frangioni 2009 (the hypothesis of
                          Theorem 3.1 in the report) hold with beta* = gamma_min.
                          gamma_min = 0 disables the floor (theory does NOT apply).
"""

import time
import numpy as np

from .lasso_utils import f_lasso, subgradient_f


def _optimal_gamma(g, d_prev, gamma_min=0.0):
    """Closed-form minimiser of ||gamma g + (1-gamma) d_prev||^2 on [gamma_min, 1].

    Setting d/d_gamma of the parabola to zero gives
        gamma* = (||d_prev||^2 - <g, d_prev>) / ||g - d_prev||^2
    which we then clip to [gamma_min, 1]. The two guards handle the degenerate
    cases where the parabola is flat: if d_prev = 0 (first iteration, since
    d_{-1} = 0 by convention) or if g == d_prev (consecutive aligned
    subgradients), gamma is irrelevant, and returning 1.0 keeps d_i = g_i.
    """
    d_sq = d_prev @ d_prev
    if d_sq < 1e-30:
        return 1.0
    diff = g - d_prev
    diff_sq = diff @ diff
    if diff_sq < 1e-30:
        return 1.0
    gamma_star = (d_sq - g @ d_prev) / diff_sq
    return float(np.clip(gamma_star, gamma_min, 1.0))


def deflected_subgradient(X, y, lam, w0=None, i_max=5000, beta=1.0,
                          delta0=None, R=None, rho=0.95,
                          f_star=None, verbose=False, verbose_freq=500,
                          gamma_min=0.05):
    _, H = X.shape

    # cold start by default: w_0 = 0. Theory-pure SGPTL is incompatible with
    # the OLS warm start used by IRLS - the warm start places w_0 very close
    # to w^*, the Polyak step numerator f(w_0) - (f_ref - delta) is dominated
    # by delta_0 rather than by the true gap, the travel-distance counter r
    # cannot accumulate to R, and delta is never contracted (we document the
    # phenomenon in Section 5.3 of the report). Callers can still pass an
    # explicit w0 (e.g., for the warm-vs-cold experiment of Section 5.X), but
    # the convergence guarantee of Theorem 3.1 is only verified for cold start
    # on this problem class.
    if w0 is None:
        w = np.zeros(H)
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    # default scales: delta0 = 0.1 f(w_0) is the report rule (no use of f^*);
    # R = 1.0 is a small absolute default, chosen so the patience threshold is
    # comparable to the step length scale on ELM LASSO with cold start (typical
    # step length 1e-3 with default delta_0 and gamma_min=0.05, so r reaches
    # R = 1 within O(1000) iterations of stalling - a reasonable patience cycle).
    # An R that scaled with sqrt(i_max) - as we had in an earlier version of
    # this code - made the threshold orders of magnitude larger than any
    # realistic accumulated travel: r could never reach R, the r > R branch
    # never fired, delta never contracted. R is a free parameter in
    # the theory of [d'Antonio-Frangioni 2009, Lemma 3.8] (only R > 0 is
    # required), so this is a calibration choice within the theoretical
    # framework, not a workaround.
    if delta0 is None:
        delta0 = max(0.1 * f_curr, 1e-4)
    if R is None:
        R = 1.0

    # state of the SGPTL loop (one-to-one with slide 14 / Lemma 3.8):
    #   delta         current target gap, monotonically non-increasing
    #   f_ref         reference value, only updated on "good improvement"
    #   f_bar         running record min_{h <= i} f(w_h), monotone non-increasing
    #   w_best        argmin iterate, the one we return
    #   d_prev        d_{i-1} for the deflection recursion (zero on first step)
    #   r             accumulator for the travel-distance patience rule
    delta = delta0
    f_ref = f_curr
    f_bar = f_curr
    w_best = w.copy()
    d_prev = np.zeros(H)
    r = 0.0

    f_vals     = [f_curr]
    f_bar_list = [f_bar]
    gaps       = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times      = [0.0]
    delta_hist = [delta]
    gamma_hist = []
    t0 = time.perf_counter()
    i = -1

    for i in range(i_max):
        # --- subgradient and deflected direction ---
        # subgradient_f returns the smooth gradient X^T (X w - y) plus
        # lam * sign(w), with sign(0) = 0 (any value in [-1, +1] is admissible
        # at indices with w_i = 0; s_i = 0 is the min-norm choice only when
        # |q_i| <= lam, the LASSO optimality condition for w_i = 0 being
        # stationary; see Section 3.5.1 of the report).
        g = subgradient_f(X, y, w, lam)
        # first iteration has no d_{-1}: set gamma = 1, so d_0 = g_0
        # (matches the convention d_{-1} = 0 in the pseudocode).
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        gamma_hist.append(gamma)
        d = gamma * g + (1.0 - gamma) * d_prev

        # --- numerical safeguard: direction collapsed ---
        # ||d_i||^2 ~ 0 means we are at (or numerically indistinguishable
        # from) a stationary point: the deflected direction has nothing to
        # contribute. The Polyak step is then ill-defined (division by zero)
        # and we exit the loop. This is a pure floating-point safeguard.
        d_sq = d @ d
        if d_sq < 1e-30:
            break

        # --- stepsize-restricted Polyak step ---
        # beta_i = min(beta, gamma) enforces beta_i <= gamma_i, the
        # stepsize-restricted rule of the course slide (Set 5, p. 15:
        # "deflection-first: Polyak alpha = beta(f - f_*) / ||d||^2,
        # beta <= gamma"). It is the hypothesis under which the deflected
        # convergence in Theorem 3.1 of the report carries the same rate
        # as the plain Polyak step.
        beta_i = min(beta, gamma)
        target = f_ref - delta
        num = beta_i * (f_curr - target)

        # --- condition (3.5) of d'Antonio-Frangioni 2009: lambda_k < 0 ---
        # f(w_i) - (f_ref - delta) <= 0 means f_curr <= f_ref - delta, so the
        # current iterate is below the target level. By (3.5), alpha_k = 0:
        # skip the step. Moreover f_curr <= f_ref - delta <= f_ref - delta/2
        # is the sufficient-descent condition of Lemma 3.8 evaluated at the
        # current iterate, so we trigger the f_ref update (theory-aligned
        # response, not a delta contraction).
        if num <= 0.0:
            f_ref = f_bar
            r = 0.0
            d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        # --- the actual Polyak step ---
        alpha = num / d_sq
        w_new = w - alpha * d

        # --- numerical safeguard: NaN/Inf in w_new ---
        # When ||d||^2 is tiny but above the 1e-30 cutoff, alpha = num / ||d||^2
        # can overflow to +inf and w_new becomes non-finite. Propagating a NaN
        # iterate would poison every subsequent operation (subgradient, norm,
        # comparisons). We skip the iteration; no other state is touched (delta
        # is NOT contracted: the contraction would be a workaround, not a
        # theory-prescribed reaction).
        if not np.all(np.isfinite(w_new)):
            d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        f_new = f_lasso(X, y, w_new, lam)

        # --- record update: f_bar = min_{h <= i+1} f(w_h) ---
        if f_new < f_bar:
            f_bar = f_new
            w_best = w_new.copy()

        # --- target-level update (slide pseudocode) ---
        # "Good improvement": f dropped below the half-target, refresh f_ref
        # and reset the travel-distance counter.
        # "Too many steps without improvement" (r > R): contract delta.
        # Otherwise accumulate the travel distance alpha ||d||.
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
            gs = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  DSM iter {i+1:6d}:  f={f_curr:.6e}  f_bar={f_bar:.6e}"
                  f"  delta={delta:.2e}{gs}")

    # we return w_best (the argmin iterate), not the last w: SGPTL is
    # non-monotone in f, and the convergence guarantee is on f_bar = min f,
    # not on f(w_last).
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
