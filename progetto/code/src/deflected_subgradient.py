"""Algorithm A2: deflected subgradient with Polyak target level (SGPTL).

Iteration: g_i = subgradient of f at w_i; gamma_i minimises
||gamma g_i + (1-gamma) d_{i-1}||^2 on [gamma_min, 1]; d_i is the convex
combination; alpha_i is the stepsize-restricted Polyak step with target
level f_ref - delta; w_{i+1} = w_i - alpha_i d_i. f_ref, delta and the
travel-distance counter r are updated by the patience rules.
"""

import time
import numpy as np

from .lasso_utils import f_lasso, subgradient_f


def _optimal_gamma(g, d_prev, gamma_min=0.0):
    """Closed-form minimiser of ||gamma g + (1-gamma) d_prev||^2 on [gamma_min, 1]."""
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

    # Cold start by default: warm-starting from OLS places w_0 too close to w^*,
    # the Polyak numerator is dominated by delta_0, r never reaches R, and
    # delta is never contracted. Callers may still pass an explicit w0.
    if w0 is None:
        w = np.zeros(H)
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    if delta0 is None:
        delta0 = max(0.1 * f_curr, 1e-4)
    if R is None:
        R = 1.0

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
        g = subgradient_f(X, y, w, lam)
        # i == 0: d_{-1} = 0 by convention, so d_0 = g_0.
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        gamma_hist.append(gamma)
        d = gamma * g + (1.0 - gamma) * d_prev

        d_sq = d @ d
        if d_sq < 1e-30:
            break

        # beta_i = min(beta, gamma) enforces the stepsize-restricted rule beta_i <= gamma_i.
        beta_i = min(beta, gamma)
        target = f_ref - delta
        num = beta_i * (f_curr - target)

        # lambda_k < 0 in (3.5) of d'Antonio-Frangioni 2009: skip the step
        # and refresh f_ref (sufficient-descent condition met at w_i).
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

        alpha = num / d_sq
        w_new = w - alpha * d

        if not np.all(np.isfinite(w_new)):
            d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        f_new = f_lasso(X, y, w_new, lam)

        if f_new < f_bar:
            f_bar = f_new
            w_best = w_new.copy()

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

    # SGPTL is non-monotone: return the argmin iterate, not the last one.
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
