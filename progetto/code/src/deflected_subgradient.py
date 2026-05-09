"""Algorithm A2: deflected subgradient with Polyak target level (SGPTL)."""

import time
import numpy as np

from .lasso_utils import f_lasso, subgradient_f


def _optimal_gamma(g, d_prev, gamma_min=0.0):
    """argmin_{gamma in [gamma_min, 1]} ||gamma g + (1-gamma) d_prev||^2."""
    d_sq = d_prev @ d_prev
    if d_sq < 1e-30:
        return 1.0   # d_prev = 0: return 1 to avoid d = 0
    diff = g - d_prev
    diff_sq = diff @ diff
    if diff_sq < 1e-30:
        return 1.0   # g == d_prev: gamma is irrelevant
    gamma_star = (d_sq - g @ d_prev) / diff_sq
    return float(np.clip(gamma_star, gamma_min, 1.0))


def deflected_subgradient(X, y, lam, w0=None, i_max=5000, beta=1.0,
                          delta0=None, R=None, rho=0.95,
                          f_star=None, verbose=False, verbose_freq=500,
                          R_iter=None, gamma_min=0.05):
    _, H = X.shape

    if w0 is None:
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
    if R_iter is None:
        R_iter = max(i_max // 100, 50)

    delta = delta0
    f_ref = f_curr
    f_bar = f_curr
    w_best = w.copy()
    d_prev = np.zeros(H)
    r = 0.0
    stalled_for = 0
    f_bar_marker = f_bar

    f_vals     = [f_curr]
    f_bar_list = [f_bar]
    gaps       = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times      = [0.0]
    delta_hist = [delta]
    gamma_hist = []
    t0 = time.perf_counter()
    i = -1

    for i in range(i_max):
        # iteration-count fallback: if f_bar has not improved in R_iter steps,
        # contract delta and reset d_prev (forces gamma = 1 next step).
        if f_bar < f_bar_marker - 1e-14:
            stalled_for = 0
            f_bar_marker = f_bar
        else:
            stalled_for += 1
        if stalled_for >= R_iter:
            delta *= rho
            d_prev = np.zeros(H)
            stalled_for = 0
            f_bar_marker = f_bar

        g = subgradient_f(X, y, w, lam)
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        gamma_hist.append(gamma)
        d = gamma * g + (1.0 - gamma) * d_prev

        d_sq = d @ d
        if d_sq < 1e-30:
            break   # direction collapsed: stationary or fresh reset

        beta_i = min(beta, gamma)
        target = f_ref - delta
        num = beta_i * (f_curr - target)

        if num <= 0.0:
            # target too aggressive: skip and contract delta
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        alpha = num / d_sq
        w_new = w - alpha * d

        if not np.all(np.isfinite(w_new)):
            # alpha blew up: discard, contract, retry
            delta *= rho
            delta_hist.append(delta)
            d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

        f_new = f_lasso(X, y, w_new, lam)

        # overshoot: when delta sits well above f_ref - f^* the step can throw f
        # well above the running record. Snap back to w_best and contract delta.
        if f_new > 1.2 * f_bar and f_new > f_bar + delta:
            w = w_best.copy()
            f_curr = f_bar
            d_prev = np.zeros(H)
            delta *= rho
            delta_hist.append(delta)
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            continue

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
