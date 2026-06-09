"""Algorithm A2 -- deflected subgradient with a Polyak target level (SGPTL).

One iteration:
    g_i = subgradient of f at w_i
    gamma_i (closed form) = argmin over [gamma_min, 1] of ||gamma g_i + (1-gamma) d_{i-1}||^2
    d_i (deflected direction) = gamma_i g_i + (1-gamma_i) d_{i-1}

    Polyak step toward the target level f_ref - delta, capped by beta_i
    alpha_i = w_{i+1}  w_i - alpha_i d_i

f_ref, delta and the travel counter r move on two patience rules: sufficient
descent (target reached -> reset f_ref) and travelled-too-far (r > R -> shrink delta by rho).
"""

from typing import Optional

import time
import warnings

import numpy as np

from .lasso_utils import f_lasso, subgradient_f

# squared norms below this are taken as zero: d_i collapsed to 0, or the degenerate argmin g_i == d_{i-1}
_NORM_FLOOR: float = 1e-30


def _optimal_gamma(
    g: np.ndarray,
    d_prev: np.ndarray,
    gamma_min: float = 0.05,
) -> float:
    """argmin over [gamma_min, 1] of ||gamma g + (1-gamma) d_prev||^2

    The objective is a parabola in gamma (its vertex is gamma_star below).
    The two degenerate cases (d_prev = 0, or g = d_prev) have no interior vertex
        => fall back to gamma = 1 -> d collapses to g
    """
    d_sq = d_prev @ d_prev
    if d_sq < _NORM_FLOOR:
        return 1.0
    diff = g - d_prev
    diff_sq = diff @ diff
    if diff_sq < _NORM_FLOOR:
        return 1.0
    gamma_star = (d_sq - g @ d_prev) / diff_sq
    return float(np.clip(gamma_star, gamma_min, 1.0))


def deflected_subgradient(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    w0: Optional[np.ndarray] = None,
    i_max: int = 5000,
    beta: float = 1.0,
    delta0: Optional[float] = None,
    R: Optional[float] = None,
    rho: float = 0.7,
    f_star: Optional[float] = None,
    verbose: bool = False,
    verbose_freq: int = 500,
    gamma_min: float = 0.05,
) -> dict:
    """SGPTL for min 1/2||Xw - y||^2 + lam||w||_1.

    Args:
        w0:        start point, default cold w_0 = 0.
        beta:      Polyak cap, the step uses beta_i = min(beta, gamma_i).
        delta0:    initial target margin, default 0.1 f(w_0).
        R:         travel-distance patience, default 1.0.
        rho:       delta shrink factor once r > R, in (0,1).
        gamma_min: lower clip on the deflection.
        f_star:    if set, gap f_bar_i - f* is logged in result['gaps'].

    Returns dict: w (best iterate seen), f_vals, f_bar, gaps, gamma_hist,
    skip_hist, times, n_iter, delta_hist. The per-step histories (f_vals, f_bar,
    gaps, times, delta_hist) start with the initial point and append one entry
    per loop pass, gamma_hist/skip_hist hold one entry per pass only.
    """
    _, H = X.shape

    if w0 is None:
        w = np.zeros(H)
    else:
        w = w0.copy()

    f_curr = f_lasso(X, y, w, lam)

    if delta0 is None:
        delta0 = 0.1 * f_curr
    if R is None:
        R = 1.0

    delta = delta0
    f_ref = f_curr
    f_bar = f_curr
    w_best = w.copy()
    d_prev = np.zeros(H)
    r = 0.0

    f_vals = [f_curr]
    f_bar_list = [f_bar]
    gaps = [max(0.0, f_bar - f_star)] if f_star is not None else []
    times = [0.0]
    delta_hist = [delta]
    gamma_hist: list = []
    skip_hist: list = []
    t0 = time.perf_counter()
    i = -1

    for i in range(i_max):
        g = subgradient_f(X, y, w, lam)
        # first pass has no previous direction (d_{-1} = 0) -> d_0 = g_0
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        d = gamma * g + (1.0 - gamma) * d_prev

        d_sq = d @ d
        if d_sq < _NORM_FLOOR:
            break                       # direction died, nowhere left to go

        gamma_hist.append(gamma)

        beta_i = min(beta, gamma)       # stepsize-restricted Polyak: beta_i <= gamma
        target = f_ref - delta
        num = beta_i * (f_curr - target)

        # num <= 0 means f_i is already at/under the target, so the descent test
        # has effectively fired here: take no step, just refresh f_ref and log
        if num <= 0.0:
            f_ref = f_bar
            r = 0.0
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            skip_hist.append(1)
            continue

        alpha = num / d_sq
        w_new = w - alpha * d

        if not np.all(np.isfinite(w_new)):
            # overflow in the step -> freeze w, log it, and keep going
            warnings.warn(
                f"SGPTL: non-finite iterate at i={i}, freezing w, "
                f"alpha={alpha:.3e}, |d|={np.sqrt(d_sq):.3e}",
                RuntimeWarning,
                stacklevel=2,
            )
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            skip_hist.append(0)         # numerical event, not a descent skip
            continue

        f_new = f_lasso(X, y, w_new, lam)

        if f_new < f_bar:
            f_bar = f_new
            w_best = w_new.copy()

        # patience update: hit half the margin -> reset, else if we have
        # travelled past R -> shrink delta, else keep accumulating travel
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
        skip_hist.append(0)

        if verbose and (i + 1) % verbose_freq == 0:
            gs = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(
                f"  DSM iter {i+1:6d}:  f={f_curr:.6e}  f_bar={f_bar:.6e}"
                f"  delta={delta:.2e}{gs}"
            )

    # the method is non-monotone, so hand back the best point, not the last one
    return {
        "w": w_best,
        "f_vals": f_vals,
        "f_bar": f_bar_list,
        "gaps": gaps,
        "gamma_hist": gamma_hist,
        "skip_hist": skip_hist,
        "times": times,
        "n_iter": i + 1,
        "delta_hist": delta_hist,
    }
