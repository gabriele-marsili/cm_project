"""Algorithm A2: deflected subgradient with Polyak target level (SGPTL).

Iteration:
- g_i: subgradient of f at w_i
- gamma_i: closed-form minimiser of ||gamma*g_i + (1-gamma)*d_{i-1}||^2 on [gamma_min, 1], computed by _optimal_gamma
- d_i: gamma_i*g_i + (1-gamma_i)*d_{i-1}
- alpha_i: stepsize-restricted Polyak step with target level f_ref - delta
- w_{i+1} = w_i - alpha_i * d_i
- f_ref, delta and the travel-distance counter r are updated by the patience rules (sufficient-descent and travel-distance triggers).
"""

from typing import Optional

import time
import warnings

import numpy as np

from .lasso_utils import f_lasso, subgradient_f

# Squared norms below this are treated as zero: d_i collapsed to 0, or the degenerate argmin case g_i == d_{i-1}.
_NORM_FLOOR: float = 1e-30


def _optimal_gamma(
    g: np.ndarray,
    d_prev: np.ndarray,
    gamma_min: float = 0.05,
) -> float:
    """Closed-form minimiser of ||gamma*g + (1-gamma)*d_prev||^2 on [gamma_min, 1].

    Falls back to gamma=1 in the two degenerate cases d_prev=0 and g=d_prev, where the parabola has no interior vertex.
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
    """Deflected subgradient (SGPTL) for LASSO.

    Args:
        w0: initial iterate. Default cold start w_0 = 0.
        i_max: max iteration count.
        beta: Polyak coefficient, beta_i = min(beta, gamma_i) at each step.
        delta0: initial target margin, default 0.1*f(w_0).
        R: travel-distance patience threshold, default 1.0.
        rho: contraction factor applied to delta when r > R, in (0,1).
        gamma_min: lower clip for the deflection, gamma_i in [gamma_min, 1].
        f_star: if given, gaps f_bar_i - f* are stored in result['gaps'].

    Returns dict with keys: w (argmin iterate), f_vals, f_bar, gaps, gamma_hist, skip_hist, times, n_iter, delta_hist. The history lists (f_vals, f_bar, gaps, times, delta_hist) carry the initial point plus one entry per completed iteration; skip_hist and gamma_hist carry one entry per completed iteration. n_iter is the loop count reached
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
        # i == 0: d_{-1} = 0 by convention, so d_0 = g_0.
        gamma = 1.0 if i == 0 else _optimal_gamma(g, d_prev, gamma_min=gamma_min)
        d = gamma * g + (1.0 - gamma) * d_prev

        d_sq = d @ d
        if d_sq < _NORM_FLOOR:
            # deflection collapsed to zero. no further progress is possible.
            break

        gamma_hist.append(gamma)

        # stepsize-restricted rule: beta_i = min(beta, gamma) <= gamma
        beta_i = min(beta, gamma)
        target = f_ref - delta
        num = beta_i * (f_curr - target)

        # Safeguard: num <= 0 implies f_i <= f_ref - delta, so the sufficient-descent test fires at w_i already. set alpha_i = 0 (no move) and refresh f_ref
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
            warnings.warn(f"SGPTL: non-finite iterate at i={i}, freezing w. alpha={alpha:.3e}, |d|={np.sqrt(d_sq):.3e}", RuntimeWarning, stacklevel=2)
            d_prev = d
            f_vals.append(f_curr)
            f_bar_list.append(f_bar)
            delta_hist.append(delta)
            if f_star is not None:
                gaps.append(max(0.0, f_bar - f_star))
            times.append(time.perf_counter() - t0)
            skip_hist.append(0)  # not a sufficient-descent skip, numerical event
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
        skip_hist.append(0)

        if verbose and (i + 1) % verbose_freq == 0:
            gs = f"  gap={gaps[-1]:.3e}" if f_star is not None else ""
            print(f"  DSM iter {i+1:6d}:  f={f_curr:.6e}  f_bar={f_bar:.6e}  delta={delta:.2e}{gs}")

    # SGPTL is non-monotone: return the argmin iterate, not the last one
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
