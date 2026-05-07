"""
Diagnostic experiment for SGPTL: greedy gamma collapse and the role of
``beta_i = min(beta, gamma)`` clipping (report §5.1).

The greedy choice of gamma in eq. (3.5) drives gamma_i towards 0 in
monotone-descent regions: consecutive subgradients align, the parabola in
gamma is minimised by the memory-only direction d_{i-1}, so the new-
subgradient component vanishes from d_i. This is structural and happens on
both warm and cold starts.

The literal stepsize-restricted reading of d'Antonio-Frangioni's per-step
bound — ``beta_i = min(beta, gamma_i)`` — couples the stepsize to gamma:
when gamma collapses so does alpha, the iterate stops moving, and the
algorithm stalls at a record gap of ~1e-2 regardless of starting point.
Holding ``beta = 1`` fixed (our implementation, see §5.1) decouples alpha
from gamma and lets the target-level mechanism do the work.

Output: a 1x2 figure with one panel per beta variant. Each panel overlays
the warm and cold record-gap traces, with the deflection-collapse regime
annotated. The literal-clip panel plateaus at ~1e-2; the beta-fixed panel
descends two to three orders of magnitude further on the same budget.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from src.lasso_utils import f_lasso, subgradient_f
from _plot_style import (apply_style, style_axes,
                         COLOR_DSM, COLOR_FCUR, COLOR_AUX, SIZE_DOUBLE)
apply_style()


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05
H, M   = 100, 300
I_MAX  = 4000

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _dsm_clipped(X, y, lam, w0, i_max, beta=1.0, delta0=None, rho=0.9, f_star=None):
    """SGPTL with the LITERAL stepsize-restricted reading: beta_i = min(beta, gamma_i).

    Kept here to expose the convergence catastrophe the literal clip causes
    on ELM~LASSO. Our production implementation in
    src/deflected_subgradient.py uses beta_i = beta.
    """
    M_, H_ = X.shape
    w = w0.copy(); f_curr = f_lasso(X, y, w, lam)
    if delta0 is None: delta0 = max(0.1 * f_curr, 1e-4)
    R = 10.0 * np.sqrt(i_max)
    R_iter = max(i_max // 100, 50)
    delta = delta0; f_ref = f_curr; f_bar = f_curr; w_best = w.copy()
    d_prev = np.zeros(H_); r_travel = 0.0; stalled = 0; f_bar_marker = f_bar
    f_vals = [f_curr]; f_bar_list = [f_bar]; gamma_hist = []
    gaps = [max(0.0, f_bar - f_star)] if f_star is not None else []
    for i in range(i_max):
        if f_bar < f_bar_marker - 1e-14:
            stalled = 0; f_bar_marker = f_bar
        else:
            stalled += 1
        if stalled >= R_iter:
            delta *= rho; d_prev = np.zeros(H_)
            stalled = 0; f_bar_marker = f_bar
        g = subgradient_f(X, y, w, lam)
        if i == 0 or np.dot(d_prev, d_prev) < 1e-30:
            gamma = 1.0
        else:
            diff = g - d_prev; diff_sq = np.dot(diff, diff)
            if diff_sq < 1e-30:
                gamma = 1.0
            else:
                gs = (np.dot(d_prev, d_prev) - np.dot(g, d_prev)) / diff_sq
                gamma = float(np.clip(gs, 0.0, 1.0))
        gamma_hist.append(gamma)
        d = gamma * g + (1.0 - gamma) * d_prev
        d_sq = float(np.dot(d, d))
        if d_sq < 1e-30: break
        beta_i = min(beta, gamma)            # <-- the clip
        target = f_ref - delta
        numerator = beta_i * (f_curr - target)
        if numerator <= 0.0:
            delta *= rho; d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None: gaps.append(max(0.0, f_bar - f_star))
            continue
        alpha = numerator / d_sq
        w_new = w - alpha * d
        if not np.all(np.isfinite(w_new)):
            delta *= rho; d_prev = d
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None: gaps.append(max(0.0, f_bar - f_star))
            continue
        f_new = f_lasso(X, y, w_new, lam)
        if f_new > 1.2 * f_bar and f_new > f_bar + delta:
            w = w_best.copy(); f_curr = f_bar; d_prev = np.zeros(H_); delta *= rho
            f_vals.append(f_curr); f_bar_list.append(f_bar)
            if f_star is not None: gaps.append(max(0.0, f_bar - f_star))
            continue
        if f_new < f_bar:
            f_bar = f_new; w_best = w_new.copy()
        if f_new <= f_ref - delta / 2.0:
            f_ref = f_bar; r_travel = 0.0
        elif r_travel > R:
            delta *= rho; r_travel = 0.0
        else:
            r_travel += alpha * np.sqrt(d_sq)
        w = w_new; f_curr = f_new; d_prev = d
        f_vals.append(f_curr); f_bar_list.append(f_bar)
        if f_star is not None: gaps.append(max(0.0, f_bar - f_star))
    return {"f_vals": f_vals, "f_bar": f_bar_list, "gamma_hist": gamma_hist,
            "gaps": gaps}


def run() -> None:
    print("=" * 60)
    print("SGPTL diagnostic: clipped vs fixed beta on warm / cold start")
    print("=" * 60)

    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    w_ols  = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    w_cold = np.zeros(H)

    # 4 runs: {fixed-beta, clipped} x {warm, cold}.
    runs = {
        ("fixed",   "warm"): deflected_subgradient(
            X, y, LAMBDA, w0=w_ols,  i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, f_star=f_star),
        ("fixed",   "cold"): deflected_subgradient(
            X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, f_star=f_star),
        ("clipped", "warm"): _dsm_clipped(
            X, y, LAMBDA, w0=w_ols,  i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, f_star=f_star),
        ("clipped", "cold"): _dsm_clipped(
            X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, f_star=f_star),
    }

    print(f"\n{'config':>22}  {'final gap':>11}  {'frac gamma<0.1':>14}")
    for key, res in runs.items():
        g = np.asarray(res["gamma_hist"], dtype=float)
        frac = float(np.mean(g < 0.1)) if g.size else float("nan")
        print(f"  {key[0]:>10}, {key[1]:>5}  {res['gaps'][-1]:>11.3e}  "
              f"{frac:>14.2%}")

    # 1x2 figure: one panel per beta variant. Each panel overlays warm-
    # and cold-start record-gap traces, with annotated final values.
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE, sharey=True)

    def _gap_panel(ax, key_warm, key_cold, title):
        gw = np.maximum(np.asarray(runs[key_warm]["gaps"], dtype=float), 1e-16)
        gc = np.maximum(np.asarray(runs[key_cold]["gaps"], dtype=float), 1e-16)
        ax.loglog(np.arange(1, len(gw) + 1), gw,
                  color=COLOR_DSM, linewidth=2.0, label="warm start (OLS)")
        ax.loglog(np.arange(1, len(gc) + 1), gc,
                  color=COLOR_FCUR, linewidth=2.0,
                  label=r"cold start ($w_0=0$)")
        # Annotate the final record gap on each trace.
        ax.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM, zorder=5)
        ax.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)
        ax.annotate(f"{gw[-1]:.1e}",
                    xy=(len(gw), gw[-1]),
                    xytext=(8, -2), textcoords="offset points",
                    fontsize=10, color=COLOR_DSM, ha="left", va="center")
        ax.annotate(f"{gc[-1]:.1e}",
                    xy=(len(gc), gc[-1]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=10, color=COLOR_FCUR, ha="left", va="center")
        ax.set_xlabel("Iteration  (log scale)")
        ax.set_ylabel(r"$\bar{f}^{i} - f^{*}$  (log scale)")
        ax.set_title(title)
        ax.legend(loc="lower left")
        style_axes(ax)

    _gap_panel(axes[0], ("fixed", "warm"), ("fixed", "cold"),
               r"Our implementation: $\beta = 1$ fixed")
    _gap_panel(axes[1], ("clipped", "warm"), ("clipped", "cold"),
               r"Literal clip: $\beta_{i} = \min(\beta, \gamma_{i})$")

    fig.suptitle(rf"SGPTL record gap on $H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$, "
                 rf"$i_{{\max}}={I_MAX}$"
                 "  —  the clipped variant stalls two decades above ours",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "warm_vs_cold.pdf")
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
