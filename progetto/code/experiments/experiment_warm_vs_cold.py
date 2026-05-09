"""SGPTL: gamma_min = 0 vs 0.05, on warm (OLS) and cold (w_0=0) starts."""

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
from _plot_style import (apply_style, style_axes,
                         COLOR_DSM, COLOR_FCUR, SIZE_DOUBLE)
apply_style()


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05
H, M   = 100, 300
I_MAX  = 4000

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run() -> None:
    print("=" * 60)
    print("SGPTL diagnostic: gamma_min = 0 vs gamma_min = 0.05")
    print("=" * 60)

    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    w_ols  = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    w_cold = np.zeros(H)

    runs = {
        ("floored",   "warm"): deflected_subgradient(
            X, y, LAMBDA, w0=w_ols,  i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, gamma_min=0.05, f_star=f_star),
        ("floored",   "cold"): deflected_subgradient(
            X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, gamma_min=0.05, f_star=f_star),
        ("unfloored", "warm"): deflected_subgradient(
            X, y, LAMBDA, w0=w_ols,  i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, gamma_min=0.0,  f_star=f_star),
        ("unfloored", "cold"): deflected_subgradient(
            X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_star, rho=0.9, gamma_min=0.0,  f_star=f_star),
    }

    print(f"\n{'config':>22}  {'final gap':>11}  {'frac gamma<0.06':>15}")
    for key, res in runs.items():
        g = np.asarray(res["gamma_hist"], dtype=float)
        frac = float(np.mean(g < 0.06)) if g.size else float("nan")
        print(f"  {key[0]:>10}, {key[1]:>5}  {res['gaps'][-1]:>11.3e}  "
              f"{frac:>15.2%}")

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE, sharey=True)

    def _gap_panel(ax, key_warm, key_cold, title):
        gw = np.maximum(np.asarray(runs[key_warm]["gaps"], dtype=float), 1e-16)
        gc = np.maximum(np.asarray(runs[key_cold]["gaps"], dtype=float), 1e-16)
        ax.loglog(np.arange(1, len(gw) + 1), gw,
                  color=COLOR_DSM, linewidth=2.0, label="warm start (OLS)")
        ax.loglog(np.arange(1, len(gc) + 1), gc,
                  color=COLOR_FCUR, linewidth=2.0,
                  label=r"cold start ($w_0=0$)")
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

    _gap_panel(axes[0], ("floored", "warm"), ("floored", "cold"),
               r"With floor: $\gamma_{\min} = 0.05$")
    _gap_panel(axes[1], ("unfloored", "warm"), ("unfloored", "cold"),
               r"Without floor: $\gamma_{\min} = 0$ (literal)")

    fig.suptitle(rf"SGPTL on $H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$, "
                 rf"$i_{{\max}}={I_MAX}$"
                 "  --- the floor decides whether the algorithm makes progress",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "warm_vs_cold.pdf")
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
