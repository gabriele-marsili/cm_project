""" Experiment IRLS: warm start (using OLS) vs cold start (w = 0) on SYNTHETIC DATA """
import os
import sys
import warnings
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.irls import irls
from src.data_generation import make_lasso_problem
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_DSM, COLOR_FCUR, SIZE_DOUBLE)
apply_style()


SEED      = 42
LAMBDA    = 0.10
NOISE     = 0.05
H, M      = 100, 300
IRLS_KMAX = 2000
EPS_THR   = 1e-8

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def run() -> None:
    print("=" * 60)
    print("IRLS convergence analysis: warm start (OLS) vs cold start (w = 0)")
    print("=" * 60)
    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    # ==========================================
    # WARM START (OLS)
    # ==========================================
    t0 = time.time()
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    sol_ols = irls(
        X, y, LAMBDA, eps_thr=EPS_THR, k_max=IRLS_KMAX,
        solver='cholesky', w0=w_ols, f_star=f_star
    )
    time_ols = time.time() - t0

    # ==========================================
    # COLD START (w = 0)
    # ==========================================
    t0 = time.time()
    w_cold = np.zeros(H)
    sol_cold = irls(
        X, y, LAMBDA, eps_thr=EPS_THR, k_max=IRLS_KMAX,
        solver='cholesky', w0=w_cold, f_star=f_star
    )
    time_cold = time.time() - t0

    print(f"Warm start execution time: {time_ols:.4f} s")
    print(f"Cold start execution time: {time_cold:.4f} s")

    fw = np.asarray(sol_ols["f_vals"], dtype=float)
    fc = np.asarray(sol_cold["f_vals"], dtype=float)
    f_min = min(fw.min(), fc.min(), f_star)
    floor = 1e-12
    gw = np.maximum(fw - f_min, floor)
    gc = np.maximum(fc - f_min, floor)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(np.arange(1, len(gw) + 1), gw,
              color=COLOR_DSM, linewidth=2.0,
              label="Warm Start (OLS)")
    ax.loglog(np.arange(1, len(gc) + 1), gc,
              color=COLOR_FCUR, linewidth=2.0,
              label=r"Cold Start ($w_0 = 0$)")
    ax.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM, zorder=5)
    ax.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)
    ax.set_xlabel(r"Iteration $k$  (log scale)")
    ax.set_ylabel(r"$f(\mathbf{w}_{k}) - f_{\min}$  (log scale)")
    ax.set_title(rf"IRLS on synthetic ($H={H}$, $M={M}$, $\lambda={LAMBDA}$)")
    ax.legend(loc="lower left")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "irls_synthetic_warm_vs_cold.pdf")
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()