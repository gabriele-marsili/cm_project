""" Experiment warm start (using OLS solution) vs cold start (w = 0) """
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

from src import deflected_subgradient, make_lasso_problem
from src.lasso_utils import f_lasso
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
    print("SGPTL convergence analysis: warm start (OLS) vs cold start (w = 0)")
    print("=" * 60)
    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    w_cold = np.zeros(H)

    
    # keep track of w_ols computing time (f* here is used only to log the optimality gap for the convergence plot)
    t0 = time.time()
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    delta0_warm = 0.1 * f_lasso(X, y, w_ols, LAMBDA)
    sol_ols = deflected_subgradient(X, y, LAMBDA, w0=w_ols, i_max=I_MAX, beta=1.0, delta0=delta0_warm, rho=0.9, gamma_min=0.05, f_star=f_star)
    time_ols = time.time() - t0

    # time for cold start time
    t0 = time.time()
    delta0_cold = 0.1 * f_lasso(X, y, w_cold, LAMBDA)
    sol_cold = deflected_subgradient(X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0, delta0=delta0_cold, rho=0.9, gamma_min=0.05, f_star=f_star)
    time_cold = time.time() - t0

    print(f"Warm start execution time: {time_ols:.4f} s")
    print(f"Cold start execution time: {time_cold:.4f} s")

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    gw = np.maximum(np.asarray(sol_ols["gaps"], dtype=float), 1e-16)
    gc = np.maximum(np.asarray(sol_cold["gaps"], dtype=float), 1e-16)

    # Panel 1: Error vs Number of Iterations (Linear X-Axis)
    ax1 = axes[0]
    ax1.semilogy(np.arange(1, len(gw) + 1), gw,
                 color=COLOR_DSM, linewidth=2.0, label=f"warm start (OLS) [{time_ols:.2f}s]")
    ax1.semilogy(np.arange(1, len(gc) + 1), gc,
                 color=COLOR_FCUR, linewidth=2.0, label=rf"cold start ($w_0=0$) [{time_cold:.2f}s]")
    
    ax1.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM, zorder=5)
    ax1.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)
    
    ax1.set_xlabel("Iteration (linear scale)")
    ax1.set_ylabel(r"$\bar{f}^{i} - f^{*}$ (log scale)")
    ax1.set_title("Convergence vs Iterations")
    ax1.legend(loc="upper right")
    style_axes(ax1)

    # Panel 2: Error vs Execution Time (Linear X-Axis)
    ax2 = axes[1]
    
    # Approximate time per iteration linearly across the execution time
    time_array_w = np.linspace(0, time_ols, len(gw))
    time_array_c = np.linspace(0, time_cold, len(gc))

    ax2.semilogy(time_array_w, gw, color=COLOR_DSM, linewidth=2.0, label="warm start (OLS)")
    ax2.semilogy(time_array_c, gc, color=COLOR_FCUR, linewidth=2.0, label=r"cold start ($w_0=0$)")
    
    ax2.scatter([time_array_w[-1]], [gw[-1]], s=40, color=COLOR_DSM, zorder=5)
    ax2.scatter([time_array_c[-1]], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)
    
    ax2.set_xlabel("Time [seconds] (linear scale)")
    ax2.set_title("Convergence vs Time")
    ax2.legend(loc="upper right")
    style_axes(ax2)

    # Finalize Figure
    fig.suptitle(rf"SGPTL on $H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$, "
                 rf"$i_{{\max}}={I_MAX}$",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sgptl_time_warm_vs_cold.pdf")
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
