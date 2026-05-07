"""
experiment_comparison.py
------------------------
Head-to-head IRLS vs SGPTL on a moderate problem. Reports iterations
and CPU time required to reach a target accuracy and produces a
side-by-side semilog comparison figure for Chapter 5.
"""

import csv
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

from src import irls, deflected_subgradient, make_lasso_problem
from src.lasso_utils import f_lasso
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM, COLOR_FCUR
apply_style()


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def iters_to_reach(gaps, threshold):
    for i, g in enumerate(gaps):
        if g <= threshold:
            return i
    return None


def time_to_reach(gaps, times, threshold):
    for g, t in zip(gaps, times):
        if g <= threshold:
            return t
    return None


def run() -> None:
    print("=" * 60)
    print("IRLS vs SGPTL comparison")
    print("=" * 60)

    H, M = 50, 200
    X, y, _, f_star, w_star = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    # OLS warm start shared by both algorithms (report §5.2)
    from src.linear_solvers import solve_spd
    A = X.T @ X
    b = X.T @ y
    w_ols = solve_spd(A + 1e-12 * np.eye(H), b, method="cholesky")

    res_irls = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=1e-12,
                    k_max=300, solver="cholesky",
                    w0=w_ols, f_star=f_star)
    res_dsm  = deflected_subgradient(X, y, LAMBDA, w0=w_ols, i_max=30000,
                                     beta=1.0, delta0=0.1 * f_star, rho=0.9,
                                     f_star=f_star)

    # -------- table --------
    eps_grid = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
    rows = []
    print(f"\n{'eps':>6}  {'IRLS iter':>10}  {'IRLS time':>10}  "
          f"{'SGPTL iter':>11}  {'SGPTL time':>11}")
    for eps in eps_grid:
        ki = iters_to_reach(res_irls["gaps"], eps)
        ti = time_to_reach(res_irls["gaps"], res_irls["times"], eps)
        kd = iters_to_reach(res_dsm["gaps"], eps)
        td = time_to_reach(res_dsm["gaps"], res_dsm["times"], eps)
        rows.append([eps, ki, ti, kd, td])
        ti_s = f"{ti:.2e}" if ti is not None else "  ---"
        td_s = f"{td:.2e}" if td is not None else "  ---"
        ki_s = str(ki) if ki is not None else "---"
        kd_s = str(kd) if kd is not None else "---"
        print(f"{eps:>6.0e}  {ki_s:>10}  {ti_s:>10}  {kd_s:>11}  {td_s:>11}")

    tab_path = os.path.join(TAB_DIR, "comparison_table.csv")
    with open(tab_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epsilon", "irls_iters", "irls_time", "dsm_iters", "dsm_time"])
        w.writerows(rows)
    print(f"Saved: {tab_path}")

    # -------- plot --------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    irls_gaps = np.maximum(res_irls["gaps"], 1e-16)
    dsm_gaps  = np.maximum(res_dsm["gaps"],  1e-16)
    dsm_curr  = np.maximum([f - f_star for f in res_dsm["f_vals"]], 1e-16)

    ax = axes[0]
    ax.semilogy(irls_gaps, color=COLOR_IRLS, marker="o", markersize=2.5,
                linewidth=1.4, label="IRLS")
    ax.semilogy(dsm_curr, color=COLOR_FCUR, linewidth=0.5, alpha=0.45,
                label=r"SGPTL  $f(w_i) - f^{*}$")
    ax.semilogy(dsm_gaps, color=COLOR_DSM, linewidth=1.4,
                label=r"SGPTL  $\bar{f}^{i} - f^{*}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"gap to $f^{*}$")
    ax.set_title("vs. iterations")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    ax.semilogy(res_irls["times"], irls_gaps,
                color=COLOR_IRLS, marker="o", markersize=2.5,
                linewidth=1.4, label="IRLS")
    ax.semilogy(res_dsm["times"], dsm_curr,
                color=COLOR_FCUR, linewidth=0.5, alpha=0.45,
                label=r"SGPTL  $f(w_i) - f^{*}$")
    ax.semilogy(res_dsm["times"], dsm_gaps,
                color=COLOR_DSM, linewidth=1.4,
                label=r"SGPTL  $\bar{f}^{i} - f^{*}$")
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel(r"gap to $f^{*}$")
    ax.set_title("vs. CPU time")
    ax.legend(loc="upper right")
    style_axes(ax)

    fig.suptitle(rf"$H={H}$, $M={M}$, $\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "comparison_irls_dsm.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # -------- solution quality summary --------
    w_irls, w_dsm = res_irls["w"], res_dsm["w"]
    print("\nSolution quality:")
    print(f"  ||w_irls - w*||_2 = {np.linalg.norm(w_irls - w_star):.3e}")
    print(f"  ||w_dsm  - w*||_2 = {np.linalg.norm(w_dsm  - w_star):.3e}")
    print(f"  IRLS sparsity   = {np.mean(np.abs(w_irls) < 1e-6):.0%}")
    print(f"  DSM  sparsity   = {np.mean(np.abs(w_dsm)  < 1e-6):.0%}")
    print(f"  True sparsity   = {np.mean(np.abs(w_star) < 1e-6):.0%}")


if __name__ == "__main__":
    run()
