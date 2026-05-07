"""
experiment_convergence.py
-------------------------
Convergence of IRLS and SGPTL on a representative ELM-style instance.

Produces three figures used in Chapter 5 of the report:
  - convergence_vs_iter.pdf   gap vs. iterations on semilog axes
  - convergence_vs_time.pdf   gap vs. wall-clock time
  - dsm_nonmonotone.pdf       DSM current-vs-record value (linear axes)

For DSM both the current gap f(w_i) - f* (smooth, sublinear) and the
record gap f_bar^i - f* (monotone step function) are plotted, because
the record alone has a staircase look that makes the algorithm appear
to "jump" — the smooth curve reveals the actual subgradient progress.
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

from src import irls, deflected_subgradient, make_lasso_problem
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM, \
    COLOR_FBAR, COLOR_FCUR, COLOR_REF
apply_style()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED       = 42
H          = 100   # hidden-layer size
M          = 300   # training samples
SPARSITY   = 0.10
LAMBDA     = 0.10
NOISE      = 0.05

IRLS_KMAX  = 100
DSM_IMAX   = 8000
DSM_RHO    = 0.9   # sweet spot from the parameter sensitivity sweep

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    """Clip non-positive values to a small positive floor for log-scale plots."""
    return np.maximum(np.asarray(arr, dtype=float), floor)


def run() -> None:
    print("=" * 60)
    print("Convergence experiment")
    print("=" * 60)

    X, y, _, f_star, w_star = make_lasso_problem(
        n=H, m=M, sparsity=SPARSITY, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")
    print(f"True sparsity: {np.mean(np.abs(w_star) < 1e-6):.0%}")

    # OLS warm start shared by both algorithms (report §5.2)
    from src.linear_solvers import solve_spd
    A = X.T @ X
    b = X.T @ y
    w_ols = solve_spd(A + 1e-12 * np.eye(H), b, method="cholesky")

    # --- IRLS ---
    res_irls = irls(X, y, LAMBDA,
                    eps_thr=1e-8, eps_stop=1e-12,
                    k_max=IRLS_KMAX, solver="cholesky",
                    w0=w_ols, f_star=f_star, verbose=False)
    print(f"IRLS : {res_irls['n_iter']} iter, "
          f"final gap = {res_irls['gaps'][-1]:.3e}, "
          f"converged = {res_irls['converged']}")

    # --- SGPTL ---
    res_dsm = deflected_subgradient(
        X, y, LAMBDA,
        w0=w_ols, i_max=DSM_IMAX, beta=1.0,
        delta0=0.1 * f_star, rho=DSM_RHO,
        f_star=f_star, verbose=False,
    )
    f_curr_gap = _safe_log([f - f_star for f in res_dsm["f_vals"]])
    print(f"SGPTL: {res_dsm['n_iter']} iter, "
          f"final record gap = {res_dsm['gaps'][-1]:.3e}, "
          f"final current gap = {f_curr_gap[-1]:.3e}")

    # =====================================================================
    # Figure 1 — convergence vs iterations (semilog)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    irls_gaps = _safe_log(res_irls["gaps"])
    ax.semilogy(irls_gaps, color=COLOR_IRLS, marker="o", markersize=2.8,
                linewidth=1.4, label=r"$f(w_k) - f^{*}$")
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"$f(w_k) - f^{*}$")
    ax.set_title(r"\textbf{IRLS}".replace("\\textbf", "") if False else "IRLS")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    dsm_fbar = _safe_log(res_dsm["gaps"])
    ax.semilogy(f_curr_gap, color=COLOR_FCUR, linewidth=0.6, alpha=0.55,
                label=r"$f(w_i) - f^{*}$  (current)")
    ax.semilogy(dsm_fbar, color=COLOR_DSM, linewidth=1.6,
                label=r"$\bar{f}^{i} - f^{*}$  (record)")
    ax.set_xlabel(r"Iteration $i$")
    ax.set_ylabel(r"gap to $f^{*}$")
    ax.set_title("SGPTL")
    ax.legend(loc="upper right")
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence_vs_iter.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # =====================================================================
    # Figure 2 — convergence vs CPU time
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(res_irls["times"], irls_gaps,
                color=COLOR_IRLS, marker="o", markersize=2.8,
                linewidth=1.4, label="IRLS")
    ax.semilogy(res_dsm["times"], dsm_fbar,
                color=COLOR_DSM, linewidth=1.4, label=r"SGPTL ($\bar{f}^{i}$)")
    ax.semilogy(res_dsm["times"], f_curr_gap,
                color=COLOR_FCUR, linewidth=0.5, alpha=0.45,
                label=r"SGPTL ($f(w_i)$)")
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel(r"gap to $f^{*}$")
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence_vs_time.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # =====================================================================
    # Figure 3 — DSM current value vs record value (linear axes)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    iters = np.arange(len(res_dsm["f_vals"]))
    ax.plot(iters, res_dsm["f_vals"],
            color=COLOR_FCUR, linewidth=0.55, alpha=0.7,
            label=r"$f(w_i)$ (oscillates)")
    ax.plot(iters, res_dsm["f_bar"],
            color=COLOR_FBAR, linewidth=1.6,
            label=r"$\bar{f}^{i}$ (record, monotone)")
    ax.axhline(f_star, color=COLOR_REF, linestyle="--", linewidth=1.0,
               label=r"$f^{*}$")
    ax.set_xlabel(r"Iteration $i$")
    ax.set_ylabel(r"objective $f$")
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dsm_nonmonotone.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
