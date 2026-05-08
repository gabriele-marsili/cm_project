"""
Convergence experiment (report §5.2).

Runs IRLS and SGPTL on the same column-normalised LASSO instance
(H = 100, M = 300, lam = 0.1) starting from the OLS warm start, and produces
the three figures used in chapter 5:

    convergence_vs_iter.pdf   gap vs. iterations on semilog axes
    convergence_vs_time.pdf   gap vs. wall-clock time
    dsm_nonmonotone.pdf       SGPTL current value vs. record (semilog y)

For SGPTL we plot both the current gap f(w_i) - f* and the record gap
f_bar^i - f*: the record alone is a staircase, which understates the
underlying sublinear trajectory.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import irls, deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_IRLS, COLOR_DSM,
                         COLOR_FBAR, COLOR_FCUR, COLOR_REF, COLOR_AUX,
                         SIZE_SINGLE, SIZE_DOUBLE)
apply_style()


SEED      = 42
H         = 100
M         = 300
SPARSITY  = 0.10
LAMBDA    = 0.10
NOISE     = 0.05

IRLS_KMAX = 100
DSM_IMAX  = 8000
DSM_RHO   = 0.9

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
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

    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")

    res_irls = irls(X, y, LAMBDA,
                    eps_thr=1e-8, eps_stop=1e-12,
                    k_max=IRLS_KMAX, solver="cholesky",
                    w0=w_ols, f_star=f_star)
    print(f"IRLS : {res_irls['n_iter']} iter, "
          f"final gap = {res_irls['gaps'][-1]:.3e}, "
          f"converged = {res_irls['converged']}")

    res_dsm = deflected_subgradient(
        X, y, LAMBDA,
        w0=w_ols, i_max=DSM_IMAX, beta=1.0,
        delta0=0.1 * f_star, rho=DSM_RHO,
        f_star=f_star,
    )
    f_curr_gap = _safe_log([f - f_star for f in res_dsm["f_vals"]])
    print(f"SGPTL: {res_dsm['n_iter']} iter, "
          f"final record gap = {res_dsm['gaps'][-1]:.3e}, "
          f"final current gap = {f_curr_gap[-1]:.3e}")

    # ---- Save full convergence trajectories to CSV ----
    irls_gaps_raw = np.asarray(res_irls["gaps"], dtype=float)
    irls_times_raw = np.asarray(res_irls["times"], dtype=float)
    dsm_gaps_raw = np.asarray(res_dsm["gaps"], dtype=float)
    dsm_times_raw = np.asarray(res_dsm["times"], dtype=float)
    dsm_fcur_raw = np.asarray([f - f_star for f in res_dsm["f_vals"]], dtype=float)

    irls_n = len(irls_gaps_raw)
    dsm_n = len(dsm_gaps_raw)

    irls_rows = [
        {
            "algorithm": "IRLS",
            "iteration": int(k),
            "gap": float(irls_gaps_raw[k]),
            "record_gap": float(irls_gaps_raw[k]),  # IRLS is monotone
            "time_s": float(irls_times_raw[k]) if k < len(irls_times_raw) else float("nan"),
        }
        for k in range(irls_n)
    ]
    dsm_record = np.minimum.accumulate(np.maximum(dsm_gaps_raw, 0.0))
    dsm_rows = [
        {
            "algorithm": "SGPTL",
            "iteration": int(i + 1),
            "gap": float(dsm_fcur_raw[i]),
            "record_gap": float(dsm_record[i]),
            "time_s": float(dsm_times_raw[i]) if i < len(dsm_times_raw) else float("nan"),
        }
        for i in range(dsm_n)
    ]

    csv_path = os.path.join(TAB_DIR, "convergence_instance.csv")
    all_rows = irls_rows + dsm_rows
    if _HAS_PANDAS:
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    else:
        header = "algorithm,iteration,gap,record_gap,time_s"
        lines = [f"{r['algorithm']},{r['iteration']},{r['gap']:.6e},"
                 f"{r['record_gap']:.6e},{r['time_s']:.6e}" for r in all_rows]
        with open(csv_path, "w") as fh:
            fh.write(header + "\n" + "\n".join(lines) + "\n")
    print(f"Saved convergence CSV: {csv_path}")

    # ---- Figure 1: gap vs iteration, IRLS (semilog) + SGPTL (loglog) ----
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    ax = axes[0]
    irls_gaps = _safe_log(res_irls["gaps"])
    irls_iters = np.arange(len(irls_gaps))
    ax.semilogy(irls_iters, irls_gaps, color=COLOR_IRLS, marker="o",
                markersize=4.0, linewidth=1.8, label=r"$f(w_k) - f^{*}$")
    # Annotate the linear-rate slope: pick two well-separated iterations and
    # report the per-iteration multiplicative reduction (= empirical rate).
    if len(irls_gaps) > 30 and irls_gaps[10] > 1e-12 and irls_gaps[30] > 1e-12:
        rate = (irls_gaps[30] / irls_gaps[10]) ** (1.0 / 20.0)
        ax.text(0.98, 0.92, rf"linear rate $\approx {rate:.3f}$ per iter",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.92))
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"$f(w_k) - f^{*}$  (log scale)")
    ax.set_title("IRLS: linear convergence")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    dsm_fbar = _safe_log(res_dsm["gaps"])
    dsm_iters = np.arange(1, len(dsm_fbar) + 1)
    # Log-uniform subsample of the (oscillating) current-value trace: at log-x
    # the late iterations cram exponentially many points into a fixed visual
    # span, so the raw trace becomes a solid band hiding the oscillation
    # pattern. We pick ~80 points per decade — enough to preserve the
    # zig-zag envelope without crowding the panel.
    n_curr = len(f_curr_gap)
    if n_curr > 600:
        log_idx = np.unique(np.round(np.logspace(
            0, np.log10(n_curr - 1), num=600)).astype(int))
        log_idx = log_idx[log_idx < n_curr]
        sub_iters = dsm_iters[log_idx]
        sub_curr  = f_curr_gap[log_idx]
    else:
        sub_iters, sub_curr = dsm_iters, f_curr_gap
    ax.loglog(sub_iters, sub_curr, color=COLOR_FCUR, linewidth=0.9,
              alpha=0.65, label=r"$f(w_i) - f^{*}$  (current, subsampled)")
    ax.loglog(dsm_iters, dsm_fbar,
              color=COLOR_DSM, linewidth=2.0,
              label=r"$\bar{f}^{i} - f^{*}$  (record)")
    ax.set_xlabel(r"Iteration $i$  (log scale)")
    ax.set_ylabel(r"gap to $f^{*}$  (log scale)")
    ax.set_title(r"SGPTL: sublinear $O(1/\sqrt{i})$ convergence")
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence_vs_iter.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ---- Figure 2: gap vs CPU time, log-log axes ----
    # Both algorithms share the OLS warm-start point at t = 0; we cannot
    # plot log(0), so we replace the warm-start time with a small offset
    # equal to half the smallest non-zero measurement. This way the panel
    # explicitly shows the common starting gap and the divergence at
    # iteration 1 — IRLS' first weighted-normal-equation solve drops the
    # gap by an order of magnitude, while SGPTL's first subgradient step
    # only nudges it.
    irls_t = np.asarray(res_irls["times"], dtype=float)
    dsm_t  = np.asarray(res_dsm["times"],  dtype=float)
    smallest = min(irls_t[1] if len(irls_t) > 1 else 1e-5,
                   dsm_t[1]  if len(dsm_t)  > 1 else 1e-5)
    t_start = smallest / 2.0
    irls_t_plot = irls_t.copy(); irls_t_plot[0] = t_start
    dsm_t_plot  = dsm_t.copy();  dsm_t_plot[0]  = t_start

    fig, ax = plt.subplots(figsize=SIZE_SINGLE)
    ax.loglog(irls_t_plot, irls_gaps,
              color=COLOR_IRLS, marker="o", markersize=4.0,
              linewidth=1.8, label=r"IRLS")
    ax.loglog(dsm_t_plot, dsm_fbar,
              color=COLOR_DSM, linewidth=1.8,
              label=r"SGPTL  ($\bar{f}^{i}$)")
    # Mark the shared warm-start point.
    ax.scatter([t_start], [irls_gaps[0]], s=80, marker="*",
               color="black", zorder=6, label="OLS warm start (shared)")
    ax.set_xlabel("CPU time (s)  (log scale)")
    ax.set_ylabel(r"gap to $f^{*}$  (log scale)")
    ax.set_title(rf"Gap vs wall-clock time  ($H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$)")
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence_vs_time.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ---- Figure 3: SGPTL non-monotonicity ----
    f_vals = np.asarray(res_dsm["f_vals"])
    f_bar  = np.asarray(res_dsm["f_bar"])
    # Show enough iterations to see oscillations clearly and the eventual
    # decay; use semilog y on (f - f*) so the dynamics across decades are
    # visible without zooming.
    cutoff = min(2000, len(f_vals))
    iters = np.arange(cutoff)
    fcur_gap_lin = np.maximum(f_vals[:cutoff] - f_star, 1e-16)
    fbar_gap_lin = np.maximum(f_bar[:cutoff]  - f_star, 1e-16)

    fig, ax = plt.subplots(figsize=SIZE_SINGLE)
    ax.semilogy(iters, fcur_gap_lin,
                color=COLOR_FCUR, linewidth=0.9, alpha=0.65,
                label=r"$f(w_i) - f^{*}$  (current, oscillates)")
    ax.semilogy(iters, fbar_gap_lin,
                color=COLOR_FBAR, linewidth=2.0,
                label=r"$\bar{f}^{i} - f^{*}$  (record, monotone)")
    # Mark the largest overshoot of f_curr above the record.
    spike_idx = int(np.argmax(f_vals[:cutoff] - f_bar[:cutoff]))
    ax.annotate(rf"largest overshoot at $i={spike_idx}$",
                xy=(spike_idx, fcur_gap_lin[spike_idx]),
                xytext=(spike_idx + 200, fcur_gap_lin[spike_idx] * 1.8),
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    ax.set_xlabel(r"Iteration $i$")
    ax.set_ylabel(r"gap to $f^{*}$  (log scale)")
    ax.set_title("SGPTL non-monotone behaviour: current vs record")
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "dsm_nonmonotone.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
