"""IRLS vs SGPTL convergence on a fixed instance (H=100, M=300, lam=0.1)."""

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

IRLS_KMAX     = 1500
IRLS_EPS_THR  = 1e-12   # tight on this figure to expose the linear tail
                        # (default 1e-8 used elsewhere reaches its O(eps_thr)
                        #  smoothing floor before the linear regime is visible)
DSM_IMAX  = 8000
DSM_RHO   = 0.7

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
    from src.lasso_utils import f_lasso
    f_w0 = float(f_lasso(X, y, w_ols, LAMBDA))
    print(f"OLS warm start (shared): f(w_0)={f_w0:.6f}, gap0={f_w0-f_star:.3e}")

    res_irls = irls(X, y, LAMBDA,
                    eps_thr=IRLS_EPS_THR, eps_stop=1e-16,
                    k_max=IRLS_KMAX, solver="cholesky",
                    w0=w_ols, f_star=f_star)
    print(f"IRLS : {res_irls['n_iter']} iter, "
          f"final gap = {res_irls['gaps'][-1]:.3e}, "
          f"converged = {res_irls['converged']}")

    # SGPTL: same OLS warm start as IRLS, theory-pure config (R = 1 by default,
    # ensuring the r > R branch triggers a sensible number of delta contractions;
    # see Section 5.3 / 5.4 of the report).
    res_dsm = deflected_subgradient(
        X, y, LAMBDA,
        w0=w_ols, i_max=DSM_IMAX, beta=1.0,
        delta0=0.1 * f_w0, rho=DSM_RHO,
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

    # Relative gap (f - f*)/|f*| is the quantity plotted in BOTH panels and in
    # every gap figure/table of the report: an absolute gap of 1e-6 means very
    # different things at f*=1e+4 vs f*=1e-3, so we normalise by |f*|.
    af = abs(f_star)
    def _rel(arr):
        return _safe_log(np.asarray(arr, dtype=float) / af)

    # ---- Figure 1: relative gap vs iteration, IRLS (semilog) + SGPTL (loglog) ----
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    ax = axes[0]
    irls_rel = _rel(res_irls["gaps"])
    irls_iters = np.arange(len(irls_rel))
    ax.semilogy(irls_iters, irls_rel, color=COLOR_IRLS, linewidth=1.8,
                label=r"$(f(w_k) - f^{*})/|f^{*}|$")
    # Asymptotic linear rate, fitted on the linear tail: after the fast
    # transient, before the O(eps_thr) smoothing floor. On semilog a constant
    # per-iteration ratio < 1 is a straight line -- this is what "linear" means.
    lo, hi = 400, min(1200, len(irls_rel) - 1)
    if hi > lo and irls_rel[lo] > 1e-14 and irls_rel[hi] > 1e-14:
        rate = (irls_rel[hi] / irls_rel[lo]) ** (1.0 / (hi - lo))
        ref_k = np.arange(lo, hi + 1)
        ref = irls_rel[lo] * rate ** (ref_k - lo)
        ax.semilogy(ref_k, ref, color="black", linestyle="--", linewidth=1.3,
                    label=rf"linear fit, rate $\approx {rate:.3f}$/iter")
        print(f"IRLS asymptotic linear rate over [{lo},{hi}] = {rate:.4f}")
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
    ax.set_title(r"IRLS: linear convergence (slow, rate $\approx 1$)")
    ax.legend(loc="upper right")
    style_axes(ax)

    ax = axes[1]
    dsm_fbar_rel = _rel(res_dsm["gaps"])
    dsm_iters = np.arange(1, len(dsm_fbar_rel) + 1)
    f_curr_rel = _rel([f - f_star for f in res_dsm["f_vals"]])
    # log-uniform subsample so the late-iteration density does not crush the trace
    n_curr = len(f_curr_rel)
    if n_curr > 600:
        log_idx = np.unique(np.round(np.logspace(
            0, np.log10(n_curr - 1), num=600)).astype(int))
        log_idx = log_idx[log_idx < n_curr]
        sub_iters = dsm_iters[log_idx]
        sub_curr  = f_curr_rel[log_idx]
    else:
        sub_iters, sub_curr = dsm_iters, f_curr_rel
    ax.loglog(sub_iters, sub_curr, color=COLOR_FCUR, linewidth=0.9,
              alpha=0.65, label=r"$(f(w_i) - f^{*})/|f^{*}|$  (current, subsampled)")
    ax.loglog(dsm_iters, dsm_fbar_rel,
              color=COLOR_DSM, linewidth=2.0,
              label=r"$(\bar{f}^{i} - f^{*})/|f^{*}|$  (record)")
    ax.set_xlabel(r"Iteration $i$  (log scale)")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
    ax.set_title(r"SGPTL: sublinear $O(1/\sqrt{i})$ convergence")
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "convergence_vs_iter.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # ---- Figure 2: gap vs CPU time (in ms), log-log axes ----
    # OLS warm-start time is 0; substitute a small offset to plot in log scale.
    irls_t = np.asarray(res_irls["times"], dtype=float) * 1000.0
    dsm_t  = np.asarray(res_dsm["times"],  dtype=float) * 1000.0
    smallest = min(irls_t[1] if len(irls_t) > 1 else 1e-2,
                   dsm_t[1]  if len(dsm_t)  > 1 else 1e-2)
    t_start = smallest / 2.0
    irls_t_plot = irls_t.copy(); irls_t_plot[0] = t_start
    dsm_t_plot  = dsm_t.copy();  dsm_t_plot[0]  = t_start

    fig, ax = plt.subplots(figsize=SIZE_SINGLE)
    ax.loglog(irls_t_plot, irls_rel,
              color=COLOR_IRLS, linewidth=1.8, label=r"IRLS")
    ax.loglog(dsm_t_plot, dsm_fbar_rel,
              color=COLOR_DSM, linewidth=1.8,
              label=r"SGPTL  (record $\bar{f}^{i}$)")
    ax.scatter([t_start], [irls_rel[0]], s=80, marker="*",
               color="black", zorder=6, label="OLS warm start (shared)")
    ax.set_xlabel("CPU time (ms)  (log scale)")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
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
    cutoff = min(2000, len(f_vals))
    iters = np.arange(cutoff)
    fcur_gap_lin = np.maximum(f_vals[:cutoff] - f_star, 1e-16) / af
    fbar_gap_lin = np.maximum(f_bar[:cutoff]  - f_star, 1e-16) / af

    fig, ax = plt.subplots(figsize=SIZE_SINGLE)
    ax.semilogy(iters, fcur_gap_lin,
                color=COLOR_FCUR, linewidth=0.9, alpha=0.65,
                label=r"$(f(w_i) - f^{*})/|f^{*}|$  (current, oscillates)")
    ax.semilogy(iters, fbar_gap_lin,
                color=COLOR_FBAR, linewidth=2.0,
                label=r"$(\bar{f}^{i} - f^{*})/|f^{*}|$  (record, monotone)")
    spike_idx = int(np.argmax(f_vals[:cutoff] - f_bar[:cutoff]))
    ax.annotate(rf"largest overshoot at $i={spike_idx}$",
                xy=(spike_idx, fcur_gap_lin[spike_idx]),
                xytext=(spike_idx + 200, fcur_gap_lin[spike_idx] * 1.8),
                fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0))
    ax.set_xlabel(r"Iteration $i$")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
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
