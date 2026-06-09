"""IRLS vs SGPTL on a moderate problem (H=50, M=200, lam=0.1).

Iterations and CPU time to reach a grid of accuracy targets, plus
support recovery.
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

from src import irls, deflected_subgradient, make_lasso_problem, support_metrics
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_IRLS, COLOR_DSM, SIZE_DOUBLE)
apply_style()


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def first_index_under(gaps, threshold):
    """Index of the first gap at or below threshold"""
    for i, g in enumerate(gaps):
        if g <= threshold:
            return i
    return None


def first_time_under(gaps, times, threshold):
    """Wall-clock time when the gap first drops to or below threshold"""
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
    abs_f_star = abs(f_star)

    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    from src.lasso_utils import f_lasso
    f_w0 = float(f_lasso(X, y, w_ols, LAMBDA))

    res_irls = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=1e-12,
                    k_max=300, solver="cholesky",
                    w0=w_ols, f_star=f_star)
    # SGPTL: same OLS warm start as IRLS, theory-pure config (R = 1 default)
    res_dsm = deflected_subgradient(
        X, y, LAMBDA, w0=w_ols, i_max=30000,
        beta=1.0, delta0=0.1 * f_w0, rho=0.7, f_star=f_star,
    )

    # relative optimality gaps (f - f*)/|f*|
    irls_rel_gaps = [g / abs_f_star for g in res_irls["gaps"]]
    dsm_rel_gaps  = [g / abs_f_star for g in res_dsm["gaps"]]

    # iterations and time to reach each relative accuracy target
    eps_grid = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
    rows = []
    print(f"\n{'eps':>6}  {'IRLS iter':>10}  {'IRLS time':>10}  "
          f"{'SGPTL iter':>11}  {'SGPTL time':>11}")
    for eps in eps_grid:
        k_irls = first_index_under(irls_rel_gaps, eps)
        t_irls = first_time_under(irls_rel_gaps, res_irls["times"], eps)
        k_dsm  = first_index_under(dsm_rel_gaps, eps)
        t_dsm  = first_time_under(dsm_rel_gaps, res_dsm["times"], eps)
        rows.append([eps, k_irls, t_irls, k_dsm, t_dsm])
        ti_s = f"{t_irls:.2e}" if t_irls is not None else "  ---"
        td_s = f"{t_dsm:.2e}"  if t_dsm  is not None else "  ---"
        ki_s = str(k_irls) if k_irls is not None else "---"
        kd_s = str(k_dsm)  if k_dsm  is not None else "---"
        print(f"{eps:>6.0e}  {ki_s:>10}  {ti_s:>10}  {kd_s:>11}  {td_s:>11}")

    tab_path = os.path.join(TAB_DIR, "comparison_table.csv")
    with open(tab_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["epsilon", "irls_iters", "irls_time", "dsm_iters", "dsm_time"])
        wr.writerows(rows)
    print(f"Saved: {tab_path}")

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    irls_gaps = np.maximum(np.asarray(irls_rel_gaps, dtype=float), 1e-16)
    dsm_gaps  = np.maximum(np.asarray(dsm_rel_gaps,  dtype=float), 1e-16)
    rel_label = r"relative gap  $(f - f^{*})/|f^{*}|$"

    # IRLS hits the relative-gap floor (~3e-8) around iter 220, then runs to
    # k_max at a constant gap. Those post-convergence iterations carry no
    # information and add timing jitter to the wall-clock panel, so cut the
    # IRLS curve at the floor. Every reported target is >= 1e-6, reached by
    # iter 125, so the cut leaves all tabulated numbers unchanged.
    irls_floor = 1e-7
    irls_cut = (int(np.argmax(irls_gaps <= irls_floor)) + 1
                if np.any(irls_gaps <= irls_floor) else len(irls_gaps))
    irls_gaps = irls_gaps[:irls_cut]

    ax = axes[0]
    irls_iters = np.arange(1, len(irls_gaps) + 1)
    dsm_iters  = np.arange(1, len(dsm_gaps)  + 1)
    ax.loglog(irls_iters, irls_gaps,
              color=COLOR_IRLS, marker="o", markersize=4.5,
              linewidth=2.0, label="IRLS")
    ax.loglog(dsm_iters, dsm_gaps,
              color=COLOR_DSM, linewidth=2.0,
              label=r"SGPTL  $(\bar{f}^{i} - f^{*})/|f^{*}|$")
    ax.scatter([1], [irls_gaps[0]], s=80, marker="*",
               color="black", zorder=6, label="OLS warm start (shared)")
    ax.set_xlabel("Iteration  (log scale)")
    ax.set_ylabel(rel_label + r"  (log scale)")
    ax.set_title("Convergence vs. iterations")
    ax.legend(loc="lower left")
    style_axes(ax)

    ax = axes[1]
    # OLS warm-start time is 0 -> offset so it shows on a log axis
    irls_t = np.asarray(res_irls["times"], dtype=float)[:irls_cut]
    dsm_t  = np.asarray(res_dsm["times"],  dtype=float)
    smallest = min(irls_t[1] if len(irls_t) > 1 else 1e-5,
                   dsm_t[1]  if len(dsm_t)  > 1 else 1e-5)
    t_start = smallest / 2.0
    irls_t_plot = irls_t.copy()
    irls_t_plot[0] = t_start
    dsm_t_plot = dsm_t.copy()
    dsm_t_plot[0] = t_start

    ax.loglog(irls_t_plot, irls_gaps,
              color=COLOR_IRLS, marker="o", markersize=4.5,
              linewidth=2.0, label="IRLS")
    ax.loglog(dsm_t_plot, dsm_gaps,
              color=COLOR_DSM, linewidth=2.0,
              label=r"SGPTL  $(\bar{f}^{i} - f^{*})/|f^{*}|$")
    ax.scatter([t_start], [irls_gaps[0]], s=80, marker="*",
               color="black", zorder=6, label="OLS warm start (shared)")
    ax.set_xlabel("CPU time (s)  (log scale)")
    ax.set_ylabel(rel_label + r"  (log scale)")
    ax.set_title("Convergence vs. CPU time")
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.suptitle(rf"IRLS vs SGPTL  ($H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$)",
                 y=1.00, fontsize=15)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "comparison_irls_dsm.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # support recovery: same threshold on IRLS and SGPTL for a fair comparison
    w_irls, w_dsm = res_irls["w"], res_dsm["w"]
    print("\nSolution quality:")
    print(f"  ||w_irls - w*||_2 = {np.linalg.norm(w_irls - w_star):.3e}")
    print(f"  ||w_dsm  - w*||_2 = {np.linalg.norm(w_dsm  - w_star):.3e}")

    print("\nSupport recovery (uniform threshold on |w_i|):")
    print(f"  {'tol':>6}  {'method':>6}  {'sparsity':>9}  "
          f"{'precision':>9}  {'recall':>7}  {'F1':>5}")
    rows_supp = []
    for tol in (1e-6, 1e-3):
        for tag, w_h in (("IRLS", w_irls), ("SGPTL", w_dsm), ("true", w_star)):
            m = support_metrics(w_star, w_h, tol=tol)
            print(f"  {tol:>6.0e}  {tag:>6}  {m['sparsity']:>8.0%}  "
                  f"{m['precision']:>9.3f}  {m['recall']:>7.3f}  {m['f1']:>5.3f}")
            rows_supp.append([tol, tag, m['sparsity'], m['precision'],
                              m['recall'], m['f1']])

    supp_path = os.path.join(TAB_DIR, "support_recovery.csv")
    with open(supp_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tol", "method", "sparsity", "precision", "recall", "f1"])
        wr.writerows(rows_supp)
    print(f"Saved: {supp_path}")


if __name__ == "__main__":
    run()
