"""Scalability: total wall-clock and per-iteration cost across H = 50..2000, M = 5H."""

import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import irls, deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_IRLS, COLOR_DSM, SIZE_DOUBLE)
apply_style()


SEED         = 42
LAMBDA       = 0.10
NOISE        = 0.05
M_RATIO      = 5
IRLS_KMAX    = 100
IRLS_EPSSTOP = 1e-8
DSM_IMAX     = 3000

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def run() -> None:
    print("=" * 60)
    print("Scalability experiment")
    print("=" * 60)

    H_values = [50, 100, 500, 1000, 2000]
    rows = []
    for H in H_values:
        M = M_RATIO * H
        print(f"\nH={H:5d}, M={M:6d} ...", end=" ", flush=True)

        X, y, _, f_star, _ = make_lasso_problem(
            n=H, m=M, sparsity=0.1, noise_std=NOISE,
            lam=LAMBDA, random_state=SEED,
        )

        w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H),
                          X.T @ y, method="cholesky")

        t0 = time.perf_counter()
        ri = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=IRLS_EPSSTOP,
                  k_max=IRLS_KMAX, solver="cholesky",
                  w0=w_ols, f_star=f_star)
        t_irls = time.perf_counter() - t0

        t0 = time.perf_counter()
        rd = deflected_subgradient(
            X, y, LAMBDA, w0=w_ols, i_max=DSM_IMAX,
            beta=1.0, delta0=0.1 * f_star, rho=0.95, f_star=f_star,
        )
        t_dsm = time.perf_counter() - t0

        gap_irls = ri["gaps"][-1] if ri["gaps"] else float("nan")
        gap_dsm  = rd["gaps"][-1] if rd["gaps"] else float("nan")
        k_irls   = ri["n_iter"]
        k_dsm    = rd["n_iter"]
        print(f"IRLS {t_irls:.3f}s ({k_irls} iter, gap={gap_irls:.1e})  |  "
              f"SGPTL {t_dsm:.3f}s ({k_dsm} iter, gap={gap_dsm:.1e})")
        rows.append({
            "n": H, "m": M,
            "t_irls": t_irls, "iter_irls": k_irls, "gap_irls": gap_irls,
            "t_dsm":  t_dsm,  "iter_dsm":  k_dsm,  "gap_dsm":  gap_dsm,
        })

    tab_path = os.path.join(TAB_DIR, "scalability.csv")
    with open(tab_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved: {tab_path}")

    Hs       = np.array([r["n"] for r in rows], dtype=float)
    t_irls_v = np.array([r["t_irls"] for r in rows])
    t_dsm_v  = np.array([r["t_dsm"]  for r in rows])
    per_irls = t_irls_v / np.array([max(r["iter_irls"], 1) for r in rows])
    per_dsm  = t_dsm_v  / np.array([max(r["iter_dsm"],  1) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    ax = axes[0]
    ax.loglog(Hs, t_irls_v, color=COLOR_IRLS, marker="o", markersize=7,
              linewidth=2.0, label="IRLS")
    ax.loglog(Hs, t_dsm_v, color=COLOR_DSM, marker="s", markersize=7,
              linewidth=2.0, label="SGPTL")
    # reference slopes anchored on smallest H
    ref_h3 = t_irls_v[0] * (Hs / Hs[0]) ** 3
    ref_h2 = t_dsm_v[0]  * (Hs / Hs[0]) ** 2
    ax.loglog(Hs, ref_h3, color=COLOR_IRLS, linestyle="--",
              alpha=0.55, linewidth=1.2, label=r"$O(H^{3})$ ref.")
    ax.loglog(Hs, ref_h2, color=COLOR_DSM, linestyle="--",
              alpha=0.55, linewidth=1.2, label=r"$O(H^{2})$ ref.")
    ax.set_xlabel(r"$H$ (hidden-layer size, $M=5H$)  (log scale)")
    ax.set_ylabel("Total CPU time (s)  (log scale)")
    ax.set_title("Total wall-clock time")
    ax.legend(loc="upper left")
    style_axes(ax)

    ax = axes[1]
    ax.loglog(Hs, per_irls, color=COLOR_IRLS, marker="o", markersize=7,
              linewidth=2.0, label="IRLS")
    ax.loglog(Hs, per_dsm, color=COLOR_DSM, marker="s", markersize=7,
              linewidth=2.0, label="SGPTL")
    ax.set_xlabel(r"$H$  (log scale)")
    ax.set_ylabel("Per-iteration CPU time (s)  (log scale)")
    ax.set_title("Per-iteration cost")
    ax.legend(loc="upper left")
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "scalability.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
