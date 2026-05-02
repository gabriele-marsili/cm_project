"""
experiment_scalability.py
-------------------------
Scaling of total wall-clock time as the hidden-layer size H grows
(M = 5H). Produces scalability.pdf and scalability.csv for Chapter 5.
"""

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
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM, COLOR_AUX
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

        t0 = time.perf_counter()
        ri = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=IRLS_EPSSTOP,
                  k_max=IRLS_KMAX, solver="cholesky", f_star=f_star)
        ti = time.perf_counter() - t0

        t0 = time.perf_counter()
        rd = deflected_subgradient(X, y, LAMBDA, i_max=DSM_IMAX,
                                   beta=1.0, delta0=0.1 * f_star, rho=0.9,
                                   f_star=f_star)
        td = time.perf_counter() - t0

        gi = ri["gaps"][-1] if ri["gaps"] else float("nan")
        gd = rd["gaps"][-1] if rd["gaps"] else float("nan")
        ki = ri["n_iter"]
        kd = rd["n_iter"]
        print(f"IRLS {ti:.3f}s ({ki} iter, gap={gi:.1e})  |  "
              f"SGPTL {td:.3f}s ({kd} iter, gap={gd:.1e})")
        rows.append({
            "n": H, "m": M,
            "t_irls": ti, "iter_irls": ki, "gap_irls": gi,
            "t_dsm":  td, "iter_dsm":  kd, "gap_dsm":  gd,
        })

    # --- table ---
    tab_path = os.path.join(TAB_DIR, "scalability.csv")
    with open(tab_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {tab_path}")

    # --- plot ---
    Hs    = np.array([r["n"] for r in rows], dtype=float)
    t_irl = np.array([r["t_irls"] for r in rows])
    t_dsm = np.array([r["t_dsm"]  for r in rows])
    per_irl = t_irl / np.array([max(r["iter_irls"], 1) for r in rows])
    per_dsm = t_dsm / np.array([max(r["iter_dsm"],  1) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.loglog(Hs, t_irl, color=COLOR_IRLS, marker="o", markersize=5,
              linewidth=1.4, label="IRLS")
    ax.loglog(Hs, t_dsm, color=COLOR_DSM,  marker="s", markersize=5,
              linewidth=1.4, label="SGPTL")
    ref_irl = t_irl[0] * (Hs / Hs[0]) ** 3
    ref_dsm = t_dsm[0] * (Hs / Hs[0]) ** 2
    ax.loglog(Hs, ref_irl, color=COLOR_IRLS, linestyle="--",
              alpha=0.45, linewidth=1.0, label=r"$O(H^{3})$")
    ax.loglog(Hs, ref_dsm, color=COLOR_DSM, linestyle="--",
              alpha=0.45, linewidth=1.0, label=r"$O(H^{2})$")
    ax.set_xlabel(r"$H$ (hidden-layer size, $M=5H$)")
    ax.set_ylabel("total CPU time (s)")
    ax.set_title("Total time")
    ax.legend(loc="upper left")
    style_axes(ax)

    ax = axes[1]
    ax.loglog(Hs, per_irl, color=COLOR_IRLS, marker="o", markersize=5,
              linewidth=1.4, label="IRLS")
    ax.loglog(Hs, per_dsm, color=COLOR_DSM,  marker="s", markersize=5,
              linewidth=1.4, label="SGPTL")
    ax.set_xlabel(r"$H$")
    ax.set_ylabel("per-iteration CPU time (s)")
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
