"""SGPTL sensitivity to the initial target gap delta_0: three families.

Family A: delta_0 = c * f_LASSO(w_OLS)        (current default, f*-free)
Family B: delta_0 = c * f_OLS(w_OLS)          (drops lam*||w||_1, f*-free)
Family C: delta_0 = c * f^*                   (oracle baseline, diagnostic)
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

from src import deflected_subgradient, irls, make_lasso_problem
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         RAMP_BLUES, RAMP_ORANGES, RAMP_PURPLES,
                         SIZE_DOUBLE)

apply_style()


SEED = 42
H, M = 100, 300
LAMBDA = 0.1
NOISE = 0.05
SPARSITY = 0.1
I_MAX = 8000
RHO = 0.7
R_PATIENCE = 1.0
GAMMA_MIN = 0.05
BETA = 1.0

C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    return np.maximum(np.asarray(arr, dtype=float), floor)


def _count_contractions(delta_hist) -> int:
    return int(np.sum(np.diff(np.asarray(delta_hist, dtype=float)) < 0))


def run() -> None:
    print("=" * 60)
    print("SGPTL: delta_0 family sweep (A / B / C)")
    print("=" * 60)

    X, y, _, _, _ = make_lasso_problem(
        n=H, m=M, sparsity=SPARSITY, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )

    # OLS warm start (ridge-regularised SPD solve to mirror experiment_params)
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")

    # f^* via IRLS run to machine precision (NOT used to tune SGPTL except in
    # the oracle Family C, and only there for diagnostic purposes).
    irls_res = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=1e-14,
                    k_max=300, solver="cholesky", w0=w_ols)
    f_star = float(np.min(irls_res["f_vals"]))

    f_A = float(f_lasso(X, y, w_ols, LAMBDA))        # full LASSO at OLS
    resid = X @ w_ols - y
    f_B = float(0.5 * resid @ resid)                 # OLS objective (no L1)
    f_C = f_star                                     # oracle

    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, sparsity={SPARSITY}, "
          f"noise={NOISE}, seed={SEED}")
    print(f"f^*           = {f_star:.6e}")
    print(f"f_LASSO(w_OLS)= {f_A:.6e}   (Family A scale)")
    print(f"f_OLS(w_OLS)  = {f_B:.6e}   (Family B scale)")
    print(f"oracle f^*    = {f_C:.6e}   (Family C scale)\n")

    families = [
        ("A", r"$c\,f_{\mathrm{LASSO}}(w_{\mathrm{OLS}})$", f_A, RAMP_BLUES),
        ("B", r"$c\,f_{\mathrm{OLS}}(w_{\mathrm{OLS}})$",   f_B, RAMP_ORANGES),
        ("C", r"$c\,f^{*}$ (oracle)",                       f_C, RAMP_PURPLES),
    ]

    records = []
    trajectories = {}

    for fam_id, fam_label, scale, _ramp in families:
        print(f"--- Family {fam_id}:  delta_0 = c * {scale:.4e} ---")
        trajectories[fam_id] = {}
        for c in C_VALUES:
            delta0 = c * scale
            res = deflected_subgradient(
                X, y, LAMBDA, w0=w_ols.copy(), i_max=I_MAX,
                beta=BETA, delta0=delta0, R=R_PATIENCE, rho=RHO,
                gamma_min=GAMMA_MIN, f_star=f_star,
            )
            gaps = np.asarray(res["gaps"], dtype=float)
            n_contr = _count_contractions(res["delta_hist"])
            final_gap = float(gaps[-1])
            trajectories[fam_id][c] = gaps
            records.append({
                "family":           fam_id,
                "c":                c,
                "delta0":           delta0,
                "scale":            scale,
                "final_record_gap": final_gap,
                "n_contractions":   n_contr,
                "n_iter":           int(res["n_iter"]),
            })
            print(f"  c={c:<5g}  delta0={delta0:.3e}  "
                  f"final_gap={final_gap:.3e}  contr={n_contr:3d}  "
                  f"iters={res['n_iter']}")
        print()

    # --- CSV ---
    csv_path = os.path.join(TAB_DIR, "delta0_families.csv")
    if _HAS_PANDAS:
        pd.DataFrame(records).to_csv(csv_path, index=False)
    else:
        header = "family,c,delta0,scale,final_record_gap,n_contractions,n_iter"
        rows = [
            f"{r['family']},{r['c']},{r['delta0']:.6e},{r['scale']:.6e},"
            f"{r['final_record_gap']:.6e},{r['n_contractions']},{r['n_iter']}"
            for r in records
        ]
        with open(csv_path, "w") as fh:
            fh.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"Saved CSV: {csv_path}")

    # --- Figure: 1x3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), sharey=True)
    for ax, (fam_id, fam_label, scale, ramp) in zip(axes, families):
        cmap = plt.get_cmap(ramp)
        colors = cmap(np.linspace(0.35, 1.0, len(C_VALUES)))
        for c, color in zip(C_VALUES, colors):
            gaps = _safe_log(trajectories[fam_id][c])
            iters = np.arange(1, len(gaps) + 1)
            ax.loglog(iters, gaps, color=color, linewidth=1.8,
                      label=rf"$c={c:g}$")
        ax.set_title(rf"Family {fam_id}: $\delta_{{0}}=${fam_label}")
        ax.set_xlabel(r"Iteration $i$  (log scale)")
        ax.legend(loc="lower left", fontsize=10)
        style_axes(ax)
    axes[0].set_ylabel(r"$\bar{f}^{i} - f^{*}$  (log scale)")

    fig.suptitle(
        rf"SGPTL: sensitivity of $\delta_{{0}}$  "
        rf"($H={H}$, $M={M}$, $\lambda={LAMBDA}$, $\rho={RHO}$, "
        rf"$\gamma_{{\min}}={GAMMA_MIN}$, $i_{{\max}}={I_MAX}$, "
        rf"warm start $w_{{0}}=w_{{\mathrm{{OLS}}}}$)",
        y=1.02, fontsize=14,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "delta0_families.pdf")
    fig.savefig(path)
    print(f"Saved figure: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
