"""Validate the proxy δ_0 = c·f(w_OLS) against the ideal δ_0 = c·f^* on synthetic.

The implementable rule (δ_0 = c·f(w_OLS)) uses only quantities a caller can
compute at runtime; the ideal rule (δ_0 = c·f^*) requires knowing f^* in
advance. On synthetic problems we know f^* exactly (sklearn-tight); this
experiment compares the SGPTL trajectory under both rules across a grid of
problem instances and c values, to quantify how much the proxy departs from
the ideal.

A third "uncalibrated" rule δ_0 = c·(½·||y||²) is added as a sanity floor: it
uses no OLS solve and is the cheapest possible scale, but ignores both X and λ.
"""

import csv
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import deflected_subgradient
from src.data_generation import make_elm_problem
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM, COLOR_REF
apply_style()


SEED      = 42
I_MAX     = 5000
RHO       = 0.7
GAMMA_MIN = 0.05
BETA      = 1.0

# (label, d, p, m, sparsity, lam, noise) — three regimes
INSTANCES = [
    ("easy",       8,  50,  300, 0.1, 0.1, 0.05),
    ("moderate",  12, 100, 1000, 0.1, 0.1, 0.10),
    ("hard",      16, 200, 1500, 0.2, 0.05, 0.20),
]
C_GRID = [0.05, 0.1, 0.5, 1.0]

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _run_sgptl(X, y, lam, delta0, f_star):
    res = deflected_subgradient(
        X, y, lam, w0=np.zeros(X.shape[1]),
        i_max=I_MAX, beta=BETA, delta0=delta0, R=1.0, rho=RHO,
        gamma_min=GAMMA_MIN, f_star=f_star,
    )
    return float(res["f_bar"][-1]) - f_star


def run() -> None:
    print("=" * 70)
    print("δ_0 proxy validation: c·f^* vs c·f(w_OLS) vs c·(½||y||²)")
    print("=" * 70)

    rows = []
    for label, d, p, m, sp, lam, noise in INSTANCES:
        X_raw, X, y, _, _, f_star, _ = make_elm_problem(
            d=d, p=p, m=m, sparsity=sp, noise_std=noise,
            lam=lam, random_state=SEED,
        )
        w_ols = solve_spd(X.T @ X + 1e-10 * np.eye(p), X.T @ y, method="cholesky")
        f_w_ols = float(f_lasso(X, y, w_ols, lam))
        scale_y = float(0.5 * (y @ y))  # f(w=0) = ½||y||² + 0

        print(f"\n[{label}] d={d}, p={p}, m={m}, sparsity={sp}, lam={lam}")
        print(f"   f^* = {f_star:.4e}, f(w_OLS) = {f_w_ols:.4e}, "
              f"½||y||² = {scale_y:.4e}")
        print(f"   gap(OLS, f^*) = {f_w_ols - f_star:.3e}  "
              f"(relative: {(f_w_ols - f_star)/max(f_star, 1e-12):.2%})")

        print(f"   {'c':>6}  {'gap_ideal':>12}  {'gap_proxy':>12}  "
              f"{'gap_naive':>12}  {'proxy/ideal':>12}")
        for c in C_GRID:
            gap_ideal = _run_sgptl(X, y, lam, c * f_star,  f_star)
            gap_proxy = _run_sgptl(X, y, lam, c * f_w_ols, f_star)
            gap_naive = _run_sgptl(X, y, lam, c * scale_y, f_star)
            ratio = gap_proxy / max(gap_ideal, 1e-30)
            print(f"   {c:>6.2f}  {gap_ideal:>12.3e}  {gap_proxy:>12.3e}  "
                  f"{gap_naive:>12.3e}  {ratio:>12.3e}")
            rows.append({
                "instance": label, "c": c, "f_star": f_star, "f_w_ols": f_w_ols,
                "gap_ideal": gap_ideal, "gap_proxy": gap_proxy,
                "gap_naive": gap_naive, "ratio_proxy_ideal": ratio,
            })

    tab_path = os.path.join(TAB_DIR, "delta0_proxy.csv")
    with open(tab_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nSaved: {tab_path}")

    fig, axes = plt.subplots(1, len(INSTANCES),
                             figsize=(6.0 * len(INSTANCES), 5.0),
                             squeeze=False, sharey=True)
    for ax, (label, *_) in zip(axes[0], INSTANCES):
        block = [r for r in rows if r["instance"] == label]
        cs = [r["c"] for r in block]
        gi = [r["gap_ideal"] for r in block]
        gp = [r["gap_proxy"] for r in block]
        gn = [r["gap_naive"] for r in block]
        ax.loglog(cs, gi, "o-", color=COLOR_REF,  label=r"$\delta_0 = c\,f^{*}$ (ideal)")
        ax.loglog(cs, gp, "s-", color=COLOR_IRLS, label=r"$\delta_0 = c\,f(\mathbf{w}_{\mathrm{OLS}})$ (proxy)")
        ax.loglog(cs, gn, "^--", color=COLOR_DSM, label=r"$\delta_0 = (c/2)\,\|\mathbf{y}\|^{2}$ (naive)")
        ax.set_xlabel(r"$c$")
        ax.set_ylabel(r"Final $\bar f^{\,i_{\max}} - f^{*}$")
        ax.set_title(label)
        ax.legend(fontsize=10, loc="best")
        style_axes(ax)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "delta0_proxy.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    run()
