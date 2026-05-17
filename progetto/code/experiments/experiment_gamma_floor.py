"""Empirical test of the gamma_min floor on ELM LASSO.

Hypothesis: with gamma_min = 0 the greedy gamma collapses to 0 frequently on
ELM LASSO, beta_i = min(1, gamma_i) = 0 forces num = 0 and the algorithm
enters the skip branch, delta underflows and the record gap stalls.
gamma_min = 0.05 prevents the collapse.

For each (H, M) pair and each seed: run SGPTL with gamma_min in {0, 0.05},
record gamma_hist, skip_hist, delta_hist and gaps, then summarise across
seeds.
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

from src import deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from src.lasso_utils import f_lasso
from _plot_style import apply_style, style_axes, SIZE_DOUBLE
apply_style()


SIZES   = [(50, 150), (100, 300), (200, 600)]
SEEDS   = [42, 7, 123, 2024, 11]
LAMBDA  = 0.10
NOISE   = 0.05
I_MAX   = 4000
RHO     = 0.7

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def run_one(H: int, M: int, seed: int, gamma_min: float) -> dict:
    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE, lam=LAMBDA, random_state=seed,
    )
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    f_w0  = float(f_lasso(X, y, w_ols, LAMBDA))
    res   = deflected_subgradient(
        X, y, LAMBDA, w0=w_ols, i_max=I_MAX, beta=1.0,
        delta0=0.1 * f_w0, rho=RHO, gamma_min=gamma_min, f_star=f_star,
    )
    g = np.asarray(res["gamma_hist"], dtype=float)
    s = np.asarray(res["skip_hist"],  dtype=float)
    d = np.asarray(res["delta_hist"], dtype=float)
    n = max(1, len(g))
    return {
        "H": H, "M": M, "seed": seed, "gamma_min": gamma_min,
        "final_gap": float(res["gaps"][-1]) if len(res["gaps"]) else float("nan"),
        "n_iter": int(res["n_iter"]),
        "n_skips": int(s.sum()),
        "frac_skip": float(s.mean()) if len(s) else float("nan"),
        "gamma_mean": float(g.mean()),
        "gamma_median": float(np.median(g)),
        "frac_gamma_near_zero": float(np.mean(g <= 0.01)),
        "n_contractions": int(np.sum(np.diff(d) < 0)) if len(d) > 1 else 0,
        "final_delta": float(d[-1]),
        # keep arrays for plotting (only on the medium size, seed 42)
        "_gamma_arr": g if (H == 100 and seed == 42) else None,
        "_delta_arr": d if (H == 100 and seed == 42) else None,
        "_gaps_arr":  np.asarray(res["gaps"], dtype=float)
                      if (H == 100 and seed == 42) else None,
    }


def run() -> None:
    print("=" * 64)
    print("Empirical test: gamma_min = 0 vs gamma_min = 0.05 on ELM LASSO")
    print("=" * 64)

    rows = []
    arrays = {}
    for H, M in SIZES:
        for seed in SEEDS:
            for gmin in (0.0, 0.05):
                r = run_one(H, M, seed, gmin)
                rows.append({k: v for k, v in r.items() if not k.startswith("_")})
                if r["_gamma_arr"] is not None:
                    arrays[gmin] = {
                        "gamma": r["_gamma_arr"],
                        "delta": r["_delta_arr"],
                        "gaps":  r["_gaps_arr"],
                    }
                print(f"  H={H:4d} M={M:5d} seed={seed:4d} gmin={gmin:.2f}  "
                      f"gap={r['final_gap']:.3e}  "
                      f"frac_skip={r['frac_skip']:.2%}  "
                      f"gamma~0={r['frac_gamma_near_zero']:.2%}")
        print()

    tab_path = os.path.join(TAB_DIR, "gamma_floor_test.csv")
    with open(tab_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {tab_path}")

    print("\nAggregate (mean over seeds, H=100 M=300):")
    print(f"{'gamma_min':>10}  {'final_gap':>11}  {'frac_skip':>10}  "
          f"{'frac_g<=0.01':>13}  {'contractions':>13}")
    for gmin in (0.0, 0.05):
        sub = [r for r in rows if r["H"] == 100 and r["gamma_min"] == gmin]
        gap = np.mean([r["final_gap"] for r in sub])
        fsk = np.mean([r["frac_skip"] for r in sub])
        fgn = np.mean([r["frac_gamma_near_zero"] for r in sub])
        nco = np.mean([r["n_contractions"] for r in sub])
        print(f"{gmin:>10.2f}  {gap:>11.3e}  {fsk:>9.2%}  "
              f"{fgn:>12.2%}  {nco:>13.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(SIZE_DOUBLE[0] * 1.3, SIZE_DOUBLE[1]))
    colors = {0.0: "#D62728", 0.05: "#1F77B4"}
    labels = {0.0: r"$\gamma_{\min}=0$ (unfloored)",
              0.05: r"$\gamma_{\min}=0.05$ (floored)"}

    ax = axes[0]
    for gmin, arr in arrays.items():
        ax.hist(arr["gamma"], bins=50, range=(0, 1), alpha=0.55,
                color=colors[gmin], label=labels[gmin], density=True)
    ax.set_xlabel(r"$\gamma_i$"); ax.set_ylabel("density")
    ax.set_title(r"Distribution of $\gamma_i$ over iterations")
    ax.legend(); style_axes(ax)

    ax = axes[1]
    for gmin, arr in arrays.items():
        d = arr["delta"]
        d = np.maximum(d, 1e-323)
        ax.semilogy(np.arange(len(d)), d, color=colors[gmin],
                    linewidth=1.8, label=labels[gmin])
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\delta_i$  (log scale)")
    ax.set_title(r"$\delta$ trajectory (single seed)")
    ax.legend(loc="lower left"); style_axes(ax)

    ax = axes[2]
    for gmin, arr in arrays.items():
        g = np.maximum(arr["gaps"], 1e-16)
        ax.loglog(np.arange(1, len(g) + 1), g, color=colors[gmin],
                  linewidth=1.8, label=labels[gmin])
    ax.set_xlabel("iteration  (log)"); ax.set_ylabel(r"$\bar f^i - f^*$  (log)")
    ax.set_title("Record gap")
    ax.legend(loc="upper right"); style_axes(ax)

    fig.suptitle(rf"Floor test on ELM LASSO ($H=100$, $M=300$, $\lambda={LAMBDA}$, "
                 rf"$i_{{\max}}={I_MAX}$, $\rho={RHO}$)", y=1.02, fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "gamma_floor_test.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    run()
