"""Playground: theory-pure SGPTL with warm (OLS) vs cold (w_0=0) start.

The theory of Lemma 3.8 in d'Antonio-Frangioni 2009 (Theorem 3.1 in the
report) does not depend on the starting point. In practice, however, the
combination of OLS warm start + Polyak target level produces pathological
behavior on ELM LASSO: the warm-started w_0 is close to w^*, the Polyak
numerator is dominated by delta_0 rather than by the true gap, and either
(i) delta_0 large -> first step overshoots and the iterate wanders, or
(ii) delta_0 small -> step lengths are too small for r to ever reach R, so
delta is never contracted.

This script documents the phenomenon on a synthetic instance, comparing the
record-value trajectory for warm vs cold start with the same algorithm.
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

from src import deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_DSM, COLOR_FCUR, SIZE_DOUBLE)
apply_style()


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05
H, M   = 100, 300
I_MAX  = 8000

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _n_contractions(delta_hist):
    """Count how many times delta was contracted (strictly decreasing step)."""
    d = np.asarray(delta_hist, dtype=float)
    return int(np.sum(np.diff(d) < 0))


def run() -> None:
    print("=" * 60)
    print("SGPTL playground: warm (OLS) vs cold (w_0=0) start")
    print("(theory-pure algorithm: no iter-fallback, no overshoot rescue)")
    print("=" * 60)

    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")

    w_ols  = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    w_cold = np.zeros(H)
    from src.lasso_utils import f_lasso
    f_w0_warm = float(f_lasso(X, y, w_ols,  LAMBDA))
    f_w0_cold = float(f_lasso(X, y, w_cold, LAMBDA))
    print(f"f(w_0) warm = {f_w0_warm:.4f}, cold = {f_w0_cold:.4f}, f* = {f_star:.4f}")

    runs = {
        "warm":  deflected_subgradient(
            X, y, LAMBDA, w0=w_ols, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_w0_warm, rho=0.7, gamma_min=0.05, f_star=f_star),
        "cold":  deflected_subgradient(
            X, y, LAMBDA, w0=w_cold, i_max=I_MAX, beta=1.0,
            delta0=0.1 * f_w0_cold, rho=0.7, gamma_min=0.05, f_star=f_star),
    }

    abs_f_star = abs(f_star)
    print(f"\n{'config':>8}  {'final abs gap':>13}  {'final rel gap':>13}  "
          f"{'#delta-contractions':>22}  {'frac gamma <=0.06':>18}")
    for key, res in runs.items():
        g = np.asarray(res["gamma_hist"], dtype=float)
        frac = float(np.mean(g <= 0.06)) if g.size else float("nan")
        n_contr = _n_contractions(res["delta_hist"])
        abs_gap = float(res["gaps"][-1])
        rel_gap = abs_gap / abs_f_star
        print(f"  {key:>6}  {abs_gap:>13.3e}  {rel_gap:>13.3e}  "
              f"{n_contr:>22d}  {frac:>17.2%}")
    print(f"\n[synthetic SGPTL] f* = {f_star:.6f}  (relative gap = (f - f*)/|f*|)")
    for key, res in runs.items():
        rel_gap = float(res["gaps"][-1]) / abs_f_star
        print(f"  SGPTL {key:>4} synthetic: relative final gap = {rel_gap:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    # Panel 1: record-gap trajectories (log-log).
    ax = axes[0]
    for key, color, label in (
        ("warm", COLOR_DSM,  "warm start (OLS)"),
        ("cold", COLOR_FCUR, r"cold start ($w_0=0$)"),
    ):
        g = np.maximum(np.asarray(runs[key]["gaps"], dtype=float) / abs_f_star,
                       1e-16)
        ax.loglog(np.arange(1, len(g) + 1), g,
                  color=color, linewidth=2.0, label=label)
        ax.scatter([len(g)], [g[-1]], s=40, color=color, zorder=5)
        ax.annotate(f"{g[-1]:.2e}", xy=(len(g), g[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=10, color=color, ha="left", va="center")
    ax.set_xlabel("Iteration  (log scale)")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
    ax.set_title("Record gap")
    ax.legend(loc="lower left")
    style_axes(ax)

    # Panel 2: delta history (linear-log).
    ax = axes[1]
    for key, color, label in (
        ("warm", COLOR_DSM,  "warm start (OLS)"),
        ("cold", COLOR_FCUR, r"cold start ($w_0=0$)"),
    ):
        d = np.asarray(runs[key]["delta_hist"], dtype=float)
        ax.semilogy(np.arange(len(d)), d, color=color, linewidth=2.0,
                    label=f"{label}  ({_n_contractions(d)} contr.)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\delta_{i}$  (log scale)")
    ax.set_title(r"Target gap $\delta$ over iterations")
    ax.legend(loc="lower left")
    style_axes(ax)

    fig.suptitle(rf"Theory-pure SGPTL: warm vs cold start on $H={H}$, $M={M}$, "
                 rf"$\lambda={LAMBDA}$, $i_{{\max}}={I_MAX}$",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "warm_vs_cold.pdf")
    fig.savefig(path)
    print(f"\nSaved: {path}")
    plt.close(fig)

    # --- Save SGPTL synthetic warm/cold rows of Table 5.2 to CSV ---
    csv_path = os.path.join(TAB_DIR, "warm_cold_sgptl_synthetic.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["start", "H", "M", "i_max", "n_contractions", "rel_final_gap"])
        for key in ("warm", "cold"):
            res = runs[key]
            w.writerow([key, H, M, I_MAX, _n_contractions(res["delta_hist"]),
                        f"{float(res['gaps'][-1]) / abs_f_star:.6e}"])
    print(f"Saved SGPTL synthetic warm/cold CSV: {csv_path}")


if __name__ == "__main__":
    run()
