"""SGPTL: hyperparameter calibration for warm-start and cold-start separately.

Goal: for each starting point (warm = OLS, cold = zero), sweep
(delta_0, rho, R) on a moderate grid, record the final record gap, and
report the best configuration.

The point is to give a fair side-by-side: if warm or cold start has any
intrinsic advantage on ELM LASSO, it should show up when each is run at
its own best configuration.

All runs use the theory-pure SGPTL (Algorithm 1 in the report, no
iter-fallback, no overshoot rescue), gamma_min = 0.05, beta = 1.
"""

import csv
import itertools
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")

from src import deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from src.lasso_utils import f_lasso


SEED   = 42
LAMBDA = 0.10
NOISE  = 0.05
H, M   = 100, 300
I_MAX  = 8000

# Hyperparameter grid (theory-pure: delta_0 > 0, rho in (0,1), R > 0).
DELTA0_FACTORS = [0.01, 0.05, 0.1, 0.5, 1.0]   # multiplied by f(w_0)
RHO_VALS       = [0.3, 0.5, 0.7, 0.8, 0.9]
R_VALS         = [0.1, 0.5, 1.0, 5.0, 10.0]

TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(TAB_DIR, exist_ok=True)


def _n_contractions(delta_hist):
    d = np.asarray(delta_hist, dtype=float)
    return int(np.sum(np.diff(d) < 0))


def _run_one(X, y, lam, w0, f_w0, delta_factor, rho, R, f_star):
    res = deflected_subgradient(
        X, y, lam,
        w0=w0, i_max=I_MAX, beta=1.0,
        delta0=delta_factor * f_w0, rho=rho, R=R, gamma_min=0.05,
        f_star=f_star,
    )
    return {
        "gap": float(res["gaps"][-1]),
        "n_contr": _n_contractions(res["delta_hist"]),
        "n_iter": int(res["n_iter"]),
    }


def _sweep(name, X, y, lam, w0, f_w0, f_star):
    rows = []
    print(f"\n--- sweep on {name} start (f(w_0) = {f_w0:.4f}) ---")
    for d_f, rho, R in itertools.product(DELTA0_FACTORS, RHO_VALS, R_VALS):
        out = _run_one(X, y, lam, w0, f_w0, d_f, rho, R, f_star)
        rows.append({
            "start":     name,
            "delta_factor": d_f,
            "rho":          rho,
            "R":            R,
            "delta_0":      d_f * f_w0,
            "gap":          out["gap"],
            "n_contractions": out["n_contr"],
            "n_iter":       out["n_iter"],
        })
    rows.sort(key=lambda r: r["gap"])
    best = rows[0]
    print(f"  best: delta_factor={best['delta_factor']}, rho={best['rho']}, "
          f"R={best['R']}: gap={best['gap']:.3e}, #contr={best['n_contractions']}")
    print(f"  top-5:")
    for r in rows[:5]:
        print(f"    delta_factor={r['delta_factor']:5g}, rho={r['rho']:4g}, "
              f"R={r['R']:5g}: gap={r['gap']:.3e}, #contr={r['n_contractions']:3d}")
    return rows


def run() -> None:
    print("=" * 70)
    print("SGPTL: per-start hyperparameter calibration (theory-pure rule)")
    print("=" * 70)

    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}")
    print(f"Grid: delta_factor in {DELTA0_FACTORS}, rho in {RHO_VALS}, "
          f"R in {R_VALS} ({len(DELTA0_FACTORS)*len(RHO_VALS)*len(R_VALS)} configs per start)")

    w_ols  = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    w_cold = np.zeros(H)
    f_w0_warm = float(f_lasso(X, y, w_ols,  LAMBDA))
    f_w0_cold = float(f_lasso(X, y, w_cold, LAMBDA))

    rows_warm = _sweep("warm", X, y, LAMBDA, w_ols, f_w0_warm, f_star)
    rows_cold = _sweep("cold", X, y, LAMBDA, w_cold, f_w0_cold, f_star)

    rows = rows_warm + rows_cold
    out_path = os.path.join(TAB_DIR, "best_configs.csv")
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Headline summary: best of each, side by side.
    print("\n" + "=" * 70)
    print("Best config for each starting point")
    print("=" * 70)
    bw = rows_warm[0]
    bc = rows_cold[0]
    print(f"warm: delta_factor={bw['delta_factor']}, rho={bw['rho']}, R={bw['R']}"
          f"  ->  gap={bw['gap']:.3e}, contractions={bw['n_contractions']}")
    print(f"cold: delta_factor={bc['delta_factor']}, rho={bc['rho']}, R={bc['R']}"
          f"  ->  gap={bc['gap']:.3e}, contractions={bc['n_contractions']}")
    if bw["gap"] < bc["gap"]:
        ratio = bc["gap"] / bw["gap"]
        winner = "warm"
    else:
        ratio = bw["gap"] / bc["gap"]
        winner = "cold"
    print(f"\nWinner at best config: {winner}  (factor {ratio:.1f}x ahead)")


if __name__ == "__main__":
    run()
