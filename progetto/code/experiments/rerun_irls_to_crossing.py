"""Rerun IRLS on real ELM-LASSO until f - f^* < 1e-6, recording the crossing
wall-time.

t_cross is the IRLS iteration-loop time to reach the relative target. The
one-time OLS warm-start factorisation is excluded here and timed separately by
time_ols_warm_start.py (warm_start_cost.csv). The real-data table reports the
loop time and quotes the factorisation cost alongside it. Mirrors
experiment_sgptl_long_run.py setup.

Writes results/tables/real_data_irls_crossing.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler

np.seterr(all="ignore")

from src.elm import ELM
from src.irls import irls
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd

TARGET: float = 1e-6  # relative target: (f - f*)/|f*| <= TARGET
LAMBDA: float = 0.1
H: int = 200
SEED: int = 42
TEST_FRACTION: float = 0.2
N_REPS: int = 5


def split_scale(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(y))
    n_test = int(TEST_FRACTION * len(y))
    tr = perm[n_test:]
    X_tr, y_tr = X[tr], y[tr].astype(float)
    X_tr = StandardScaler().fit_transform(X_tr)
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) or 1.0
    y_tr = (y_tr - y_mean) / y_std
    return X_tr, y_tr


def build_elm(X_raw: np.ndarray) -> np.ndarray:
    elm = ELM(d=X_raw.shape[1], p=H, activation="sigmoid", lam=LAMBDA, random_state=SEED)
    return elm.transform(X_raw)


def fstar_from_cache(name: str) -> float:
    cache = os.path.join(
        os.path.dirname(__file__), os.pardir, "results",
        "tables", "long_run_cache", f"{name}.npz",
    )
    return float(np.load(cache)["f_star"])


def time_irls_full(X: np.ndarray, y: np.ndarray, fstar: float) -> dict:
    """Wall-time of (OLS warm-start + IRLS loop) until the RELATIVE gap
    (f - f*)/|f*| <= TARGET.

    IRLS converges far below relative 1e-6 on both datasets, so two things
    get reported:
        - wall-time to first reach relative 1e-6 (the crossing the table
          needs), from the same warm-start + iteration loop
        - iteration count and relative gap at full convergence, both below
          the target
    """
    rel_thresh = TARGET * abs(fstar)
    t_cross_list: list[float] = []
    t_full_list: list[float] = []
    k_cross_list: list[int] = []
    iters: list[int] = []
    rel_final: list[float] = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        A = X.T @ X + 1e-12 * np.eye(X.shape[1])
        b = X.T @ y
        w0 = solve_spd(A, b, method="cholesky")
        res = irls(
            X, y, LAMBDA, w0=w0,
            eps_thr=1e-12, eps_stop=1e-14,
            k_max=2000, solver="cholesky", f_star=fstar,
        )
        elapsed_full = time.perf_counter() - t0

        # res["gaps"][k] = f(w_{k+1}) - f*, first iter at/below the rel target
        gaps_arr = np.asarray(res["gaps"], dtype=float)
        mask = gaps_arr <= rel_thresh
        if mask.any():
            k_cross = int(np.argmax(mask))
            # res["times"] is cumulative wall-time inside the IRLS loop
            # the one-time OLS factorisation before the loop is timed
            # separately (warm_start_cost.csv) and excluded here
            t_cross = float(res["times"][k_cross])
        else:
            k_cross = res["n_iter"]
            t_cross = elapsed_full

        f_final = float(f_lasso(X, y, res["w"], LAMBDA))
        rel_gap_final = (f_final - fstar) / abs(fstar)

        t_cross_list.append(t_cross)
        t_full_list.append(elapsed_full)
        k_cross_list.append(k_cross + 1)  # 1-based iter count
        iters.append(res["n_iter"])
        rel_final.append(rel_gap_final)
    return {
        "t_cross_ms": min(t_cross_list) * 1000.0,
        "t_full_ms": min(t_full_list) * 1000.0,
        "k_cross": int(np.median(k_cross_list)),
        "n_iter": int(np.median(iters)),
        "rel_final_gap": float(np.median(rel_final)),
    }


def main() -> None:
    rows = []
    for name in ["diabetes", "california"]:
        print(f"--- {name} ---", flush=True)
        d = load_diabetes() if name == "diabetes" else fetch_california_housing()
        X_tr, y_tr = split_scale(d.data, d.target)
        X = build_elm(X_tr)
        fstar = fstar_from_cache(name)
        print(f"  M={X.shape[0]}, H={X.shape[1]}, f*={fstar:.6f}", flush=True)

        out = time_irls_full(X, y_tr, fstar)
        out["dataset"] = name
        rows.append(out)
        for k, v in out.items():
            print(f"  {k}: {v}", flush=True)
        print(flush=True)

    out_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "results", "tables", "real_data_irls_crossing.csv",
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
