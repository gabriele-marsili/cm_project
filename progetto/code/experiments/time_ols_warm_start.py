"""Measure OLS warm-start cost vs total IRLS cost on real data.

For P3 of project_review/REVIEW.md: the prof asks the warm-start cost be
quantified relative to the total. Mirrors the setup of
experiment_sgptl_long_run.py (SEED=42, H=200, lam=0.1, ELM with sigmoid).
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
from src.linear_solvers import solve_spd

SEED: int = 42
H: int = 200
LAMBDA: float = 0.1
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


def time_ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """One Cholesky-based OLS solve. Returns (best_time_s, w_ols)."""
    A = X.T @ X + 1e-12 * np.eye(X.shape[1])
    b = X.T @ y
    times: list[float] = []
    w = None
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        w = solve_spd(A, b, method="cholesky")
        times.append(time.perf_counter() - t0)
    return min(times), w


def time_full_setup(X: np.ndarray, y: np.ndarray) -> float:
    """The whole 'form A=X^T X, b=X^T y, factorise' that the warm start pays."""
    times: list[float] = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        A = X.T @ X + 1e-12 * np.eye(X.shape[1])
        b = X.T @ y
        _ = solve_spd(A, b, method="cholesky")
        times.append(time.perf_counter() - t0)
    return min(times)


def time_irls(X: np.ndarray, y: np.ndarray, w0: np.ndarray) -> tuple[float, int]:
    """IRLS to convergence with the same defaults the report uses."""
    times: list[float] = []
    n_iter = 0
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        res = irls(
            X, y, LAMBDA,
            w0=w0.copy(), eps_thr=1e-8, eps_stop=1e-14,
            k_max=2000, solver="cholesky",
        )
        times.append(time.perf_counter() - t0)
        n_iter = res["n_iter"]
    return min(times), n_iter


def main() -> None:
    rows = []
    for name in ["diabetes", "california"]:
        print(f"--- {name} ---", flush=True)
        d = load_diabetes() if name == "diabetes" else fetch_california_housing()
        X_tr, y_tr = split_scale(d.data, d.target)
        X = build_elm(X_tr)
        M, H_ = X.shape
        print(f"  M={M}, H={H_}", flush=True)

        t_setup = time_full_setup(X, y_tr)
        t_ols, w_ols = time_ols(X, y_tr)
        t_irls_warm, niter_warm = time_irls(X, y_tr, w_ols)
        t_irls_cold, niter_cold = time_irls(X, y_tr, np.zeros(H_))

        # share of warm-start setup vs (setup + IRLS iterations excluding setup)
        # IRLS reports total time including the inner factorisations of the
        # WLS subproblems; t_setup is the one-shot OLS work that precedes them.
        total_warm = t_setup + t_irls_warm
        share_warm = 100.0 * t_setup / total_warm

        row = {
            "dataset": name, "M": M, "H": H_,
            "t_ols_setup_ms": t_setup * 1000.0,
            "t_irls_warm_ms": t_irls_warm * 1000.0,
            "t_irls_cold_ms": t_irls_cold * 1000.0,
            "iter_irls_warm": niter_warm,
            "iter_irls_cold": niter_cold,
            "total_warm_ms": total_warm * 1000.0,
            "share_warm_pct": share_warm,
        }
        rows.append(row)
        for k, v in row.items():
            print(f"  {k}: {v}", flush=True)
        print(flush=True)

    out_path = os.path.join(
        os.path.dirname(__file__), os.pardir,
        "results", "tables", "warm_start_cost.csv",
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
