"""Rerun SGPTL on real ELM-LASSO until record gap to f^* drops below 1e-6.

Measures t_cross directly from the internal times[] array of
deflected_subgradient, instead of scaling it linearly from the long-run total.

Mirrors experiment_sgptl_long_run.py setup verbatim (SEED=42, H=200, lam=0.1,
StandardScaler split, ELM with sigmoid hidden layer, same f* proxy).

Writes results/tables/real_data_crossing.csv.
"""
from __future__ import annotations

import csv
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler

np.seterr(all="ignore")

from src.deflected_subgradient import deflected_subgradient
from src.elm import ELM
from src.lasso_utils import f_lasso

TARGET: float = 1e-6  # relative target: (f - f*)/|f*| <= TARGET
LAMBDA: float = 0.1
H: int = 200
SEED: int = 42
TEST_FRACTION: float = 0.2

# relative-1e-6 crossing from long_run_cache is ~56k (diabetes), ~298k
# (california); budgets give a comfortable safety margin above each.
BUDGETS: dict[str, int] = {"diabetes": 90_000, "california": 360_000}


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


def load_problem(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = load_diabetes() if name == "diabetes" else fetch_california_housing()
    X_tr_raw, y_tr = split_scale(d.data, d.target)
    return build_elm(X_tr_raw), y_tr


def fstar_from_cache(name: str) -> float:
    cache = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "results",
        "tables",
        "long_run_cache",
        f"{name}.npz",
    )
    return float(np.load(cache)["f_star"])


def run_until_crossing(name: str) -> dict:
    print(f"--- {name} ---", flush=True)
    X, y = load_problem(name)
    fstar = fstar_from_cache(name)
    print(f"f*={fstar:.10f}, X={X.shape}, lam={LAMBDA}", flush=True)
    print(f"running up to {BUDGETS[name]} iter...", flush=True)

    f_w0 = float(f_lasso(X, y, np.zeros(X.shape[1]), LAMBDA))
    delta0 = 0.1 * f_w0
    result = deflected_subgradient(
        X, y, lam=LAMBDA, w0=None,
        i_max=BUDGETS[name], beta=1.0, delta0=delta0,
        R=1.0, rho=0.7, f_star=fstar, gamma_min=0.05,
        verbose=False,
    )
    gaps = np.asarray(result["gaps"], dtype=float)
    times = np.asarray(result["times"], dtype=float)
    # relative crossing: (f - f*)/|f*| <= TARGET  <=>  gap <= TARGET*|f*|
    rel_thresh = TARGET * abs(fstar)
    mask = gaps <= rel_thresh
    if mask.any():
        k_cross = int(np.argmax(mask))
        t_cross = float(times[k_cross])
        gap_cross = float(gaps[k_cross])
        print(
            f"  CROSSED (relative) at k={k_cross}, gap={gap_cross:.3e}, "
            f"rel={gap_cross/abs(fstar):.3e}, "
            f"t={t_cross*1000:.1f} ms ({t_cross:.2f} s)",
            flush=True,
        )
    else:
        k_cross = -1
        t_cross = float(times[-1])
        gap_cross = float(gaps[-1])
        print(
            f"  NOT crossed: final gap={gap_cross:.3e} at k={len(gaps)-1}, "
            f"t={t_cross:.1f}s",
            flush=True,
        )
    return {
        "dataset": name,
        "k_cross": k_cross,
        "t_cross_s": t_cross,
        "t_cross_ms": t_cross * 1000.0,
        "gap_at_cross": gap_cross,
        "rel_gap_at_cross": gap_cross / abs(fstar),
        "fstar": fstar,
        "i_max_used": BUDGETS[name],
    }


def main() -> None:
    out_path = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        "results",
        "tables",
        "real_data_crossing.csv",
    )
    rows = [run_until_crossing(name) for name in ["diabetes", "california"]]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
