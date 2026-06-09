"""Smoke test: sensitivity of SGPTL to rho on small synthetic and real ELM instances.

Used to motivate the rho default in the revised submission. Output goes to stdout
and to results/tables/smoke_rho.csv
"""

import csv
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore")

import numpy as np
np.seterr(all="ignore")

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SkLasso

from src import deflected_subgradient
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from src.data_generation import make_lasso_problem


SEED = 42
RHO_GRID = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
TAB = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(TAB, exist_ok=True)


def _ols(X, y):
    return solve_spd(X.T @ X + 1e-10 * np.eye(X.shape[1]), X.T @ y, method="cholesky")


def _split_scale(X, y, frac=0.2, seed=SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_te = int(frac * len(y))
    te, tr = perm[:n_te], perm[n_te:]
    sx = StandardScaler().fit(X[tr])
    X_tr, X_te = sx.transform(X[tr]), sx.transform(X[te])
    m, s = float(y[tr].mean()), float(y[tr].std()) or 1.0
    return X_tr, X_te, (y[tr] - m) / s, (y[te] - m) / s


def _elm(X_tr_raw, X_te_raw, d_in, H=200, lam=0.1, seed=SEED):
    e = ELM(d=d_in, p=H, activation="sigmoid", lam=lam, random_state=seed)
    return e.transform(X_tr_raw), e.transform(X_te_raw)


def _bench_synthetic(lam=0.1, H=80, M=300):
    print(f"\n--- synthetic (H={H}, M={M}, lam={lam}) ---")
    X, y, _, f_star, _ = make_lasso_problem(n=H, m=M, lam=lam, random_state=SEED)
    w0 = _ols(X, y)
    fw0 = float(f_lasso(X, y, w0, lam))
    print(f"f(w_0)={fw0:.4f}, f*={f_star:.4f}, gap0={fw0-f_star:.3e}")
    rows = []
    for rho in RHO_GRID:
        res = deflected_subgradient(
            X, y, lam, w0=w0, i_max=5000, beta=1.0,
            delta0=0.1 * fw0, rho=rho, gamma_min=0.05, f_star=f_star,
        )
        gap = float(res["f_bar"][-1] - f_star)
        spar = float(np.mean(np.abs(res["w"]) < 1e-3))
        print(f"  rho={rho:.2f}: gap={gap:.3e}, sparsity@1e-3={spar:.0%}")
        rows.append({"dataset": "synthetic", "rho": rho, "gap": gap,
                     "sparsity_1e3": spar, "f_w0": fw0, "f_star": f_star})
    return rows


def _bench_real(name, H=200, lam=0.1, i_max=8000):
    print(f"\n--- {name} (H={H}, lam={lam}) ---")
    if name == "diabetes":
        d = load_diabetes()
    else:
        d = fetch_california_housing()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)
    X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)
    X_tr, X_te = _elm(X_tr_raw, X_te_raw, X.shape[1], H=H, lam=lam)
    w0 = _ols(X_tr, y_tr)
    fw0 = float(f_lasso(X_tr, y_tr, w0, lam))

    # sklearn reference where it is cheap enough to run
    if name == "diabetes":
        sk = SkLasso(alpha=lam / X_tr.shape[0], fit_intercept=False,
                     max_iter=2000, tol=1e-5)
        sk.fit(X_tr, y_tr)
        f_skl = float(f_lasso(X_tr, y_tr, sk.coef_, lam))
    else:
        f_skl = None  # too slow

    print(f"f(w_0)={fw0:.4f}, f_skl={f_skl}")
    rows = []
    for rho in RHO_GRID:
        res = deflected_subgradient(
            X_tr, y_tr, lam, w0=w0, i_max=i_max, beta=1.0,
            delta0=0.1 * fw0, rho=rho, gamma_min=0.05, f_star=None,
        )
        f_best = float(res["f_bar"][-1])
        w_b = res["w"]
        spar = float(np.mean(np.abs(w_b) < 1e-3))
        mse = float(np.mean((y_te - X_te @ w_b) ** 2))
        print(f"  rho={rho:.2f}: f_best={f_best:.4f}, sparsity@1e-3={spar:.0%}, test_mse={mse:.4f}")
        rows.append({"dataset": name, "rho": rho, "gap": f_best,
                     "sparsity_1e3": spar, "f_w0": fw0,
                     "f_star": f_skl if f_skl else f_best, "test_mse": mse})
    return rows


def main():
    all_rows = []
    all_rows += _bench_synthetic()
    all_rows += _bench_real("diabetes")
    all_rows += _bench_real("california")

    path = os.path.join(TAB, "smoke_rho.csv")
    keys = ["dataset", "rho", "gap", "sparsity_1e3", "f_w0", "f_star", "test_mse"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
