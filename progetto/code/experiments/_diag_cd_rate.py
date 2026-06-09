"""Diagnostic: empirical per-iteration contraction of sklearn coordinate
descent on the two ELM-LASSO instances.

Checks whether the slower CD convergence on california (relative gap ~2.5e-5
at 1e6 sweeps) vs diabetes (machine precision by ~1e6) tracks california's
larger condition number. Measures the contraction factor
rho_per_iter = (gap_{k+1}/gap_k)^(1/di) and the implied iterations-per-decade
in a matched, floor-isolated window.

Not a report figure. Writes results/tables/cd_rate_diag.csv and prints the
analysis. Run from progetto/code with the project venv.
"""
import os
import sys
import csv
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.linear_model import Lasso as SkLasso

from src import irls
from src.lasso_utils import f_lasso
from experiment_real_data import (load_dataset, build_hidden,
                                   ols_warm_start, LAMBDA)

FLOOR = 1e-13
# diabetes is cheap per sweep (M=354) -> cover its full linear drop
# california (M=16512) capped at the figure budget
BUDGETS = {"diabetes": 2_000_000, "california": 1_000_000}
N_SAMPLES = 70


def fstar_irls(X, y, lam):
    """Tight IRLS reference f*, same family the figure uses"""
    res = irls(X, y, lam, eps_thr=1e-14, eps_stop=1e-16, k_max=3000,
               solver="cholesky", w0=ols_warm_start(X, y))
    return float(np.min(res["f_vals"]))


def cd_trace(X, y, lam, budget, n=N_SAMPLES, tol=1e-16):
    M = X.shape[0]
    iters = np.unique(np.geomspace(1.0, budget, n).astype(int))
    fvals = np.empty(len(iters))
    sk = SkLasso(alpha=lam / M, fit_intercept=False,
                 warm_start=True, max_iter=1, tol=tol)
    prev = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, m in enumerate(iters):
            sk.set_params(max_iter=int(m) - prev)
            sk.fit(X, y)
            fvals[k] = f_lasso(X, y, sk.coef_, lam)
            prev = int(m)
    return iters, fvals


def analyse(name):
    print(f"\n{'='*64}\n{name}\n{'='*64}", flush=True)
    Xr, Xer, ytr, yte = load_dataset(name)
    X, _ = build_hidden(Xr, Xer, Xr.shape[1])
    cond = float(np.linalg.cond(X.T @ X))
    fstar = fstar_irls(X, ytr, LAMBDA)
    print(f"  M={X.shape[0]}, H={X.shape[1]}, cond(X^T X)={cond:.3e}, "
          f"f*={fstar:.6f}", flush=True)

    it, fv = cd_trace(X, ytr, LAMBDA, BUDGETS[name])
    gap = np.maximum((fv - fstar) / abs(fstar), FLOOR)

    # per-iteration contraction between consecutive geomspace samples
    di = np.diff(it).astype(float)
    ratio = gap[1:] / gap[:-1]
    rho = ratio ** (1.0 / di)  # per-iteration contraction
    mid_i = np.sqrt(it[1:] * it[:-1])  # geometric midpoint of interval
    # iters to cut the gap 10x at this local rate
    with np.errstate(divide="ignore", invalid="ignore"):
        iters_per_decade = -1.0 / np.log10(rho)

    rows = []
    print("   i_mid        gap          rho/iter    iters/decade", flush=True)
    for im, g0, r, ipd in zip(mid_i, gap[:-1], rho, iters_per_decade):
        flag = "  (floor)" if g0 <= 10 * FLOOR else ""
        print(f"  {im:11.3e}  {g0:11.3e}  {r:9.6f}   {ipd:11.3e}{flag}",
              flush=True)
        rows.append((name, im, g0, r, ipd))
    return rows, cond, gap, it, fstar


def main():
    all_rows = []
    summary = {}
    for name in ("diabetes", "california"):
        rows, cond, gap, it, fstar = analyse(name)
        all_rows.extend(rows)
        # stabilised-window estimate: intervals above the floor and past the
        # early transient (i > 1e3), where local rho is most representative
        sel = [(im, r) for (_, im, g0, r, _ipd) in rows
               if g0 > 1e3 * FLOOR and im > 1e3 and r < 1.0]
        if sel:
            rr = np.array([r for _, r in sel])
            # slowest (closest-to-1) stabilised tail -> rate proxy
            tail = rr[-5:] if len(rr) >= 5 else rr
            rho_tail = float(np.median(tail))
            ipd = -1.0 / np.log10(rho_tail)
            summary[name] = (cond, rho_tail, ipd)
    print(f"\n{'='*64}\nSUMMARY (floor-isolated, i>1e3 tail)\n{'='*64}",
          flush=True)
    print("  dataset      cond(X^TX)   median rho/iter   iters/decade",
          flush=True)
    for name, (cond, rho_t, ipd) in summary.items():
        print(f"  {name:11}  {cond:.3e}    {rho_t:.7f}        {ipd:.3e}",
              flush=True)

    tab = os.path.join(os.path.dirname(__file__), os.pardir,
                       "results", "tables", "cd_rate_diag.csv")
    with open(tab, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "i_mid", "gap", "rho_per_iter",
                     "iters_per_decade"])
        wr.writerows(all_rows)
    print(f"\nSaved: {tab}", flush=True)


if __name__ == "__main__":
    main()
