"""Diagnostic: when does SGPTL actually move on california ELM?

The main delta_0 sweep shows california stalls at the OLS warm start on every
(family, c) pair. Two candidate explanations:

(A) structural -> on column-normalised sigmoid(X * W1) at the budgets we use,
    SGPTL cannot escape the warm start
(B) parametric -> the default lambda=0.1 and OLS warm start put us in a regime
    where the LASSO optimum is close to OLS, leaving SGPTL no room to improve.
    Different lambda or cold start would change this.

This script tests (B) by sweeping lambda and starting point on california ELM
(H=50 and H=200), measuring how far SGPTL moves from its starting f-value and
how it compares to IRLS converged.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

np.seterr(all="ignore")

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from src import deflected_subgradient, irls
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd


SEED = 42
TEST_FRACTION = 0.2
I_MAX = 8000
RHO = 0.7
GAMMA_MIN = 0.05
BETA = 1.0
R_PATIENCE = 1.0


def _load_california_elm(H_hidden, seed=SEED):
    d = fetch_california_housing()
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_test = int(TEST_FRACTION * len(y))
    tr = perm[n_test:]
    X_tr_raw = StandardScaler().fit_transform(X[tr])
    y_tr_raw = y[tr]
    y_tr = (y_tr_raw - y_tr_raw.mean()) / (y_tr_raw.std() or 1.0)

    elm = ELM(d=X_tr_raw.shape[1], p=H_hidden,
              activation="sigmoid", random_state=seed)
    X_hidden = elm.transform(X_tr_raw)
    return X_hidden, y_tr


def _cvxpy_fstar(X, y, lam):
    if not _HAS_CVXPY:
        return None
    M, H = X.shape
    if H * M > 1_500_000:
        return None
    w = cp.Variable(H)
    obj = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm(w, 1))
    try:
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=200_000)
        if prob.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return float(prob.value)
    except Exception:
        return None


def _run_one(X, y, lam, w0, label):
    res = deflected_subgradient(
        X, y, lam, w0=w0.copy(), i_max=I_MAX, beta=BETA,
        delta0=0.1 * float(f_lasso(X, y, w0, lam)),
        R=R_PATIENCE, rho=RHO, gamma_min=GAMMA_MIN, f_star=None,
    )
    f0 = float(f_lasso(X, y, w0, lam))
    f_final = float(res["f_bar"][-1])
    return f0, f_final


def sweep_for_H(H_hidden):
    print(f"\n{'=' * 70}")
    print(f"CALIFORNIA ELM H={H_hidden}: lambda x start sweep")
    print(f"{'=' * 70}")
    X, y = _load_california_elm(H_hidden)
    M, H = X.shape
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    w_cold = np.zeros(H)

    print(f"  Matrix shape: M={M}, H={H}")
    print(f"  cond(X^T X) approx (top-bottom eigenvalue ratio):")
    A = X.T @ X
    eigs = np.linalg.eigvalsh(A)
    print(f"    max eig={eigs[-1]:.3e}, min eig={eigs[0]:.3e}, "
          f"cond={eigs[-1]/max(eigs[0], 1e-30):.3e}\n")

    lams = [0.001, 0.01, 0.1, 1.0, 10.0]
    print(f"{'lam':>8}  {'start':>5}  {'f0':>13}  {'f_star':>13}  "
          f"{'f_final':>13}  {'gap_start':>11}  {'gap_final':>11}  "
          f"{'reduction':>11}")
    print("-" * 110)

    records = []
    for lam in lams:
        # CVXPY f^* for benchmarking, IRLS-converged as fallback
        irls_res = irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-14,
                        k_max=300, solver="cholesky", w0=w_ols)
        f_irls = float(np.min(irls_res["f_vals"]))
        f_cvxpy = _cvxpy_fstar(X, y, lam)
        f_star = f_cvxpy if f_cvxpy is not None else f_irls

        for start_label, w0 in [("warm", w_ols), ("cold", w_cold)]:
            f0, f_final = _run_one(X, y, lam, w0, f"lam={lam} {start_label}")
            gap_start = f0 - f_star
            gap_final = f_final - f_star
            reduction = (gap_start - gap_final) / max(abs(gap_start), 1e-16)
            print(f"  {lam:>6g}  {start_label:>5}  {f0:>13.4e}  "
                  f"{f_star:>13.4e}  {f_final:>13.4e}  "
                  f"{gap_start:>11.3e}  {gap_final:>11.3e}  "
                  f"{reduction:>10.1%}")
            records.append({
                "H": H, "lam": lam, "start": start_label,
                "f0": f0, "f_star": f_star, "f_final": f_final,
                "gap_start": gap_start, "gap_final": gap_final,
                "rel_reduction": reduction,
            })
    return records


if __name__ == "__main__":
    print("Diagnostic: lambda x start sweep on california ELM\n"
          "Goal: does any (lambda, start) pair let SGPTL actually reduce the gap?")
    all_recs = []
    for H_hidden in [50, 200]:
        all_recs.extend(sweep_for_H(H_hidden))

    print("\n" + "=" * 70)
    print("SUMMARY: configurations achieving > 50% gap reduction")
    print("=" * 70)
    good = [r for r in all_recs if r["rel_reduction"] > 0.50]
    if not good:
        print("  NONE. SGPTL fails to reduce the gap by more than 50% on")
        print("  any (lambda, start) tested on california ELM at H in {50, 200}.")
        print("  The stall is therefore structural to this dataset under our")
        print("  current algorithmic defaults (R=1, gamma_min=0.05, rho=0.7,")
        print("  beta=1), not a parametric fluke of lambda=0.1 specifically.")
    else:
        for r in good:
            print(f"  H={r['H']}, lam={r['lam']}, start={r['start']}: "
                  f"reduction={r['rel_reduction']:.1%} "
                  f"(start gap {r['gap_start']:.2e} -> final {r['gap_final']:.2e})")
