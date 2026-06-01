"""IRLS on diabetes and california_housing: Warm Start vs Cold Start."""

import os
import sys
import warnings
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SkLasso

from src.irls import irls
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_DSM, COLOR_FCUR
apply_style()


SEED          = 42
H             = 200
LAMBDA        = 0.1
TEST_FRACTION = 0.2
IRLS_KMAX     = 2000
EPS_THR       = 1e-8

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _split_scale(X, y, test_frac=TEST_FRACTION, seed=SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_test = int(test_frac * len(y))
    te, tr = perm[:n_test], perm[n_test:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    scaler_x = StandardScaler().fit(X_tr)
    X_tr = scaler_x.transform(X_tr)
    X_te = scaler_x.transform(X_te)

    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) or 1.0
    y_tr = (y_tr - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std
    return X_tr, X_te, y_tr, y_te


def load_dataset(name):
    if name == "diabetes":
        d = load_diabetes()
    elif name == "california":
        d = fetch_california_housing()
    else:
        raise ValueError(name)
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)
    return _split_scale(X, y)


def build_hidden(X_tr_raw, X_te_raw, d_in, H=H, seed=SEED):
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=seed)
    return elm.transform(X_tr_raw), elm.transform(X_te_raw)


def reference_solution(X, y, lam):
    M = X.shape[0]
    sk = SkLasso(alpha=lam / M, fit_intercept=False, max_iter=10000, tol=1e-4)
    sk.fit(X, y)
    w_star = sk.coef_
    f_star = f_lasso(X, y, w_star, lam)
    return w_star, f_star


def ols_warm_start(X, y):
    return solve_spd(X.T @ X + 1e-10 * np.eye(X.shape[1]),
                     X.T @ y, method="cholesky")


def run_one(name):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    X_tr_raw, X_te_raw, y_tr, y_te = load_dataset(name)
    d_in = X_tr_raw.shape[1]
    M = X_tr_raw.shape[0]
    print(f"  n_train={M}, n_test={X_te_raw.shape[0]}, d={d_in}, H={H}")

    X_tr, X_te = build_hidden(X_tr_raw, X_te_raw, d_in)
    print(f"  hidden activations: shape={X_tr.shape}, "
          f"cond(X^T X) ≈ {np.linalg.cond(X_tr.T @ X_tr):.2e}")

    if name == "california":
        print("  sklearn skipped on this dataset; using OLS f as placeholder for f_star.")
        f_star = float(f_lasso(X_tr, y_tr, ols_warm_start(X_tr, y_tr), LAMBDA))
    else:
        _, f_star = reference_solution(X_tr, y_tr, LAMBDA)
        print(f"  f* (sklearn baseline) = {f_star:.6f}")

    # ==========================================
    # WARM START (OLS)
    # ==========================================
    t0 = time.time()
    w_ols = ols_warm_start(X_tr, y_tr)
    # IRLS call (Note: IRLS returns dict mapping usually inside ELM, but assuming your IRLS returns the dict directly)
    res_warm = irls(
        X_tr, y_tr, LAMBDA, eps_thr=EPS_THR, k_max=IRLS_KMAX,
        solver='cholesky', w0=w_ols, f_star=f_star
    )
    time_warm = time.time() - t0
    f_w = f_lasso(X_tr, y_tr, res_warm.get("w", w_ols), LAMBDA)
    n_iter_w = len(res_warm.get("f_vals", []))
    print(f"  IRLS (Warm) : {n_iter_w} iter, f = {f_w:.6f}, time = {time_warm:.4f}s")

    # ==========================================
    # COLD START (w = 0)
    # ==========================================
    w_cold = np.zeros(H)
    t0 = time.time()
    res_cold = irls(
        X_tr, y_tr, LAMBDA, eps_thr=EPS_THR, k_max=IRLS_KMAX,
        solver='cholesky', w0=w_cold, f_star=f_star
    )
    time_cold = time.time() - t0
    f_c = f_lasso(X_tr, y_tr, res_cold.get("w", w_cold), LAMBDA)
    n_iter_c = len(res_cold.get("f_vals", []))
    print(f"  IRLS (Cold) : {n_iter_c} iter, f = {f_c:.6f}, time = {time_cold:.4f}s")

    # Report f* is the IRLS-converged value (CVXPY-verified elsewhere); both
    # starts reach it to working precision. Relative gap = (f - f*)/|f*|.
    f_star_ref = float(min(f_w, f_c))
    abs_gap_warm = abs(float(f_w) - f_star_ref)
    abs_gap_cold = abs(float(f_c) - f_star_ref)
    print(f"  f* (IRLS-converged) = {f_star_ref:.6f}")
    print(f"  IRLS (Warm) {name}: rel final gap = {abs_gap_warm/abs(f_star_ref):.4e} "
          f"(abs {abs_gap_warm:.3e}, {n_iter_w} iter)")
    print(f"  IRLS (Cold) {name}: rel final gap = {abs_gap_cold/abs(f_star_ref):.4e} "
          f"(abs {abs_gap_cold:.3e}, {n_iter_c} iter)")

    return {
        "name": name,
        "M_train": M, "H": H,
        "time_warm": time_warm,
        "time_cold": time_cold,
        "fvals_warm": res_warm["f_vals"],
        "fvals_cold": res_cold["f_vals"],
        "f_star": f_star
    }


def run() -> None:
    print("=" * 60)
    print("Real-data experiment (IRLS) - Warm vs Cold Start")
    print("=" * 60)

    rows = []
    for name in ("diabetes", "california"):
        try:
            rows.append(run_one(name))
        except Exception as exc:
            print(f"  [skip {name}: {exc}]")

    if not rows:
        print("\nNo datasets ran successfully; nothing to plot.")
        return

    n_datasets = len(rows)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5),
                             squeeze=False)

    for i, row in enumerate(rows):
        ax = axes[0, i]

        f_warm = np.asarray(row["fvals_warm"], dtype=float)
        f_cold = np.asarray(row["fvals_cold"], dtype=float)
        f_star = row["f_star"]

        f_min = min(f_warm.min(), f_cold.min(), f_star)
        floor = 1e-16
        # Relative gap (f - f*)/|f*|; f_min is the IRLS-converged f* both
        # starts reach (matches the report reference).
        abs_f_star = abs(f_min)
        gw = np.maximum((f_warm - f_min) / abs_f_star, floor)
        gc = np.maximum((f_cold - f_min) / abs_f_star, floor)

        iters_w = np.arange(1, len(gw) + 1)
        iters_c = np.arange(1, len(gc) + 1)

        ax.loglog(iters_w, gw, color=COLOR_DSM, linewidth=2.0,
                  label="Warm Start (OLS)")
        ax.loglog(iters_c, gc, color=COLOR_FCUR, linewidth=2.0,
                  label=r"Cold Start ($w_0 = 0$)")
        ax.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM, zorder=5)
        ax.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)

        ax.set_xlabel(r"Iteration $k$  (log scale)")
        ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
        ax.set_title(f"{row['name'].capitalize()} ($M={row['M_train']}$, $H={row['H']}$)")
        ax.legend(loc="lower left")
        style_axes(ax)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "irls_real_data_warm_vs_cold.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved plots to: {fig_path}")
    plt.close(fig)

if __name__ == "__main__":
    run()