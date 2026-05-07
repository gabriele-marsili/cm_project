"""
experiment_real_data.py
-----------------------
Validation of IRLS and SGPTL on real regression datasets piped through the
ELM transformation. Complements the synthetic experiments by checking that:

  1. The optimisation gap f(w_k) - f*  (where f* is sklearn's high-tolerance
     coordinate-descent reference) converges with the same qualitative
     behaviour on real X as on synthetic X — IRLS linear, SGPTL sublinear;
  2. IRLS still produces sparse output weights even though the data does not
     embed a known w_true;
  3. The optimised w generalises: test-set MSE for IRLS is comparable to
     sklearn-Lasso and to a Ridge baseline.

Datasets (all included with sklearn, no external download except California
housing on first run):
  - diabetes   (442 samples, 10 features)         — regression, classic
  - california (~20640 samples, 8 features)       — regression, larger scale

Pipeline per dataset:
  - 80/20 train/test split, fixed seed
  - Standardise features (StandardScaler fit on train only)
  - ELM transform with H=200 sigmoid hidden units (fixed random W1)
  - Solve LASSO on hidden activations: f(w) = (1/2)||X_h w - y||^2 + lam*||w||_1
  - Reference f* and w* from sklearn.linear_model.Lasso (alpha = lam/M)
  - Run IRLS (100 iter, eps_thr=1e-8) and SGPTL (8000 iter, OLS warm start,
    delta0 = 0.1 f*, rho = 0.9)
  - Report: optimisation gap, train MSE, test MSE, sparsity

Outputs:
  - results/figures/real_data_convergence.pdf
  - results/tables/real_data.csv
"""

import csv
import os
import sys
import warnings

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
from sklearn.linear_model import Lasso as SkLasso, Ridge as SkRidge

from src import irls, deflected_subgradient
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM
apply_style()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED          = 42
H             = 200      # hidden-layer size for the ELM
LAMBDA        = 0.1
TEST_FRACTION = 0.2
IRLS_KMAX     = 100
DSM_IMAX      = 8000
DSM_DELTA0    = 0.1
DSM_RHO       = 0.9

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def _split_scale(X, y, test_frac=TEST_FRACTION, seed=SEED):
    """
    Train/test split with feature AND target standardisation. Statistics
    are fit on the training set only. Standardising y to unit variance
    makes lambda comparable across datasets and keeps f* on the same
    order of magnitude as in the synthetic experiments.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_test = int(test_frac * len(y))
    te, tr = idx[:n_test], idx[n_test:]
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


# ---------------------------------------------------------------------------
# ELM hidden-feature pipeline + solvers
# ---------------------------------------------------------------------------
def build_hidden(X_train_raw, X_test_raw, d_in, H=H, seed=SEED):
    """Apply a fixed random ELM projection sigma(X W1^T)."""
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=seed)
    X_h_tr = elm.transform(X_train_raw)
    X_h_te = elm.transform(X_test_raw)
    return X_h_tr, X_h_te


def reference_solution(X, y, lam):
    """High-tolerance sklearn LASSO reference. Returns (w_star, f_star)."""
    m = X.shape[0]
    sk = SkLasso(alpha=lam / m, fit_intercept=False,
                 max_iter=100000, tol=1e-12)
    sk.fit(X, y)
    w_star = sk.coef_
    f_star = f_lasso(X, y, w_star, lam)
    return w_star, f_star


def ols_warm_start(X, y):
    """w_ols = (X^T X + eps I)^-1 X^T y via Cholesky."""
    A = X.T @ X
    b = X.T @ y
    return solve_spd(A + 1e-10 * np.eye(X.shape[1]), b, method="cholesky")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _sparsity(w, tol=1e-6):
    return float(np.mean(np.abs(w) < tol))


def run_one(name):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    X_tr_raw, X_te_raw, y_tr, y_te = load_dataset(name)
    d_in = X_tr_raw.shape[1]
    M = X_tr_raw.shape[0]
    print(f"  n_train={M}, n_test={X_te_raw.shape[0]}, d={d_in}, H={H}")

    X_tr, X_te = build_hidden(X_tr_raw, X_te_raw, d_in)
    print(f"  hidden activations: shape={X_tr.shape}, "
          f"cond(X^T X) ≈ {np.linalg.cond(X_tr.T @ X_tr):.2e}")

    # Reference
    w_star, f_star = reference_solution(X_tr, y_tr, LAMBDA)
    print(f"  f* (sklearn) = {f_star:.6f}, "
          f"sklearn sparsity = {_sparsity(w_star):.0%}, "
          f"sklearn test MSE = {_mse(y_te, X_te @ w_star):.4f}")

    # OLS warm start shared by both algorithms
    w_ols = ols_warm_start(X_tr, y_tr)

    # IRLS
    res_i = irls(X_tr, y_tr, LAMBDA,
                 eps_thr=1e-8, eps_stop=1e-12,
                 k_max=IRLS_KMAX, solver="cholesky",
                 w0=w_ols, f_star=f_star)
    w_i = res_i["w"]
    f_i = f_lasso(X_tr, y_tr, w_i, LAMBDA)
    note_i = " (matches sklearn precision)" if res_i["gaps"][-1] == 0.0 else ""
    print(f"  IRLS : {res_i['n_iter']} iter, "
          f"gap = {res_i['gaps'][-1]:.3e}{note_i}, f = {f_i:.6f}, "
          f"sparsity = {_sparsity(w_i):.0%}, "
          f"test MSE = {_mse(y_te, X_te @ w_i):.4f}")

    # SGPTL
    res_d = deflected_subgradient(X_tr, y_tr, LAMBDA,
                                  w0=w_ols, i_max=DSM_IMAX, beta=1.0,
                                  delta0=DSM_DELTA0 * f_star, rho=DSM_RHO,
                                  f_star=f_star)
    w_d = res_d["w"]
    f_d = f_lasso(X_tr, y_tr, w_d, LAMBDA)
    note_d = " (matches sklearn precision)" if res_d["gaps"][-1] == 0.0 else ""
    print(f"  SGPTL: {res_d['n_iter']} iter, "
          f"record gap = {res_d['gaps'][-1]:.3e}{note_d}, f = {f_d:.6f}, "
          f"sparsity = {_sparsity(w_d):.0%}, "
          f"test MSE = {_mse(y_te, X_te @ w_d):.4f}")

    # Ridge baseline (closed form, alpha matched so that the L2 weight
    # equals lam in our objective)
    A_ridge = X_tr.T @ X_tr + LAMBDA * np.eye(X_tr.shape[1])
    w_ridge = solve_spd(A_ridge, X_tr.T @ y_tr, method="cholesky")
    print(f"  Ridge: closed form, "
          f"test MSE = {_mse(y_te, X_te @ w_ridge):.4f}")

    return {
        "name": name,
        "M_train": M, "M_test": X_te_raw.shape[0], "d": d_in, "H": H,
        "f_star": f_star,
        "f_irls": f_i,
        "f_dsm":  f_d,
        "gap_irls": res_i["gaps"][-1],
        "gap_dsm":  res_d["gaps"][-1],
        "iter_irls": res_i["n_iter"],
        "iter_dsm":  res_d["n_iter"],
        "spar_skl":  _sparsity(w_star),
        "spar_irls": _sparsity(w_i),
        "spar_dsm":  _sparsity(w_d),
        "mse_skl":   _mse(y_te, X_te @ w_star),
        "mse_irls":  _mse(y_te, X_te @ w_i),
        "mse_dsm":   _mse(y_te, X_te @ w_d),
        "mse_ridge": _mse(y_te, X_te @ w_ridge),
        "_irls_gaps": res_i["gaps"],
        "_dsm_gaps":  res_d["gaps"],
    }


def run() -> None:
    print("=" * 60)
    print("Real-data experiment (ELM + LASSO)")
    print("=" * 60)

    rows = []
    for name in ("diabetes", "california"):
        try:
            rows.append(run_one(name))
        except Exception as exc:  # noqa
            print(f"  [skip {name}: {exc}]")

    if not rows:
        print("\nNo datasets ran successfully; nothing to save.")
        return

    # ------------------------------------------------------------------
    # CSV table (drops the per-iteration trace columns)
    # ------------------------------------------------------------------
    tab_path = os.path.join(TAB_DIR, "real_data.csv")
    public_keys = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(tab_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=public_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in public_keys})
    print(f"\nSaved: {tab_path}")

    # ------------------------------------------------------------------
    # Convergence figure: one panel per dataset
    # ------------------------------------------------------------------
    n_panels = len(rows)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.0),
                             squeeze=False)
    floor = 1e-16
    for ax, row in zip(axes[0], rows):
        ig = np.maximum(row["_irls_gaps"], floor)
        dg = np.maximum(row["_dsm_gaps"],  floor)
        ax.semilogy(ig, color=COLOR_IRLS, marker="o", markersize=2.6,
                    linewidth=1.4, label="IRLS")
        ax.semilogy(dg, color=COLOR_DSM, linewidth=1.4,
                    label=r"SGPTL ($\bar f^{\,i}$)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$f-f^{*}$")
        ax.set_title(f"{row['name']} (M={row['M_train']}, H={row['H']})")
        ax.legend(loc="upper right")
        style_axes(ax)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "real_data_convergence.pdf")
    fig.savefig(fig_path)
    print(f"Saved: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
