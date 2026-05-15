"""SGPTL (Deflected Subgradient) on diabetes and california_housing: Warm Start vs Cold Start."""

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

from src import deflected_subgradient
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_DSM, COLOR_FCUR
apply_style()


SEED          = 42
H             = 200
LAMBDA        = 0.1
TEST_FRACTION = 0.2
DSM_IMAX      = 8000
DSM_DELTA0    = 0.1
DSM_RHO       = 0.9

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _split_scale(X, y, test_frac=TEST_FRACTION, seed=SEED):
    """Train/test split + standardisation fit on the train split only (no leakage)."""
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
    """Apply the ELM projection to both splits using the same fixed W_1."""
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=seed)
    return elm.transform(X_tr_raw), elm.transform(X_te_raw)


def reference_solution(X, y, lam):
    """sklearn coordinate descent reference at moderate tolerance."""
    M = X.shape[0]
    sk = SkLasso(alpha=lam / M, fit_intercept=False, max_iter=300, tol=1e-3)
    sk.fit(X, y)
    w_star = sk.coef_
    f_star = f_lasso(X, y, w_star, lam)
    return w_star, f_star


def ols_warm_start(X, y):
    """Cholesky-based (X^T X + eps I)^{-1} X^T y."""
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

    # f_star: sklearn on diabetes, OLS placeholder on california (sklearn
    # does not converge there within a reasonable iteration budget).
    if name == "california":
        print("  sklearn skipped on this dataset; using OLS f as placeholder for f_star.")
        f_star = float(f_lasso(X_tr, y_tr, ols_warm_start(X_tr, y_tr), LAMBDA))
    else:
        _, f_star = reference_solution(X_tr, y_tr, LAMBDA)
        print(f"  f* (sklearn baseline) = {f_star:.6f}")

    # ------------------------------------------------------------------
    # WARM START (OLS)
    # ------------------------------------------------------------------
    t0    = time.time()
    w_ols = ols_warm_start(X_tr, y_tr)
    res_warm = deflected_subgradient(
        X_tr, y_tr, LAMBDA,
        w0=w_ols, i_max=DSM_IMAX, beta=1.0,
        delta0=DSM_DELTA0 * f_star, rho=DSM_RHO,
        f_star=f_star,
    )
    time_warm = time.time() - t0
    f_w = f_lasso(X_tr, y_tr, res_warm["w"], LAMBDA)
    print(f"  SGPTL (Warm) : {res_warm['n_iter']} iter, "
          f"gap = {res_warm['gaps'][-1]:.3e}, f = {f_w:.6f}, time = {time_warm:.4f}s")

    # ------------------------------------------------------------------
    # COLD START (w = 0)
    # ------------------------------------------------------------------
    t0     = time.time()
    w_cold = np.zeros(H)
    res_cold = deflected_subgradient(
        X_tr, y_tr, LAMBDA,
        w0=w_cold, i_max=DSM_IMAX, beta=1.0,
        delta0=DSM_DELTA0 * f_star, rho=DSM_RHO,
        f_star=f_star,
    )
    time_cold = time.time() - t0
    f_c = f_lasso(X_tr, y_tr, res_cold["w"], LAMBDA)
    print(f"  SGPTL (Cold) : {res_cold['n_iter']} iter, "
          f"gap = {res_cold['gaps'][-1]:.3e}, f = {f_c:.6f}, time = {time_cold:.4f}s")

    return {
        "name":      name,
        "M_train":   M,
        "H":         H,
        "f_star":    f_star,
        "time_warm": time_warm,
        "time_cold": time_cold,
        # Store f_bar directly so no information is lost when f_bar < f_star
        # (gaps are clamped to 0 in that region, making reconstruction lossy).
        "fbar_warm": res_warm["f_bar"],
        "fbar_cold": res_cold["f_bar"],
    }


def run() -> None:
    print("=" * 60)
    print("Real-data experiment (ELM + LASSO) — Warm vs Cold Start")
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

    floor      = 1e-12
    n_datasets = len(rows)
    fig, axes  = plt.subplots(n_datasets, 2,
                               figsize=(14, 5 * n_datasets), squeeze=False)

    for i, row in enumerate(rows):
        ax_iter = axes[i, 0]
        ax_time = axes[i, 1]

        f_star = row["f_star"]
        f_warm = np.asarray(row["fbar_warm"], dtype=float)
        f_cold = np.asarray(row["fbar_cold"], dtype=float)

        # Use f_star as a fixed, interpretable baseline for both curves so
        # that the Y-axis reads f_bar - f* directly.
        # A dynamic f_min built from whichever run converges furthest would
        # drag the baseline down and make the other curve look like it never
        # converges, even when both runs reach the same optimum.
        gw = np.maximum(f_warm - f_star, floor)
        gc = np.maximum(f_cold - f_star, floor)

        t_warm = row["time_warm"]
        t_cold = row["time_cold"]

        # ------------------------------------------------------------------
        # Panel 1: gap vs iterations
        # ------------------------------------------------------------------
        iters_w = np.arange(1, len(gw) + 1)
        iters_c = np.arange(1, len(gc) + 1)

        ax_iter.semilogy(iters_w, gw, color=COLOR_DSM,  linewidth=2.0,
                         label=f"Warm Start (OLS) [{t_warm:.2f}s]")
        ax_iter.semilogy(iters_c, gc, color=COLOR_FCUR, linewidth=2.0,
                         label=rf"Cold Start ($w_0=0$) [{t_cold:.2f}s]")
        ax_iter.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM,  zorder=5)
        ax_iter.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)

        ax_iter.set_xlabel("Iteration (linear scale)")
        ax_iter.set_ylabel(r"$\bar{f}^{\,i} - f^{*}$ (log scale)")
        ax_iter.set_title(f"{row['name'].capitalize()} — Convergence vs Iterations")
        ax_iter.legend(loc="upper right")
        style_axes(ax_iter)

        # ------------------------------------------------------------------
        # Panel 2: gap vs wall-clock time
        # ------------------------------------------------------------------
        time_arr_w = np.linspace(0, t_warm, len(gw))
        time_arr_c = np.linspace(0, t_cold, len(gc))

        ax_time.semilogy(time_arr_w, gw, color=COLOR_DSM,  linewidth=2.0,
                         label="Warm Start (OLS)")
        ax_time.semilogy(time_arr_c, gc, color=COLOR_FCUR, linewidth=2.0,
                         label=r"Cold Start ($w_0=0$)")
        ax_time.scatter([time_arr_w[-1]], [gw[-1]], s=40, color=COLOR_DSM,  zorder=5)
        ax_time.scatter([time_arr_c[-1]], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)

        ax_time.set_xlabel("Time [seconds] (linear scale)")
        ax_time.set_ylabel(r"$\bar{f}^{\,i} - f^{*}$ (log scale)")
        ax_time.set_title(f"{row['name'].capitalize()} — Convergence vs Time")
        ax_time.legend(loc="upper right")
        style_axes(ax_time)

    fig.suptitle("SGPTL Initialization Impact: Warm Start vs Cold Start on Real Data",
                 fontsize=16, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(FIG_DIR, "real_data_warm_vs_cold.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()