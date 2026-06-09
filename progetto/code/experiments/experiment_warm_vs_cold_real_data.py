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

from src import deflected_subgradient, irls
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_DSM, COLOR_FCUR
apply_style()


SEED          = 42
H             = 200
LAMBDA_DIABETES = 0.1
LAMBDA_CALIFORNIA = 0.1
TEST_FRACTION = 0.2
# SGPTL runs 8000 iterations everywhere with the shared defaults of the
# long-run experiment (rho=0.7, delta0 = 0.1 f(w_0)); this keeps the cold
# real-data gaps consistent with the long-run rate-verification figure.
DSM_IMAX_DIABETES = 8000
DSM_IMAX_CALIFORNIA = 8000
DSM_DELTA0_DIABETES = 0.1
DSM_DELTA0_CALIFORNIA = 0.1
DSM_RHO_DIABETES = 0.7
DSM_RHO_CALIFORNIA = 0.7

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


def build_hidden(X_tr_raw, X_te_raw, d_in, H=H, seed=SEED, name="california"):
    """Apply the ELM projection to both splits using the same fixed W_1."""
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA_CALIFORNIA if name == "california" else LAMBDA_DIABETES, random_state=seed)
    return elm.transform(X_tr_raw), elm.transform(X_te_raw)


def _n_contractions(delta_hist):
    """Count how many times delta was contracted (strictly decreasing step)."""
    d = np.asarray(delta_hist, dtype=float)
    return int(np.sum(np.diff(d) < 0))


def ols_warm_start(X, y):
    """Cholesky-based (X^T X + eps I)^{-1} X^T y."""
    return solve_spd(X.T @ X + 1e-10 * np.eye(X.shape[1]),
                     X.T @ y, method="cholesky")


def reference_fstar(X, y, lam, w_ols):
    """Compute an independent reference f^* (IRLS-converged + CVXPY validation).

    Two structurally different solvers (MM-style IRLS + interior-point Clarabel)
    cross-validate the optimum. sklearn CD is too loose on these ELM-transformed
    instances to serve as the reference.
    """
    M, p = X.shape
    res = irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-14,
               k_max=500, solver="cholesky", w0=w_ols)
    f_irls = float(np.min(res["f_vals"]))
    f_cvxpy = None
    if M * p <= 4_000_000:
        try:
            import cvxpy as cp
            w = cp.Variable(p)
            obj = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm(w, 1))
            prob = cp.Problem(obj)
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=200_000)
            if prob.status in ("optimal", "optimal_inaccurate") and prob.value is not None:
                f_cvxpy = float(prob.value)
        except Exception:
            pass
    if f_cvxpy is not None:
        return min(f_irls, f_cvxpy), f"cvxpy ({f_cvxpy:.6e}) + irls ({f_irls:.6e})"
    return f_irls, "irls-converged"


def run_one(name):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    X_tr_raw, X_te_raw, y_tr, y_te = load_dataset(name)
    d_in = X_tr_raw.shape[1]
    M = X_tr_raw.shape[0]
    print(f"  n_train={M}, n_test={X_te_raw.shape[0]}, d={d_in}, H={H}")

    X_tr, X_te = build_hidden(X_tr_raw, X_te_raw, d_in, name=name)
    print(f"  hidden activations: shape={X_tr.shape}, "
          f"cond(X^T X) ≈ {np.linalg.cond(X_tr.T @ X_tr):.2e}")

    lambda_ = LAMBDA_CALIFORNIA if name == "california" else LAMBDA_DIABETES
    i_max = DSM_IMAX_CALIFORNIA if name == "california" else DSM_IMAX_DIABETES
    rho = DSM_RHO_CALIFORNIA if name == "california" else DSM_RHO_DIABETES

    # Shared OLS warm start; used both to anchor δ_0 (which must be computable
    # a priori, before f^* is known) and as the warm-start iterate.
    w_ols = ols_warm_start(X_tr, y_tr)
    f_w_ols = float(f_lasso(X_tr, y_tr, w_ols, lambda_))

    # Independent reference f^* - IRLS-converged + CVXPY validation. Used only
    # for the gap-to-f^* visualisation; the SGPTL algorithm itself never sees it.
    f_star, src = reference_fstar(X_tr, y_tr, lambda_, w_ols)
    print(f"  f(w_OLS) = {f_w_ols:.6f}")
    print(f"  f* (independent reference) = {f_star:.6f}  [source: {src}]")
    print(f"  gap(OLS, f*) = {f_w_ols - f_star:.3e}")

    # δ_0 = c · f(w_0) with w_0 the run's OWN starting point (report §5.4): the
    # warm run anchors on f(w_OLS), the cold run on f(0). Using the cold start's
    # own f(0) (not f(w_OLS)) keeps the cold gaps consistent with the long-run
    # rate-verification figure, which also uses δ_0 = c·f(0) cold.
    c_delta = DSM_DELTA0_CALIFORNIA if name == "california" else DSM_DELTA0_DIABETES
    w_cold = np.zeros(H)
    f_w_cold = float(f_lasso(X_tr, y_tr, w_cold, lambda_))
    delta0_warm = c_delta * f_w_ols
    delta0_cold = c_delta * f_w_cold


    # WARM START (OLS)
    t0    = time.time()
    res_warm = deflected_subgradient(
        X_tr, y_tr, lam=lambda_,
        w0=w_ols, i_max=i_max, beta=1.0,
        delta0=delta0_warm, rho=rho,
        f_star=f_star,
    )
    time_warm = time.time() - t0
    f_w = f_lasso(X_tr, y_tr, res_warm["w"], LAMBDA_CALIFORNIA if name == "california" else LAMBDA_DIABETES)
    print(f"  SGPTL (Warm) : {res_warm['n_iter']} iter, "
          f"gap = {res_warm['gaps'][-1]:.3e}, f = {f_w:.6f}, time = {time_warm:.4f}s")


    # COLD START (w = 0)
    t0     = time.time()
    res_cold = deflected_subgradient(
        X_tr, y_tr, lam=lambda_,
        w0=w_cold, i_max=i_max, beta=1.0,
        delta0=delta0_cold, rho=rho,
        f_star=f_star,
    )
    time_cold = time.time() - t0
    f_c = f_lasso(X_tr, y_tr, res_cold["w"], lambda_)
    print(f"  SGPTL (Cold) : {res_cold['n_iter']} iter, "
          f"gap = {res_cold['gaps'][-1]:.3e}, f = {f_c:.6f}, time = {time_cold:.4f}s")

    # Relative final gaps (f - f*)/|f*| at i_max iterations.
    abs_f_star = abs(f_star)
    rel_warm = float(res_warm["gaps"][-1]) / abs_f_star
    rel_cold = float(res_cold["gaps"][-1]) / abs_f_star
    n_contr_warm = _n_contractions(res_warm["delta_hist"])
    n_contr_cold = _n_contractions(res_cold["delta_hist"])
    print(f"  f* = {f_star:.6f}  (relative gap = (f - f*)/|f*|)")
    print(f"  SGPTL warm {name}: rel final gap = {rel_warm:.4e} "
          f"(abs {float(res_warm['gaps'][-1]):.3e}, {res_warm['n_iter']} iter, "
          f"{n_contr_warm} delta-contractions)")
    print(f"  SGPTL cold {name}: rel final gap = {rel_cold:.4e} "
          f"(abs {float(res_cold['gaps'][-1]):.3e}, {res_cold['n_iter']} iter, "
          f"{n_contr_cold} delta-contractions)")

    return {
        "name":      name,
        "M_train":   M,
        "H":         H,
        "f_star":    f_star,
        "time_warm": time_warm,
        "time_cold": time_cold,
        "rel_warm":  rel_warm,
        "rel_cold":  rel_cold,
        "n_iter_warm": res_warm["n_iter"],
        "n_iter_cold": res_cold["n_iter"],
        "n_contr_warm": n_contr_warm,
        "n_contr_cold": n_contr_cold,
        # Store f_bar directly so no information is lost when f_bar < f_star
        # (gaps are clamped to 0 in that region, making reconstruction lossy).
        "fbar_warm": res_warm["f_bar"],
        "fbar_cold": res_cold["f_bar"],
    }


def run() -> None:
    print("=" * 60)
    print("Real-data experiment (ELM + LASSO) - Warm vs Cold Start")
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
        abs_f_star = abs(f_star)
        gw = np.maximum((f_warm - f_star) / abs_f_star, floor)
        gc = np.maximum((f_cold - f_star) / abs_f_star, floor)

        t_warm = row["time_warm"]
        t_cold = row["time_cold"]

        # Panel 1: gap vs iterations
        iters_w = np.arange(1, len(gw) + 1)
        iters_c = np.arange(1, len(gc) + 1)

        ax_iter.semilogy(iters_w, gw, color=COLOR_DSM,  linewidth=2.0,
                         label=f"Warm Start (OLS) [{t_warm:.2f}s]")
        ax_iter.semilogy(iters_c, gc, color=COLOR_FCUR, linewidth=2.0,
                         label=rf"Cold Start ($w_0=0$) [{t_cold:.2f}s]")
        ax_iter.scatter([len(gw)], [gw[-1]], s=40, color=COLOR_DSM,  zorder=5)
        ax_iter.scatter([len(gc)], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)

        ax_iter.set_xlabel("Iteration (linear scale)")
        ax_iter.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$ (log scale)")
        ax_iter.set_title(f"{row['name'].capitalize()} - Convergence vs Iterations")
        ax_iter.legend(loc="upper right")
        style_axes(ax_iter)

        # Panel 2: gap vs wall-clock time
        time_arr_w = np.linspace(0, t_warm, len(gw))
        time_arr_c = np.linspace(0, t_cold, len(gc))

        ax_time.semilogy(time_arr_w, gw, color=COLOR_DSM,  linewidth=2.0,
                         label="Warm Start (OLS)")
        ax_time.semilogy(time_arr_c, gc, color=COLOR_FCUR, linewidth=2.0,
                         label=r"Cold Start ($w_0=0$)")
        ax_time.scatter([time_arr_w[-1]], [gw[-1]], s=40, color=COLOR_DSM,  zorder=5)
        ax_time.scatter([time_arr_c[-1]], [gc[-1]], s=40, color=COLOR_FCUR, zorder=5)

        ax_time.set_xlabel("Time [seconds] (linear scale)")
        ax_time.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$ (log scale)")
        ax_time.set_title(f"{row['name'].capitalize()} - Convergence vs Time")
        ax_time.legend(loc="upper right")
        style_axes(ax_time)

    fig.suptitle("SGPTL Initialization Impact: Warm Start vs Cold Start on Real Data",
                 fontsize=16, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(FIG_DIR, "real_data_warm_vs_cold.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")
    plt.close(fig)

    # --- Save SGPTL real-data warm/cold rows of Table 5.2 to CSV ---
    csv_path = os.path.join(TAB_DIR, "warm_cold_sgptl_real.csv")
    import csv as _csv
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["dataset", "M_train", "H", "i_max", "start",
                    "n_iter", "n_contractions", "rel_final_gap", "f_star"])
        for row in rows:
            imax = DSM_IMAX_CALIFORNIA if row["name"] == "california" else DSM_IMAX_DIABETES
            w.writerow([row["name"], row["M_train"], row["H"], imax, "warm",
                        row["n_iter_warm"], row["n_contr_warm"],
                        f"{row['rel_warm']:.6e}", f"{row['f_star']:.10g}"])
            w.writerow([row["name"], row["M_train"], row["H"], imax, "cold",
                        row["n_iter_cold"], row["n_contr_cold"],
                        f"{row['rel_cold']:.6e}", f"{row['f_star']:.10g}"])
    print(f"Saved SGPTL real-data warm/cold CSV: {csv_path}")


if __name__ == "__main__":
    run()
