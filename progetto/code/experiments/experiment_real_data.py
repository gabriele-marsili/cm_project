"""IRLS and SGPTL on diabetes and california_housing under an ELM transformation."""

import csv
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Pin BLAS to a single thread before numpy/sklearn load. sklearn's
# coordinate-descent core is single-threaded, but its level-1 BLAS calls
# (one dot/axpy per coordinate update) trigger thread spawning under the
# macOS Accelerate backend, on the long sklearn_trace runs this
# oversubscription makes wall time 3-4x worse. One thread is also faster
# for the small-matrix IRLS/Clarabel/ELM steps here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
np.seterr(all="ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SkLasso

from src import irls, deflected_subgradient
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM
apply_style()


SEED = 42
H = 200
LAMBDA = 0.1
TEST_FRACTION = 0.2
IRLS_KMAX = 2000  # run to convergence with tight smoothing
DSM_IMAX = 8000
DSM_DELTA0 = 0.1  # multiplier of f(w_0), never uses f*
DSM_RHO = 0.7  # default settled in Section 5.3.3

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _split_scale(X, y, test_frac=TEST_FRACTION, seed=SEED):
    """Train/test split, standardisation fit on the train split only (no leakage)"""
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
    y_std = float(y_tr.std()) or 1.0
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
    """ELM projection of both splits with the same fixed W_1"""
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=seed)
    return elm.transform(X_tr_raw), elm.transform(X_te_raw)


def ols_warm_start(X, y):
    """Cholesky solve of (X^T X + eps I) w = X^T y"""
    return solve_spd(X.T @ X + 1e-10 * np.eye(X.shape[1]),
                     X.T @ y, method="cholesky")


def sklearn_best_effort(X, y, lam, max_iter=100_000, tol=1e-10):
    """sklearn coordinate descent at a large budget with a tight tolerance.

    On these ELM-transformed problems sklearn CD does not meet its `tol`
    stop within `max_iter` (it returns n_iter_ == max_iter), so it is a
    third-party reference, not the f^* oracle. The f^* oracle comes from
    `reference_fstar` (IRLS to convergence, cross-validated by CVXPY-Clarabel).
    """
    M = X.shape[0]
    sk = SkLasso(alpha=lam / M, fit_intercept=False,
                 max_iter=max_iter, tol=tol)
    sk.fit(X, y)
    return sk.coef_, f_lasso(X, y, sk.coef_, lam), int(sk.n_iter_)


def sklearn_trace(X, y, lam, max_iter, n_samples=28, tol=1e-16):
    """Convergence trace of sklearn coordinate descent over its own budget.

    Sampled at geometrically spaced sweep counts up to `max_iter` (set to the
    SGPTL long-run budget so both span the same iteration axis). One CD run is
    advanced incrementally with `warm_start`: each fit resumes from the
    previous coefficient and runs the next block of sweeps. CD uses cyclic
    coordinate selection and never meets `tol` on these problems
    (cf. `sklearn_best_effort`), so each block runs in full and the snapshots
    trace one continuous trajectory at total cost ~one run to `max_iter`.
    Returns (iters, fvals) with fvals[k] the LASSO objective after iters[k] sweeps.
    """
    M = X.shape[0]
    iters = np.unique(np.geomspace(1.0, max_iter, n_samples).astype(int))
    fvals = np.empty(len(iters), dtype=float)
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


def reference_fstar(X, y, lam, w0=None):
    """Independent reference f^*.

    IRLS run to convergence (high budget, tight stop), optionally
    cross-validated against CVXPY's Clarabel interior-point solver. Returns
    (f_star, source) with source in {'cvxpy-clarabel', 'irls-converged'}.

    IRLS and Clarabel are structurally different optimisers (MM + linear solve
    vs primal-dual interior point). They agree to 6+ significant digits on the
    ELM-transformed real data (diabetes-ELM, California-ELM), which makes this
    a trustworthy proxy for the true optimum.
    """
    M, p = X.shape
    if w0 is None:
        w0 = ols_warm_start(X, y)
    # tight smoothing for the reference: eps_thr=1e-14 pushes the smoothed
    # surrogate to within machine precision of the true LASSO objective, so
    # f_irls here is a faithful upper bound on f^*
    res = irls(X, y, lam,
               eps_thr=1e-14, eps_stop=1e-16,
               k_max=3000, solver="cholesky", w0=w0)
    f_irls = float(np.min(res["f_vals"]))

    f_cvxpy = None
    w_cvxpy = None
    if M * p <= 4_000_000:  # Clarabel is tractable on this size
        try:
            import cvxpy as cp
            w = cp.Variable(p)
            obj = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm(w, 1))
            prob = cp.Problem(obj)
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=200_000,
                       tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
            if prob.status in ("optimal", "optimal_inaccurate") and prob.value is not None:
                f_cvxpy = float(prob.value)
                w_cvxpy = np.asarray(w.value, dtype=float)
        except Exception:
            pass

    if f_cvxpy is not None:
        f_star = min(f_irls, f_cvxpy)
        return f_star, f"cvxpy-clarabel ({f_cvxpy:.6e}) vs irls ({f_irls:.6e})", w_cvxpy
    return f_irls, "irls-converged", None


def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _sparsity(w, tol=1e-6):
    return float(np.mean(np.abs(w) < tol))


SPARSITY_TOLS = (1e-6, 1e-3)


def run_one(name):
    print(f"\n{'='*60}\nDataset: {name}\n{'='*60}")
    X_tr_raw, X_te_raw, y_tr, y_te = load_dataset(name)
    d_in = X_tr_raw.shape[1]
    M = X_tr_raw.shape[0]
    print(f"  n_train={M}, n_test={X_te_raw.shape[0]}, d={d_in}, H={H}")

    X_tr, X_te = build_hidden(X_tr_raw, X_te_raw, d_in)
    print(f"  hidden activations: shape={X_tr.shape}, "
          f"cond(X^T X) ≈ {np.linalg.cond(X_tr.T @ X_tr):.2e}")

    # independent reference f^*, IRLS-converged, validated by CVXPY-Clarabel
    # when tractable. sklearn CD on the ELM-transformed problems is reported
    # separately below as a third-party comparison, not as the reference: at
    # the budget the report quotes (max_iter=300, tol=1e-3) it sits well above
    # the true optimum (~+5 on diabetes-ELM, ~+130 on California-ELM)
    w_ols = ols_warm_start(X_tr, y_tr)
    f_w0 = float(f_lasso(X_tr, y_tr, w_ols, LAMBDA))
    f_star, src, w_cvxpy = reference_fstar(X_tr, y_tr, LAMBDA, w0=w_ols)
    print(f"  OLS warm start (shared): f(w_0) = {f_w0:.6f}")
    print(f"  f* (independent reference) = {f_star:.6f}  [source: {src}]")

    # third-party sklearn baseline at a generous budget (max_iter=100k,
    # tol=1e-10). sklearn CD never satisfies tol on these ELM-transformed
    # problems and returns at max_iter -> residual gap to f* on California is
    # then a few units, which is what disqualifies sklearn as the f^* reference
    w_star, f_skl, n_iter_skl = sklearn_best_effort(X_tr, y_tr, LAMBDA)
    skl_spar_str = ", ".join(f"sp@{tol:.0e}={_sparsity(w_star, tol):.0%}"
                             for tol in SPARSITY_TOLS)
    print(f"  sklearn (best-effort, n_iter={n_iter_skl}, gap to f* = {f_skl-f_star:+.2e}): f = {f_skl:.6f}, "
          f"{skl_spar_str}, test MSE = {_mse(y_te, X_te @ w_star):.4f}")

    res_i = irls(X_tr, y_tr, LAMBDA,
                 eps_thr=1e-12, eps_stop=1e-16,
                 k_max=IRLS_KMAX, solver="cholesky",
                 w0=w_ols, f_star=f_star)
    w_i = res_i["w"]
    f_i = f_lasso(X_tr, y_tr, w_i, LAMBDA)
    tag_i = " (gap ≤ 1e-12)" if res_i["gaps"][-1] < 1e-12 else ""
    spar_i_str = ", ".join(f"sp@{tol:.0e}={_sparsity(w_i, tol):.0%}"
                           for tol in SPARSITY_TOLS)
    print(f"  IRLS (warm) : {res_i['n_iter']} iter, "
          f"gap = {res_i['gaps'][-1]:.3e}{tag_i}, f = {f_i:.6f}, "
          f"{spar_i_str}, "
          f"test MSE = {_mse(y_te, X_te @ w_i):.4f}")

    # SGPTL: run both starts. The convergence figure overlays the descending
    # cold trajectory (the default configuration, in which SGPTL actually
    # optimises) with the warm one, which on California is pinned at the OLS
    # floor. The table reports the per-dataset start with the smaller final
    # gap, cold on diabetes (1.74 < 2.62), warm on California (0.46 < 64).
    # Both are admissible since Theorem 3.x does not constrain w_0.
    def _run_sgptl(w0):
        return deflected_subgradient(
            X_tr, y_tr, LAMBDA,
            w0=w0, i_max=DSM_IMAX, beta=1.0,
            delta0=DSM_DELTA0 * float(f_lasso(X_tr, y_tr, w0, LAMBDA)),
            rho=DSM_RHO, f_star=f_star,
        )

    # SGPTL at the submitted i_max=8000 budget, kept for diagnostics and for
    # the in-memory warm-start trace used by Section 5.3. The algorithmic
    # comparison of Section 5.6 instead uses the long-run cold result from
    # experiment_sgptl_long_run.py, which terminates near machine precision
    # and makes the iteration cost explicit (cache loaded below when present).
    res_d_cold = _run_sgptl(np.zeros(X_tr.shape[1]))
    res_d_warm = _run_sgptl(w_ols)
    # report the cold start on both datasets, the configuration in which the
    # method actually optimises. The warm-start run on California gives back
    # the OLS floor rather than an SGPTL result, reported separately in §5.3.
    res_d = res_d_cold
    start_tag = "cold"
    w_d = res_d["w"]
    f_d = f_lasso(X_tr, y_tr, w_d, LAMBDA)
    tag_d = " (gap ≤ 1e-12)" if res_d["gaps"][-1] < 1e-12 else ""
    spar_d_str = ", ".join(f"sp@{tol:.0e}={_sparsity(w_d, tol):.0%}"
                           for tol in SPARSITY_TOLS)
    print(f"  SGPTL ({start_tag}, rho={DSM_RHO}, delta0=c*f(w_0)): "
          f"{res_d['n_iter']} iter, "
          f"record gap = {res_d['gaps'][-1]:.3e}{tag_d}, f = {f_d:.6f}, "
          f"{spar_d_str}, test MSE = {_mse(y_te, X_te @ w_d):.4f}")

    # closed-form Ridge baseline at the same regularisation strength
    A_ridge = X_tr.T @ X_tr + LAMBDA * np.eye(X_tr.shape[1])
    w_ridge = solve_spd(A_ridge, X_tr.T @ y_tr, method="cholesky")
    print(f"  Ridge: closed form, test MSE = {_mse(y_te, X_te @ w_ridge):.4f}")

    # independent CVXPY-Clarabel solution, kept for the table. NaNs when
    # Clarabel was not run (large M*p)
    if w_cvxpy is not None:
        f_cvx = float(f_lasso(X_tr, y_tr, w_cvxpy, LAMBDA))
        mse_cvx = _mse(y_te, X_te @ w_cvxpy)
        sp_cvx_1e3 = _sparsity(w_cvxpy, 1e-3)
        sp_cvx_1e6 = _sparsity(w_cvxpy, 1e-6)
        print(f"  CVXPY-Clarabel: f = {f_cvx:.6f}, gap = {f_cvx - f_star:+.2e}, "
              f"sp@1e-3 = {sp_cvx_1e3:.0%}, test MSE = {mse_cvx:.4f}")
    else:
        f_cvx = float("nan")
        mse_cvx = float("nan")
        sp_cvx_1e3 = float("nan")
        sp_cvx_1e6 = float("nan")

    # long-run SGPTL cache (cold start, large i_max) if present -> feeds the
    # algorithmic-comparison row of Table 5.8 and the SGPTL trace in Fig 5.12
    cache_path = os.path.join(
        os.path.dirname(__file__), os.pardir, "results", "tables",
        "long_run_cache", f"{name}.npz",
    )
    sgptl_long = None
    if os.path.exists(cache_path):
        with np.load(cache_path) as z:
            sgptl_long = {
                "i_sampled":    z["i_sampled"],
                "gaps_sampled": z["gaps_sampled"],
                "f_final":      float(z["f_final"]),
                "mse":          float(z["mse"]),
                "sp_1e3":       float(z["sp_1e3"]),
                "sp_1e6":       float(z["sp_1e6"]),
                "i_max":        int(z["i_max"]),
                "elapsed_s":    float(z["elapsed_s"]),
            }
        gap_long = sgptl_long["f_final"] - f_star
        print(f"  SGPTL (long-run cache, i_max={sgptl_long['i_max']:_}): "
              f"f = {sgptl_long['f_final']:.6f}, gap = {gap_long:+.2e}, "
              f"sp@1e-3 = {sgptl_long['sp_1e3']:.0%}, "
              f"test MSE = {sgptl_long['mse']:.4f}, "
              f"elapsed = {sgptl_long['elapsed_s']:.1f}s")
    else:
        print(f"  [no long-run cache at {cache_path}, "
              f"figure/table will fall back to i_max={DSM_IMAX} run]")

    # sklearn CD convergence trace over the same iteration budget as SGPTL
    # (the long-run i_max) -> Figure 5.12 compares both on a common x-axis
    skl_budget = sgptl_long["i_max"] if sgptl_long is not None else 100_000
    skl_iters, skl_fvals = sklearn_trace(X_tr, y_tr, LAMBDA, max_iter=skl_budget)
    skl_trace_gap = (skl_fvals[-1] - f_star) / abs(f_star)
    print(f"  sklearn CD trace (budget={skl_budget:_} sweeps): "
          f"final rel. gap = {skl_trace_gap:+.2e}")

    return {
        "name": name,
        "M_train": M, "M_test": X_te_raw.shape[0], "d": d_in, "H": H,
        "f_star": f_star,
        "f_star_source": src,
        "f_w0":  f_w0,
        "f_irls": f_i,
        "f_dsm":  f_d,
        "f_skl":  f_skl,
        "sgptl_start": start_tag,
        "gap_irls": res_i["gaps"][-1],
        "gap_dsm":  res_d["gaps"][-1],
        "gap_skl":  f_skl - f_star,
        "iter_irls": res_i["n_iter"],
        "iter_dsm":  res_d["n_iter"],
        "spar_skl_1e6":  _sparsity(w_star,   1e-6),
        "spar_irls_1e6": _sparsity(w_i,      1e-6),
        "spar_dsm_1e6":  _sparsity(w_d,      1e-6),
        "spar_skl_1e3":  _sparsity(w_star,   1e-3),
        "spar_irls_1e3": _sparsity(w_i,      1e-3),
        "spar_dsm_1e3":  _sparsity(w_d,      1e-3),
        "mse_skl":     _mse(y_te, X_te @ w_star),
        "mse_irls":    _mse(y_te, X_te @ w_i),
        "mse_dsm":     _mse(y_te, X_te @ w_d),
        "mse_ridge":   _mse(y_te, X_te @ w_ridge),
        "f_cvxpy":     f_cvx,
        "gap_cvxpy":   (f_cvx - f_star) if not (f_cvx != f_cvx) else float("nan"),
        "spar_cvxpy_1e3": sp_cvx_1e3,
        "spar_cvxpy_1e6": sp_cvx_1e6,
        "mse_cvxpy":   mse_cvx,
        "_irls_gaps":  res_i["gaps"],
        "_dsm_gaps":   res_d["gaps"],
        "_irls_fvals": res_i["f_vals"],
        "_dsm_fbar":   res_d["f_bar"],
        "_dsm_fbar_cold": res_d_cold["f_bar"],
        "_dsm_fbar_warm": res_d_warm["f_bar"],
        "_sgptl_long": sgptl_long,
        "_skl_trace": {"iters": skl_iters, "fvals": skl_fvals},
    }


def run() -> None:
    print("=" * 60)
    print("Real-data experiment (ELM + LASSO)")
    print("=" * 60)

    rows = []
    for name in ("diabetes", "california"):
        try:
            rows.append(run_one(name))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {name}: {exc}]")

    if not rows:
        print("\nNo datasets ran successfully. Nothing to save.")
        return

    # CSV table, strip the per-iteration trace columns (prefix "_")
    tab_path = os.path.join(TAB_DIR, "real_data.csv")
    public_keys = [k for k in rows[0].keys() if not k.startswith("_")]
    with open(tab_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=public_keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in public_keys})
    print(f"\nSaved: {tab_path}")

    # gap to f* per dataset. IRLS reaches f* to machine precision in ~100
    # iterations, SGPTL cold descends at the sublinear rate to the f* floor.
    # sklearn CD is overlaid as a third-party trace (gap per coordinate sweep)
    # that stalls above f* and returns at max_iter.
    n_panels = len(rows)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 5.5),
                             squeeze=False)
    floor = 1e-12
    for ax, row in zip(axes[0], rows):
        irls_fvals = np.asarray(row["_irls_fvals"], dtype=float)
        f_baseline = row["f_star"]
        abs_fstar = abs(f_baseline)

        # relative optimality gap (f - f*)/|f*| on the y-axis: the report's
        # cost-to-target uses a relative target, so the two datasets
        # (diabetes f*~53, california f*~2358) sit on a common scale
        irls_gap = np.maximum((irls_fvals - f_baseline) / abs_fstar, floor)
        irls_iters = np.arange(1, len(irls_gap) + 1)

        # prefer the long-run SGPTL trace (cache from
        # experiment_sgptl_long_run.py) when present: the configuration in
        # which SGPTL reaches the f^* floor. Fall back to the in-memory
        # i_max=DSM_IMAX cold trace otherwise.
        if row.get("_sgptl_long") is not None:
            sl = row["_sgptl_long"]
            cold_iters = sl["i_sampled"]
            cold_gap   = np.maximum(sl["gaps_sampled"] / abs_fstar, floor)
            sgptl_label = (
                rf"SGPTL (cold, $i_{{\max}}=10^{{{int(np.log10(sl['i_max']))}}}$)"
            )
        else:
            dsm_cold = np.asarray(row["_dsm_fbar_cold"], dtype=float)
            cold_gap = np.maximum((dsm_cold - f_baseline) / abs_fstar, floor)
            cold_iters = np.arange(1, len(cold_gap) + 1)
            sgptl_label = r"SGPTL (cold, $i_{\max}=8000$)"

        # sklearn CD convergence trace (gap per coordinate sweep)
        skl_iters = np.asarray(row["_skl_trace"]["iters"], dtype=float)
        skl_gap = np.maximum(
            (np.asarray(row["_skl_trace"]["fvals"], dtype=float) - f_baseline)
            / abs_fstar, floor)

        ax.loglog(irls_iters, irls_gap,
                  color=COLOR_IRLS, marker="o", markersize=4.0,
                  linewidth=2.0, label=r"IRLS (warm)")
        ax.loglog(cold_iters, cold_gap,
                  color=COLOR_DSM, linewidth=2.0, label=sgptl_label)
        ax.loglog(skl_iters, skl_gap, color="#2c7a30", linestyle=":",
                  linewidth=1.6, marker="s", markersize=3.0, alpha=0.9,
                  label=r"sklearn CD (cold, tol $10^{-16}$)")

        ax.set_xlabel("Iteration  (log scale)")
        ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$")
        ax.set_title(f"{row['name']} ($M={row['M_train']}$, $H={row['H']}$)")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(bottom=floor / 5)
        style_axes(ax)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "real_data_convergence.pdf")
    fig.savefig(fig_path)
    print(f"Saved: {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
