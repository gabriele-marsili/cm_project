"""Long-run SGPTL rate verification on the ELM-transformed real datasets.

Runs SGPTL cold from w_0 = 0 with very large i_max on diabetes and california
to test the O(1/sqrt(k)) envelope of Theorem 3.1 over several decades of k.
Produces a log-log figure overlaying the empirical record gap against the
envelope g_0 / sqrt(k), with g_0 the initial gap to f^*.

Independent of experiment_real_data.py and experiment_warm_vs_cold_real_data.py:
those use i_max = 8000 (the reported submission budget), this one pushes i_max
much higher to check that the predicted rate holds at scale.
"""
import csv
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

np.seterr(all="ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import StandardScaler

from src.deflected_subgradient import deflected_subgradient
from src.elm import ELM
from src.irls import irls
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd

from _plot_style import (
    SIZE_DOUBLE,
    apply_style,
    plot_long_run_panel,
)

apply_style()


SEED = 42
H = 200
LAMBDA = 0.1
TEST_FRACTION = 0.2
DSM_DELTA0_FACTOR = 0.1
DSM_RHO = 0.7

# per-dataset iteration budgets. diabetes is small (M=354) -> deeper sweep,
# california is large (M=16512) -> capped at 1e6 to bound runtime
I_MAX = {
    "diabetes": 10_000_000,
    "california": 1_000_000,
}

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _split_scale(X, y, seed=SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    n_test = int(TEST_FRACTION * len(y))
    te, tr = perm[:n_test], perm[n_test:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr].astype(float), y[te].astype(float)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
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
    return _split_scale(d.data, d.target)


def build_hidden(X_raw, d_in):
    elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=SEED)
    return elm.transform(X_raw)


def reference_fstar(X, y):
    """IRLS-converged reference value (same proxy used elsewhere)"""
    w0_ols = solve_spd(
        X.T @ X + 1e-12 * np.eye(X.shape[1]), X.T @ y, method="cholesky"
    )
    res = irls(
        X,
        y,
        LAMBDA,
        eps_thr=1e-8,
        eps_stop=1e-14,
        k_max=500,
        solver="cholesky",
        w0=w0_ols,
    )
    return float(f_lasso(X, y, res["w"], LAMBDA))


def long_run(name):
    i_max = I_MAX[name]
    print(f"\n{'=' * 60}\n{name}: i_max = {i_max:_}\n{'=' * 60}")
    X_tr_raw, X_te_raw, y_tr, y_te = load_dataset(name)
    d_in = X_tr_raw.shape[1]
    X = build_hidden(X_tr_raw, d_in)
    X_te = build_hidden(X_te_raw, d_in)
    print(f"  shape: {X.shape}, cond(X^T X) = {np.linalg.cond(X.T @ X):.2e}")

    f_star = reference_fstar(X, y_tr)
    print(f"  f* = {f_star:.6f}")

    w0 = np.zeros(X.shape[1])
    f_w0 = float(f_lasso(X, y_tr, w0, LAMBDA))
    gap0 = f_w0 - f_star
    print(f"  cold-start gap g_0 = f(0) - f* = {gap0:.4e}")

    delta0 = DSM_DELTA0_FACTOR * f_w0
    t0 = time.time()
    res = deflected_subgradient(
        X,
        y_tr,
        LAMBDA,
        w0=w0,
        i_max=i_max,
        beta=1.0,
        delta0=delta0,
        rho=DSM_RHO,
        f_star=f_star,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")

    fbar = np.asarray(res["f_bar"], dtype=float)
    gaps = np.maximum(fbar - f_star, 1e-300)
    final_gap = float(gaps[-1])
    print(f"  final record gap = {final_gap:.4e}")

    w_final = np.asarray(res["w"], dtype=float)
    f_final = float(f_lasso(X, y_tr, w_final, LAMBDA))
    mse = float(np.mean((y_te - X_te @ w_final) ** 2))
    sp_1e3 = float(np.mean(np.abs(w_final) < 1e-3))
    sp_1e6 = float(np.mean(np.abs(w_final) < 1e-6))
    print(f"  final f={f_final:.6f}  test MSE={mse:.4f}  sp@1e-3={sp_1e3:.0%}")

    # persist the gap trace (geometric sub-sample) and final w so downstream
    # scripts can reuse without re-running
    cache_dir = os.path.join(TAB_DIR, "long_run_cache")
    os.makedirs(cache_dir, exist_ok=True)
    idx = np.unique(np.geomspace(1, len(gaps), 2000).astype(int)) - 1
    np.savez_compressed(
        os.path.join(cache_dir, f"{name}.npz"),
        i_sampled=(idx + 1).astype(np.int64),
        gaps_sampled=gaps[idx].astype(np.float64),
        w_final=w_final,
        f_final=f_final,
        f_star=f_star,
        gap0=gap0,
        i_max=i_max,
        mse=mse,
        sp_1e3=sp_1e3,
        sp_1e6=sp_1e6,
        elapsed_s=elapsed,
    )

    # sample at decade boundaries for the summary table
    sample_ks = [8_000, 80_000, 800_000, 8_000_000, 10_000_000]
    samples = []
    for k in sample_ks:
        if k <= len(gaps):
            samples.append(
                {
                    "k": k,
                    "observed": float(gaps[k - 1]),
                    "predicted_env": gap0 / np.sqrt(k),
                }
            )
    return {
        "name": name,
        "i_max": i_max,
        "f_star": f_star,
        "gap0": gap0,
        "elapsed_s": elapsed,
        "gaps": gaps,
        "final_gap": final_gap,
        "samples": samples,
        "f_final": f_final,
        "mse": mse,
        "sp_1e3": sp_1e3,
    }


def save_table(rows):
    path = os.path.join(TAB_DIR, "sgptl_long_run.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "i_max", "k", "observed_gap", "predicted_envelope"])
        for r in rows:
            for s in r["samples"]:
                wr.writerow(
                    [
                        r["name"],
                        r["i_max"],
                        s["k"],
                        f"{s['observed']:.6e}",
                        f"{s['predicted_env']:.6e}",
                    ]
                )
    print(f"Saved: {path}")


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)
    for ax, r in zip(axes, rows):
        gaps = r["gaps"]
        n = len(gaps)
        # geometric subsample to keep the plot file small
        idx = np.unique(np.geomspace(1, max(n - 1, 1), 2000).astype(int))
        iters = idx + 1
        title = (
            rf"{r['name']} cold (ELM, $H={H}$, $\lambda={LAMBDA}$, "
            rf"$i_{{\max}}={{{r['i_max']:,}}}$)".replace(",", r"{,}")
        )
        fs = r.get("f_star", 1.0)
        plot_long_run_panel(ax, iters, gaps[idx] / fs, r["gap0"] / fs, title)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "sgptl_long_run.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("SGPTL long-run rate verification (cold start, ELM real data)")
    print("=" * 60)
    rows = []
    for name in ["diabetes", "california"]:
        rows.append(long_run(name))
    save_table(rows)
    plot(rows)
    print("\n=== summary ===")
    for r in rows:
        print(f"\n{r['name']} (i_max={r['i_max']:_}, {r['elapsed_s']:.1f}s):")
        print(f"  g_0 = {r['gap0']:.4e}")
        print("  k         observed           predicted (g_0/sqrt(k))")
        for s in r["samples"]:
            print(
                f"  {s['k']:>10_}   {s['observed']:.4e}        {s['predicted_env']:.4e}"
            )


if __name__ == "__main__":
    main()
