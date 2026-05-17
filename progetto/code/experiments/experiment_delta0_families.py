"""SGPTL sensitivity to the initial target gap delta_0: three families.

Family A: delta_0 = c * f_LASSO(w_OLS)        (current default, f*-free)
Family B: delta_0 = c * f_OLS(w_OLS)          (drops lam*||w||_1, f*-free)
Family C: delta_0 = c * f^*                   (oracle baseline, diagnostic)

Run on three instances:
  - synthetic (H=100, M=300): f^* known via IRLS to machine precision
  - diabetes ELM-transformed (H=200): f^* proxy via IRLS run to convergence
  - california ELM-transformed (H=200): f^* proxy via IRLS run to convergence
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
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler

from src import deflected_subgradient, irls, make_lasso_problem
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         RAMP_BLUES, RAMP_ORANGES, RAMP_PURPLES)

apply_style()


SEED = 42
LAMBDA = 0.1
NOISE = 0.05
SPARSITY = 0.1
I_MAX = 8000
RHO = 0.7
R_PATIENCE = 1.0
GAMMA_MIN = 0.05
BETA = 1.0
TEST_FRACTION = 0.2
H_SYNTH, M_SYNTH = 100, 300
H_REAL = 200

C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    return np.maximum(np.asarray(arr, dtype=float), floor)


def _count_contractions(delta_hist) -> int:
    return int(np.sum(np.diff(np.asarray(delta_hist, dtype=float)) < 0))


def _load_real(name):
    """Return train-side ELM hidden matrix and target for diabetes / california."""
    if name == "diabetes":
        d = load_diabetes()
    elif name == "california":
        d = fetch_california_housing()
    else:
        raise ValueError(name)
    X = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target, dtype=float)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(y))
    n_test = int(TEST_FRACTION * len(y))
    tr = perm[n_test:]
    X_tr_raw, y_tr_raw = X[tr], y[tr]

    sx = StandardScaler().fit(X_tr_raw)
    X_tr_raw = sx.transform(X_tr_raw)
    y_tr = (y_tr_raw - y_tr_raw.mean()) / (y_tr_raw.std() or 1.0)

    elm = ELM(d=X_tr_raw.shape[1], p=H_REAL,
              activation="sigmoid", random_state=SEED)
    X_hidden = elm.transform(X_tr_raw)
    return X_hidden, y_tr, X_hidden.shape[1]


def _sweep_instance(name, X, y, lam, H):
    print(f"\n{'=' * 60}\nInstance: {name}  (X shape {X.shape}, lambda={lam})\n{'=' * 60}")

    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    # f^* proxy via IRLS to convergence (used only as the gap reference; for
    # Family C we use this value as the diagnostic 'oracle' scale).
    irls_res = irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-14,
                    k_max=300, solver="cholesky", w0=w_ols)
    f_star = float(np.min(irls_res["f_vals"]))

    f_A = float(f_lasso(X, y, w_ols, lam))
    resid = X @ w_ols - y
    f_B = float(0.5 * resid @ resid)
    f_C = f_star

    print(f"f^* (IRLS-converged proxy) = {f_star:.6e}")
    print(f"Family A scale  f_LASSO(w_OLS) = {f_A:.6e}")
    print(f"Family B scale  f_OLS(w_OLS)   = {f_B:.6e}")
    print(f"Family C scale  f^*            = {f_C:.6e}")

    families = [("A", f_A), ("B", f_B), ("C", f_C)]
    records = []
    trajectories = {fam_id: {} for fam_id, _ in families}

    for fam_id, scale in families:
        print(f"\n--- Family {fam_id} ({name}):  delta_0 = c * {scale:.4e} ---")
        for c in C_VALUES:
            delta0 = c * scale
            res = deflected_subgradient(
                X, y, lam, w0=w_ols.copy(), i_max=I_MAX,
                beta=BETA, delta0=delta0, R=R_PATIENCE, rho=RHO,
                gamma_min=GAMMA_MIN, f_star=f_star,
            )
            gaps = np.asarray(res["gaps"], dtype=float)
            n_contr = _count_contractions(res["delta_hist"])
            final_gap = float(gaps[-1])
            trajectories[fam_id][c] = gaps
            records.append({
                "dataset":          name,
                "family":           fam_id,
                "c":                c,
                "delta0":           delta0,
                "scale":            scale,
                "final_record_gap": final_gap,
                "n_contractions":   n_contr,
                "n_iter":           int(res["n_iter"]),
            })
            print(f"  c={c:<5g}  delta0={delta0:.3e}  "
                  f"final_gap={final_gap:.3e}  contr={n_contr:3d}")

    return records, trajectories


def _plot_instance(ax_row, name, trajectories):
    families = [("A", r"$c\,f_{\mathrm{LASSO}}(w_{\mathrm{OLS}})$", RAMP_BLUES),
                ("B", r"$c\,f_{\mathrm{OLS}}(w_{\mathrm{OLS}})$",   RAMP_ORANGES),
                ("C", r"$c\,f^{*}$ (oracle)",                       RAMP_PURPLES)]
    for ax, (fam_id, fam_label, ramp) in zip(ax_row, families):
        cmap = plt.get_cmap(ramp)
        colors = cmap(np.linspace(0.35, 1.0, len(C_VALUES)))
        for c, color in zip(C_VALUES, colors):
            gaps = _safe_log(trajectories[fam_id][c])
            iters = np.arange(1, len(gaps) + 1)
            ax.loglog(iters, gaps, color=color, linewidth=1.5,
                      label=rf"$c={c:g}$")
        ax.set_title(rf"{name} -- Family {fam_id}: {fam_label}",
                     fontsize=11)
        ax.legend(loc="lower left", fontsize=8)
        style_axes(ax)
    ax_row[0].set_ylabel(rf"{name}: $\bar f^{{i}}-f^{{*}}$", fontsize=10)


def run() -> None:
    print("=" * 60)
    print("SGPTL: delta_0 family sweep (A / B / C) -- multi-instance")
    print("=" * 60)

    instances = []

    X_s, y_s, _, _, _ = make_lasso_problem(
        n=H_SYNTH, m=M_SYNTH, sparsity=SPARSITY, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    instances.append(("synthetic", X_s, y_s, LAMBDA, H_SYNTH))

    X_d, y_d, H_d = _load_real("diabetes")
    instances.append(("diabetes", X_d, y_d, LAMBDA, H_d))

    X_c, y_c, H_c = _load_real("california")
    instances.append(("california", X_c, y_c, LAMBDA, H_c))

    all_records = []
    traj_per_instance = {}
    for name, X, y, lam, H in instances:
        recs, trajs = _sweep_instance(name, X, y, lam, H)
        all_records.extend(recs)
        traj_per_instance[name] = trajs

    csv_path = os.path.join(TAB_DIR, "delta0_families.csv")
    if _HAS_PANDAS:
        pd.DataFrame(all_records).to_csv(csv_path, index=False)
    else:
        header = ("dataset,family,c,delta0,scale,final_record_gap,"
                  "n_contractions,n_iter")
        rows = [
            f"{r['dataset']},{r['family']},{r['c']},{r['delta0']:.6e},"
            f"{r['scale']:.6e},{r['final_record_gap']:.6e},"
            f"{r['n_contractions']},{r['n_iter']}"
            for r in all_records
        ]
        with open(csv_path, "w") as fh:
            fh.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nSaved CSV: {csv_path}")

    fig, axes = plt.subplots(3, 3, figsize=(16.0, 12.0), sharex=False)
    for row_idx, (name, _, _, _, _) in enumerate(instances):
        _plot_instance(axes[row_idx], name, traj_per_instance[name])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Iteration $i$  (log)")

    fig.suptitle(
        rf"SGPTL: $\delta_{{0}}$ family sweep across instances  "
        rf"($\rho={RHO}$, $\gamma_{{\min}}={GAMMA_MIN}$, "
        rf"$i_{{\max}}={I_MAX}$, warm start $w_{{\mathrm{{OLS}}}}$)",
        y=1.005, fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "delta0_families.pdf")
    fig.savefig(path)
    print(f"Saved figure: {path}")
    plt.close(fig)

    print("\n=== Aggregate: Family A vs Family C, by dataset and c ===")
    print(f"{'dataset':>12}  {'c':>6}  {'gap A':>10}  {'gap C':>10}  {'ratio A/C':>10}")
    for name, _, _, _, _ in instances:
        for c in C_VALUES:
            ga = next(r["final_record_gap"] for r in all_records
                      if r["dataset"] == name and r["family"] == "A" and r["c"] == c)
            gc = next(r["final_record_gap"] for r in all_records
                      if r["dataset"] == name and r["family"] == "C" and r["c"] == c)
            ratio = ga / gc if gc > 0 else float("nan")
            print(f"  {name:>10}  {c:>6.2g}  {ga:>10.3e}  {gc:>10.3e}  {ratio:>10.3f}")


if __name__ == "__main__":
    run()
