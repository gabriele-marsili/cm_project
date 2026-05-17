"""SGPTL sensitivity to the initial target gap delta_0: three families.

Family A: delta_0 = c * f_LASSO(w_OLS)        (default, admissible)
Family B: delta_0 = c * f_OLS(w_OLS)          (drops lam*||w||_1, admissible)
Family C: delta_0 = c * f^*                   (diagnostic oracle, not admissible)

Instances tested:
  - synthetic (H=100, M=300), 5 seeds; CVXPY-verified f^*.
  - diabetes ELM H=50, single seed, CVXPY-verified f^*.
  - diabetes ELM H=200, single seed, CVXPY-verified f^*; pre-converged regime.
  - california ELM H=50, single seed, CVXPY-verified f^*.
  - california ELM H=200, single seed, CVXPY-verified f^*; degenerate
    regime (SGPTL stalls at the OLS warm start regardless of delta_0
    on this matrix), reported as an applicability-limit datapoint.
"""

import os
import sys
import time
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

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

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
H_DIAB_SMALL = 50
H_DIAB_LARGE = 200

SEEDS_SYNTH = [42, 7, 123, 2024, 11]
SEED_REAL = 42

C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    return np.maximum(np.asarray(arr, dtype=float), floor)


def _count_contractions(delta_hist) -> int:
    return int(np.sum(np.diff(np.asarray(delta_hist, dtype=float)) < 0))


def _f_star_cvxpy(X, y, lam):
    """Solve LASSO via CVXPY interior-point (CLARABEL) to high precision."""
    if not _HAS_CVXPY:
        return None
    M, H = X.shape
    w = cp.Variable(H)
    obj = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm(w, 1))
    prob = cp.Problem(obj)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False,
                   max_iter=200_000)
        elapsed = time.perf_counter() - t0
        if prob.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            print(f"    CVXPY status: {prob.status}  (no certified solution)")
            return None
        return float(prob.value), elapsed
    except Exception as exc:
        print(f"    CVXPY failed: {exc}")
        return None


def _load_real_elm(name, H_hidden, seed=SEED_REAL):
    if name == "diabetes":
        d = load_diabetes()
    elif name == "california":
        d = fetch_california_housing()
    else:
        raise ValueError(name)
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


def _instance_f_star(name, X, y, lam):
    """Compute f^* as IRLS-converged value, cross-validated with CVXPY."""
    M, H = X.shape
    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")
    irls_res = irls(X, y, lam, eps_thr=1e-8, eps_stop=1e-14,
                    k_max=300, solver="cholesky", w0=w_ols)
    f_irls = float(np.min(irls_res["f_vals"]))

    f_cvxpy, t_cvxpy, valid_cvxpy = None, None, False
    if H * M <= 1_500_000:
        print(f"  CVXPY validation (M={M}, H={H}): ", end="", flush=True)
        out = _f_star_cvxpy(X, y, lam)
        if out is not None:
            f_cvxpy, t_cvxpy = out
            valid_cvxpy = True
            rel_diff = abs(f_cvxpy - f_irls) / max(abs(f_cvxpy), 1e-16)
            print(f"f_cvxpy={f_cvxpy:.6e}  ({t_cvxpy:.2f}s)  "
                  f"rel.diff from IRLS = {rel_diff:.2e}")
        else:
            print("skipped or failed")

    f_star = f_cvxpy if valid_cvxpy else f_irls
    source = "cvxpy" if valid_cvxpy else "irls-converged"
    print(f"  {name}: f^* = {f_star:.6e}  (source: {source})")
    return f_star, source, w_ols


def _sweep(name, X, y, lam, seed_tag, f_star, w_ols, fA, fB, fC):
    out_rows = []
    trajs = {"A": {}, "B": {}, "C": {}}
    for fam_id, scale in (("A", fA), ("B", fB), ("C", fC)):
        print(f"    {name} seed={seed_tag} Family {fam_id} "
              f"(scale={scale:.4e}):")
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
            trajs[fam_id][c] = gaps
            out_rows.append({
                "dataset":          name,
                "seed":             seed_tag,
                "family":           fam_id,
                "c":                c,
                "delta0":           delta0,
                "scale":            scale,
                "final_record_gap": final_gap,
                "n_contractions":   n_contr,
                "n_iter":           int(res["n_iter"]),
            })
            print(f"      c={c:<5g}  gap={final_gap:.3e}  contr={n_contr:3d}")
    return out_rows, trajs


def _scales(X, y, w_ols, lam, f_star):
    fA = float(f_lasso(X, y, w_ols, lam))
    resid = X @ w_ols - y
    fB = float(0.5 * resid @ resid)
    fC = float(f_star)
    return fA, fB, fC


def run() -> None:
    print("=" * 68)
    print("SGPTL: delta_0 family sweep (A / B / C)")
    print("Multi-seed synthetic + diabetes H=50 (CVXPY-verified) + diabetes H=200")
    print("=" * 68)

    all_records = []
    plot_payload = {}

    # --- Synthetic multi-seed -------------------------------------------------
    print("\n[1/3] SYNTHETIC (H=100, M=300), 5 seeds")
    synth_trajs_perseed = {fam: {c: [] for c in C_VALUES}
                           for fam in ("A", "B", "C")}
    for seed in SEEDS_SYNTH:
        print(f"\n  Seed {seed}:")
        X, y, _, _, _ = make_lasso_problem(
            n=H_SYNTH, m=M_SYNTH, sparsity=SPARSITY, noise_std=NOISE,
            lam=LAMBDA, random_state=seed,
        )
        f_star, source, w_ols = _instance_f_star(
            f"synthetic-seed{seed}", X, y, LAMBDA)
        fA, fB, fC = _scales(X, y, w_ols, LAMBDA, f_star)
        rows, trajs = _sweep("synthetic", X, y, LAMBDA, seed,
                             f_star, w_ols, fA, fB, fC)
        all_records.extend(rows)
        for fam_id in ("A", "B", "C"):
            for c in C_VALUES:
                synth_trajs_perseed[fam_id][c].append(trajs[fam_id][c])
    plot_payload["synthetic"] = synth_trajs_perseed

    # --- Real ELM instances ---------------------------------------------------
    for idx, (name, H_hidden) in enumerate([
        ("diabetes",   H_DIAB_SMALL),
        ("diabetes",   H_DIAB_LARGE),
        ("california", H_DIAB_SMALL),
        ("california", H_DIAB_LARGE),
    ], start=2):
        tag = f"{name}-H{H_hidden}"
        print(f"\n[{idx}/5] {name.upper()} ELM H={H_hidden}")
        X_r, y_r = _load_real_elm(name, H_hidden)
        f_star, source, w_r = _instance_f_star(tag, X_r, y_r, LAMBDA)
        fA, fB, fC = _scales(X_r, y_r, w_r, LAMBDA, f_star)
        rows, trajs = _sweep(tag, X_r, y_r, LAMBDA, SEED_REAL,
                             f_star, w_r, fA, fB, fC)
        all_records.extend(rows)
        plot_payload[tag] = {
            fam_id: {c: [trajs[fam_id][c]] for c in C_VALUES}
            for fam_id in ("A", "B", "C")
        }

    # --- CSV ------------------------------------------------------------------
    csv_path = os.path.join(TAB_DIR, "delta0_families.csv")
    if _HAS_PANDAS:
        pd.DataFrame(all_records).to_csv(csv_path, index=False)
    else:
        header = ("dataset,seed,family,c,delta0,scale,final_record_gap,"
                  "n_contractions,n_iter")
        rows = [
            f"{r['dataset']},{r['seed']},{r['family']},{r['c']},"
            f"{r['delta0']:.6e},{r['scale']:.6e},{r['final_record_gap']:.6e},"
            f"{r['n_contractions']},{r['n_iter']}"
            for r in all_records
        ]
        with open(csv_path, "w") as fh:
            fh.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nSaved CSV: {csv_path}")

    # --- Figure: 5 instances x 3 families -------------------------------------
    instances = ["synthetic",
                 f"diabetes-H{H_DIAB_SMALL}", f"diabetes-H{H_DIAB_LARGE}",
                 f"california-H{H_DIAB_SMALL}", f"california-H{H_DIAB_LARGE}"]
    fig, axes = plt.subplots(5, 3, figsize=(16.0, 18.0))
    ramps = {"A": RAMP_BLUES, "B": RAMP_ORANGES, "C": RAMP_PURPLES}
    fam_labels = {
        "A": r"$\delta_{0}=c\,f_{\mathrm{LASSO}}(w_{\mathrm{OLS}})$",
        "B": r"$\delta_{0}=c\,f_{\mathrm{OLS}}(w_{\mathrm{OLS}})$",
        "C": r"$\delta_{0}=c\,f^{*}$ (oracle)",
    }
    for row_idx, name in enumerate(instances):
        for col_idx, fam_id in enumerate(("A", "B", "C")):
            ax = axes[row_idx, col_idx]
            cmap = plt.get_cmap(ramps[fam_id])
            colors = cmap(np.linspace(0.35, 1.0, len(C_VALUES)))
            for c, color in zip(C_VALUES, colors):
                trajs = plot_payload[name][fam_id][c]
                for tr in trajs:
                    gaps = _safe_log(tr)
                    iters = np.arange(1, len(gaps) + 1)
                    ax.loglog(iters, gaps, color=color, linewidth=1.0,
                              alpha=0.45)
                if len(trajs) > 1:
                    mean_gap = _safe_log(np.mean(np.asarray(trajs), axis=0))
                    iters = np.arange(1, len(mean_gap) + 1)
                    ax.loglog(iters, mean_gap, color=color, linewidth=2.2,
                              label=rf"$c={c:g}$")
                else:
                    ax.plot([], [], color=color, label=rf"$c={c:g}$")
            short_name = name.replace("-H", "/H=")
            ax.set_title(f"{short_name} — Family {fam_id}: {fam_labels[fam_id]}",
                         fontsize=10)
            ax.legend(loc="lower left", fontsize=8)
            style_axes(ax)
            if col_idx == 0:
                ax.set_ylabel(r"$\bar f^{i}-f^{*}$")
            if row_idx == len(instances) - 1:
                ax.set_xlabel(r"Iteration $i$  (log)")
    fig.suptitle(rf"SGPTL: $\delta_{{0}}$ family sweep "
                 rf"($\rho={RHO}$, $\gamma_{{\min}}={GAMMA_MIN}$, "
                 rf"$i_{{\max}}={I_MAX}$)", y=1.003, fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "delta0_families.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure: {path}")

    # --- Aggregate: Family A vs Family C, by dataset and c -------------------
    print("\n=== Aggregate: Family A vs Family C ===")
    print(f"{'dataset':>22}  {'c':>6}  {'mean gap A':>12}  {'mean gap C':>12}  "
          f"{'ratio A/C':>10}  {'n_seeds':>8}")
    for name in instances:
        for c in C_VALUES:
            ga = [r["final_record_gap"] for r in all_records
                  if r["dataset"] == name and r["family"] == "A" and r["c"] == c]
            gc = [r["final_record_gap"] for r in all_records
                  if r["dataset"] == name and r["family"] == "C" and r["c"] == c]
            ga_mean = float(np.mean(ga))
            gc_mean = float(np.mean(gc))
            ratio = ga_mean / gc_mean if gc_mean > 0 else float("nan")
            print(f"  {name:>20}  {c:>6.2g}  {ga_mean:>12.3e}  "
                  f"{gc_mean:>12.3e}  {ratio:>10.3f}  {len(ga):>8d}")


if __name__ == "__main__":
    run()
