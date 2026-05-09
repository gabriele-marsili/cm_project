"""Parameter sweeps: IRLS (eps_thr, lambda, Cholesky vs CG) and SGPTL (delta_0, rho)."""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

from src import irls, deflected_subgradient, make_lasso_problem
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         RAMP_BLUES, RAMP_REDS, RAMP_ORANGES, RAMP_PURPLES,
                         SIZE_DOUBLE, SIZE_SINGLE, COLOR_IRLS, COLOR_FCUR)
apply_style()


SEED  = 42
H, M  = 100, 400
NOISE = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    return np.maximum(np.asarray(arr, dtype=float), floor)


def run() -> None:
    print("=" * 60)
    print("Parameter sensitivity & Solver Analysis")
    print("=" * 60)

    LAMBDA = 0.1
    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE,
        lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}\n")

    w_ols = solve_spd(X.T @ X + 1e-12 * np.eye(H), X.T @ y, method="cholesky")

    # ---- IRLS sweeps ----
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    print("--- IRLS: varying eps_thr ---")
    eps_vals = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    cmap = plt.get_cmap(RAMP_BLUES)
    colors = cmap(np.linspace(0.40, 1.0, len(eps_vals)))
    for eps_thr, color in zip(eps_vals, colors):
        res = irls(X, y, LAMBDA, eps_thr=eps_thr, eps_stop=1e-12,
                   k_max=100, solver="cholesky", w0=w_ols, f_star=f_star)
        sparsity = np.mean(np.abs(res["w"]) < 1e-6)
        gaps = _safe_log(res["gaps"])
        label = (rf"$\varepsilon_{{\mathrm{{thr}}}}=10^{{{int(np.log10(eps_thr))}}}$"
                 f"  (sp. {sparsity:.0%})")
        axes[0].semilogy(gaps, color=color, linewidth=1.8, label=label)
        print(f"  eps_thr={eps_thr:.0e}: gap={gaps[-1]:.2e}, "
              f"sparsity={sparsity:.0%}")

    axes[0].set_xlabel(r"Iteration $k$")
    axes[0].set_ylabel(r"$f(w_k) - f^{*}$  (log scale)")
    axes[0].set_title(r"IRLS: effect of $\varepsilon_{\mathrm{thr}}$")
    axes[0].legend(loc="upper right", fontsize=10)
    style_axes(axes[0])

    print("\n--- IRLS: varying lambda ---")
    lam_vals = [0.01, 0.05, 0.1, 0.5, 1.0]
    cmap = plt.get_cmap(RAMP_REDS)
    colors = cmap(np.linspace(0.30, 1.0, len(lam_vals)))
    for lam, color in zip(lam_vals, colors):
        Xl, yl, _, fs_l, _ = make_lasso_problem(
            n=H, m=M, sparsity=0.1, noise_std=NOISE,
            lam=lam, random_state=SEED,
        )
        w_ols_l = solve_spd(Xl.T @ Xl + 1e-12 * np.eye(H), Xl.T @ yl,
                            method="cholesky")
        res = irls(Xl, yl, lam, eps_thr=1e-8, eps_stop=1e-12,
                   k_max=100, solver="cholesky", w0=w_ols_l, f_star=fs_l)
        sparsity = np.mean(np.abs(res["w"]) < 1e-6)
        gaps = _safe_log(res["gaps"])
        label = rf"$\lambda={lam:g}$  (sp. {sparsity:.0%})"
        axes[1].semilogy(gaps, color=color, linewidth=1.8, label=label)
        print(f"  lambda={lam}: gap={gaps[-1]:.2e}, sparsity={sparsity:.0%}")

    axes[1].set_xlabel(r"Iteration $k$")
    axes[1].set_ylabel(r"$f(w_k) - f^{*}$  (log scale)")
    axes[1].set_title(r"IRLS: effect of $\lambda_{\mathrm{LASSO}}$")
    axes[1].legend(loc="upper right", fontsize=10)
    style_axes(axes[1])

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "params_irls.pdf")
    fig.savefig(path)
    print(f"Saved: {path}\n")
    plt.close(fig)

    # ---- SGPTL sweeps ----
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    print("--- SGPTL: varying delta_0 ---")
    delta0_factors = [0.01, 0.05, 0.1, 0.5, 1.0]
    cmap = plt.get_cmap(RAMP_ORANGES)
    colors = cmap(np.linspace(0.30, 1.0, len(delta0_factors)))
    for factor, color in zip(delta0_factors, colors):
        res = deflected_subgradient(
            X, y, LAMBDA, w0=w_ols, i_max=5000, beta=1.0,
            delta0=factor * f_star, rho=0.9, f_star=f_star,
        )
        gaps = _safe_log(res["gaps"])
        iters = np.arange(1, len(gaps) + 1)
        label = rf"$\delta_{{0}}={factor:g}\,f^{{*}}$"
        axes[0].loglog(iters, gaps, color=color, linewidth=1.8, label=label)
        print(f"  delta0={factor}*f*={factor*f_star:.4f}: gap={gaps[-1]:.2e}")

    axes[0].set_xlabel(r"Iteration $i$  (log scale)")
    axes[0].set_ylabel(r"$\bar{f}^{i} - f^{*}$  (log scale)")
    axes[0].set_title(r"SGPTL: effect of $\delta_{0}$")
    axes[0].legend(loc="lower left", fontsize=10)
    style_axes(axes[0])

    print("\n--- SGPTL: varying rho ---")
    rho_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
    cmap = plt.get_cmap(RAMP_PURPLES)
    colors = cmap(np.linspace(0.30, 1.0, len(rho_vals)))
    rho_records = []
    for rho, color in zip(rho_vals, colors):
        res = deflected_subgradient(
            X, y, LAMBDA, w0=w_ols, i_max=5000, beta=1.0,
            delta0=0.1 * f_star, rho=rho, f_star=f_star,
        )
        gaps = _safe_log(res["gaps"])
        iters = np.arange(1, len(gaps) + 1)
        n_contr = int(np.sum(np.diff(res["delta_hist"]) < 0))
        label = rf"$\rho={rho:g}$  ({n_contr} contr.)"
        axes[1].loglog(iters, gaps, color=color, linewidth=1.8, label=label)
        print(f"  rho={rho}: gap={gaps[-1]:.2e}, contractions={n_contr}")
        # record raw (unclipped) final gap for CSV
        raw_gaps = np.asarray(res["gaps"], dtype=float)
        rho_records.append({
            "rho": rho,
            "final_record_gap": float(raw_gaps[-1]),
            "n_iter": int(res["n_iter"]),
            "final_f": float(np.asarray(res.get("f_bar", res["gaps"]))[-1] + f_star),
        })

    # --- Save rho sweep results to CSV ---
    csv_path = os.path.join(TAB_DIR, "rho_sweep.csv")
    if _HAS_PANDAS:
        import pandas as pd
        pd.DataFrame(rho_records).to_csv(csv_path, index=False)
    else:
        header = "rho,final_record_gap,n_iter,final_f"
        rows = [f"{r['rho']},{r['final_record_gap']:.6e},{r['n_iter']},{r['final_f']:.6e}"
                for r in rho_records]
        with open(csv_path, "w") as fh:
            fh.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"Saved rho sweep CSV: {csv_path}")

    axes[1].set_xlabel(r"Iteration $i$  (log scale)")
    axes[1].set_ylabel(r"$\bar{f}^{i} - f^{*}$  (log scale)")
    axes[1].set_title(r"SGPTL: effect of $\rho$")
    axes[1].legend(loc="lower left", fontsize=10)
    style_axes(axes[1])

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "params_dsm.pdf")
    fig.savefig(path)
    print(f"Saved: {path}\n")
    plt.close(fig)

    # ---- IRLS Solver Comparison (Cholesky vs CG) ----
    # Two panels: gap vs iteration (left, shows the algorithmic difference)
    # and gap vs wall-clock time (right, log-log axes so Cholesky's ~10 ms
    # run is not crushed against the y-axis next to CG's ~100 ms one).
    print("\n--- IRLS: Solver Comparison (Cholesky vs CG) ---")
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    solvers = ["cholesky", "cg"]
    plot_colors = {"cholesky": COLOR_IRLS, "cg": COLOR_FCUR}
    runs = {}

    k_irls_solvers = 300 # both methods converge at 265 iterations
    for sol in solvers:
        res = irls(X, y, LAMBDA, eps_thr=1e-8, eps_stop=1e-12,
                   k_max=k_irls_solvers, solver=sol, w0=w_ols, f_star=f_star)
        exec_time = res["times"][-1]
        n_iters = res["n_iter"]
        sparsity = np.mean(np.abs(res["w"]) < 1e-6)
        converged = res["converged"]
        print(f"  Solver: {sol.upper()}")
        print(f"    Execution Time: {exec_time:.4f} s")
        print(f"    Iterations:     {n_iters} (Converged: {converged})")
        print(f"    Sparsity:       {sparsity:.2%}")
        runs[sol] = {"gaps": _safe_log(res["gaps"]), "times": np.asarray(res["times"], dtype=float), "exec_time": exec_time, "sparsity": sparsity}

    # Left panel: gap vs iteration (semilog y).
    ax = axes[0]
    for sol in solvers:
        gaps = runs[sol]["gaps"]
        iters = np.arange(len(gaps))
        ax.semilogy(iters, gaps, linewidth=2.0, color=plot_colors[sol],
                    label=f"{sol.upper()}  (sparsity {runs[sol]['sparsity']:.0%})")
    ax.set_xlabel(r"Iteration $k$")
    ax.set_ylabel(r"$f(w_k) - f^{*}$  (log scale)")
    ax.set_title("IRLS solver: convergence vs iterations")
    ax.legend(loc="upper right", fontsize=10)
    style_axes(ax)

    # Right panel: gap vs CPU time, log-log with shared warm-start marker so
    # Cholesky's 10 ms run is visible alongside CG's 100 ms run. We replace
    # the t=0 warm-start point with a small offset (half the smallest non-
    # zero measurement across both solvers) so the shared starting gap shows
    # on the log axis.
    ax = axes[1]
    min_pos = min(runs[s]["times"][1] for s in solvers
                  if len(runs[s]["times"]) > 1)
    t_start = min_pos / 2.0
    for sol in solvers:
        t = runs[sol]["times"].copy(); t[0] = t_start
        ax.loglog(t, runs[sol]["gaps"], linewidth=2.0,
                  color=plot_colors[sol],
                  label=f"{sol.upper()}  (total {runs[sol]['exec_time']*1000:.1f} ms)")
    ax.scatter([t_start], [runs["cholesky"]["gaps"][0]], s=80, marker="*",
               color="black", zorder=6, label="OLS warm start (shared)")
    ax.set_xlabel(r"CPU time (s)  (log scale)")
    ax.set_ylabel(r"$f(w_k) - f^{*}$  (log scale)")
    ax.set_title("IRLS solver: convergence vs CPU time")
    ax.legend(loc="lower left", fontsize=10)
    style_axes(ax)

    fig.suptitle(rf"IRLS inner solver: Cholesky vs CG  ($H={H}$, $M={M}$, "
                 rf"$\lambda_{{\mathrm{{LASSO}}}}={LAMBDA}$, "
                 rf"$\varepsilon_{{\mathrm{{thr}}}}=10^{{-8}}$, $k_{{\max}}={k_irls_solvers}$)",
                 y=1.00, fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "solver_comparison.pdf")
    fig.savefig(path)
    print(f"Saved: {path}\n")
    plt.close(fig)


if __name__ == "__main__":
    run()