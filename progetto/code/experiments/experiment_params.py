"""
experiment_params.py
--------------------
Parameter sensitivity for IRLS (eps_thr, lambda) and SGPTL (delta_0, rho).
Produces params_irls.pdf and params_dsm.pdf for Chapter 5.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import irls, deflected_subgradient, make_lasso_problem
from _plot_style import (apply_style, style_axes,
                         RAMP_BLUES, RAMP_REDS, RAMP_ORANGES, RAMP_PURPLES)
apply_style()


SEED  = 42
H, M  = 100, 400
NOISE = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _safe_log(arr, floor=1e-16):
    return np.maximum(np.asarray(arr, dtype=float), floor)


def run() -> None:
    print("=" * 60)
    print("Parameter sensitivity")
    print("=" * 60)

    LAMBDA = 0.1
    X, y, _, f_star, _ = make_lasso_problem(
        n=H, m=M, sparsity=0.1, noise_std=NOISE, lam=LAMBDA, random_state=SEED,
    )
    print(f"Problem: H={H}, M={M}, lambda={LAMBDA}, f*={f_star:.6f}\n")

    # ------------------------------------------------------------------
    # IRLS — eps_thr and lambda
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    print("--- IRLS: varying eps_thr ---")
    eps_thr_vals = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    cmap = plt.get_cmap(RAMP_BLUES)
    colors = cmap(np.linspace(0.40, 1.0, len(eps_thr_vals)))
    for eps_thr, color in zip(eps_thr_vals, colors):
        res = irls(X, y, LAMBDA, eps_thr=eps_thr, eps_stop=1e-12,
                   k_max=100, solver="cholesky", f_star=f_star)
        sparsity = np.mean(np.abs(res["w"]) < 1e-6)
        gaps = _safe_log(res["gaps"])
        label = (rf"$\varepsilon_{{\mathrm{{thr}}}}=10^{{{int(np.log10(eps_thr))}}}$"
                 f"  (sparsity {sparsity:.0%})")
        axes[0].semilogy(gaps, color=color, linewidth=1.4, label=label)
        print(f"  eps_thr={eps_thr:.0e}: gap={gaps[-1]:.2e}, "
              f"sparsity={sparsity:.0%}")

    axes[0].set_xlabel(r"Iteration $k$")
    axes[0].set_ylabel(r"$f(w_k) - f^{*}$")
    axes[0].set_title(r"IRLS: effect of $\varepsilon_{\mathrm{thr}}$")
    axes[0].legend(loc="upper right", fontsize=8)
    style_axes(axes[0])

    print("\n--- IRLS: varying lambda ---")
    lam_vals = [0.01, 0.05, 0.1, 0.5, 1.0]
    cmap = plt.get_cmap(RAMP_REDS)
    colors = cmap(np.linspace(0.30, 1.0, len(lam_vals)))
    for lam, color in zip(lam_vals, colors):
        Xl, yl, _, fs_l, _ = make_lasso_problem(
            n=H, m=M, sparsity=0.1, noise_std=NOISE, lam=lam, random_state=SEED,
        )
        res = irls(Xl, yl, lam, eps_thr=1e-8, eps_stop=1e-12,
                   k_max=100, solver="cholesky", f_star=fs_l)
        sparsity = np.mean(np.abs(res["w"]) < 1e-6)
        gaps = _safe_log(res["gaps"])
        label = rf"$\lambda={lam:g}$  (sparsity {sparsity:.0%})"
        axes[1].semilogy(gaps, color=color, linewidth=1.4, label=label)
        print(f"  lambda={lam}: gap={gaps[-1]:.2e}, sparsity={sparsity:.0%}")

    axes[1].set_xlabel(r"Iteration $k$")
    axes[1].set_ylabel(r"$f(w_k) - f^{*}$")
    axes[1].set_title(r"IRLS: effect of $\lambda_{\mathrm{LASSO}}$")
    axes[1].legend(loc="upper right", fontsize=8)
    style_axes(axes[1])

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "params_irls.pdf")
    fig.savefig(path)
    print(f"Saved: {path}\n")
    plt.close(fig)

    # ------------------------------------------------------------------
    # SGPTL — delta0 and rho
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    print("--- SGPTL: varying delta_0 ---")
    delta0_factors = [0.01, 0.05, 0.1, 0.5, 1.0]
    cmap = plt.get_cmap(RAMP_ORANGES)
    colors = cmap(np.linspace(0.30, 1.0, len(delta0_factors)))
    for factor, color in zip(delta0_factors, colors):
        res = deflected_subgradient(
            X, y, LAMBDA, i_max=5000, beta=1.0,
            delta0=factor * f_star, rho=0.9, f_star=f_star,
        )
        gaps = _safe_log(res["gaps"])
        label = rf"$\delta_{{0}}={factor:g}\,f^{{*}}$"
        axes[0].semilogy(gaps, color=color, linewidth=1.2, label=label)
        print(f"  delta0={factor}*f*={factor*f_star:.4f}: gap={gaps[-1]:.2e}")

    axes[0].set_xlabel(r"Iteration $i$")
    axes[0].set_ylabel(r"$\bar{f}^{i} - f^{*}$")
    axes[0].set_title(r"SGPTL: effect of $\delta_{0}$")
    axes[0].legend(loc="upper right", fontsize=8)
    style_axes(axes[0])

    print("\n--- SGPTL: varying rho (cold start to expose effect) ---")
    rho_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
    cmap = plt.get_cmap(RAMP_PURPLES)
    colors = cmap(np.linspace(0.30, 1.0, len(rho_vals)))
    n_dim = X.shape[1]
    for rho, color in zip(rho_vals, colors):
        res = deflected_subgradient(
            X, y, LAMBDA, w0=np.zeros(n_dim),
            i_max=10000, beta=1.0,
            delta0=0.1 * f_star, rho=rho, R=200.0, f_star=f_star,
        )
        gaps = _safe_log(res["gaps"])
        n_contr = int(np.sum(np.diff(res["delta_hist"]) < 0))
        label = rf"$\rho={rho:g}$  ({n_contr} contr.)"
        axes[1].semilogy(gaps, color=color, linewidth=1.2, label=label)
        print(f"  rho={rho}: gap={gaps[-1]:.2e}, contractions={n_contr}")

    axes[1].set_xlabel(r"Iteration $i$")
    axes[1].set_ylabel(r"$\bar{f}^{i} - f^{*}$")
    axes[1].set_title(r"SGPTL: effect of $\rho$")
    axes[1].legend(loc="upper right", fontsize=8)
    style_axes(axes[1])

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "params_dsm.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    run()
