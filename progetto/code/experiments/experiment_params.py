"""
experiment_params.py
--------------------
Experiment 4c: Effect of algorithmic parameters.

IRLS:
  - Effect of eps_thr in {1e-4, 1e-6, 1e-8, 1e-10, 1e-12}
    on convergence speed and sparsity

DSM:
  - Effect of delta0 (initial gap estimate)
  - Effect of rho (reduction rate)
  - Effect of R (patience threshold)
  - Effect of beta (step modulation)

Also studies the effect of lambda on sparsity and convergence.
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np
np.seterr(all='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import irls, deflected_subgradient, make_lasso_problem
from src.lasso_utils import f_lasso

SEED  = 42
N, M  = 100, 400
NOISE = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def run():
    print("=" * 60)
    print("Experiment: Parameter sensitivity")
    print("=" * 60)

    LAM = 0.1
    X, y, _, f_star, w_star = make_lasso_problem(
        n=N, m=M, sparsity=0.1, noise_std=NOISE, lam=LAM, random_state=SEED)
    print(f"Problem: n={N}, m={M}, lambda={LAM}, f*={f_star:.6f}")

    # ==================================================================
    # 1. IRLS: effect of eps_thr
    # ==================================================================
    print("\n--- IRLS: varying eps_thr ---")
    eps_thr_vals = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(eps_thr_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for eps_thr, color in zip(eps_thr_vals, colors):
        res = irls(X, y, LAM, eps_thr=eps_thr, eps_stop=1e-12,
                   k_max=100, solver='cholesky', f_star=f_star)
        gaps = np.maximum(res['gaps'], 1e-16)
        sparsity = np.mean(np.abs(res['w']) < 1e-6)
        label = f'$\\varepsilon_{{thr}}={eps_thr:.0e}$  (spar={sparsity:.0%})'
        axes[0].semilogy(gaps, color=color, lw=1.5, label=label)
        print(f"  eps_thr={eps_thr:.0e}: iters={res['n_iter']}, "
              f"gap={gaps[-1]:.2e}, sparsity={sparsity:.1%}")

    axes[0].set_xlabel('Iteration $k$')
    axes[0].set_ylabel('$f(w_k) - f^*$')
    axes[0].set_title('IRLS: effect of $\\varepsilon_{thr}$')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both', alpha=0.3)

    # ==================================================================
    # 2. Effect of lambda (shared plot)
    # ==================================================================
    print("\n--- Effect of lambda ---")
    lam_vals = [0.01, 0.05, 0.1, 0.5, 1.0]
    colors2 = plt.cm.Reds(np.linspace(0.3, 1.0, len(lam_vals)))

    for lam, color in zip(lam_vals, colors2):
        X_l, y_l, _, f_star_l, _ = make_lasso_problem(
            n=N, m=M, sparsity=0.1, noise_std=NOISE, lam=lam, random_state=SEED)
        res = irls(X_l, y_l, lam, eps_thr=1e-8, eps_stop=1e-10,
                   k_max=100, f_star=f_star_l)
        gaps = np.maximum(res['gaps'], 1e-16)
        sparsity = np.mean(np.abs(res['w']) < 1e-6)
        label = f'$\\lambda={lam}$ (spar={sparsity:.0%})'
        axes[1].semilogy(gaps, color=color, lw=1.5, label=label)
        print(f"  lambda={lam}: iters={res['n_iter']}, "
              f"gap={gaps[-1]:.2e}, sparsity={sparsity:.1%}")

    axes[1].set_xlabel('Iteration $k$')
    axes[1].set_ylabel('$f(w_k) - f^*$')
    axes[1].set_title('IRLS: effect of $\\lambda$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'params_irls.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()

    # ==================================================================
    # 3. DSM: effect of delta0
    # ==================================================================
    print("\n--- DSM: varying delta0 ---")
    delta0_factors = [0.01, 0.05, 0.1, 0.5, 1.0]
    colors3 = plt.cm.Oranges(np.linspace(0.3, 1.0, len(delta0_factors)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for factor, color in zip(delta0_factors, colors3):
        delta0 = factor * f_star
        res = deflected_subgradient(X, y, LAM, i_max=5000, beta=1.0,
                                    delta0=delta0, rho=0.95, f_star=f_star)
        gaps = np.maximum(res['gaps'], 1e-16)
        label = f'$\\delta_0 = {factor}\\,f^*$'
        axes[0].semilogy(gaps, color=color, lw=1.0, label=label)
        print(f"  delta0={factor}*f*={delta0:.4f}: final gap={gaps[-1]:.2e}")

    axes[0].set_xlabel('Iteration $i$')
    axes[0].set_ylabel(r'$\bar{f}^i - f^*$')
    axes[0].set_title('DSM: effect of $\\delta_0$')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both', alpha=0.3)

    # ==================================================================
    # 4. DSM: effect of rho.
    # We deliberately start from w_0 = 0 (NOT the OLS warm start used in
    # production) so that the patience mechanism is actively triggered
    # and the contraction factor rho has observable effect on the
    # trajectory.  With the OLS warm start the algorithm starts very
    # close to the optimum and rho would never play a measurable role.
    # ==================================================================
    print("\n--- DSM: varying rho (cold start w0=0 to expose the effect) ---")
    rho_vals = [0.5, 0.7, 0.9, 0.95, 0.99]
    colors4  = plt.cm.Purples(np.linspace(0.3, 1.0, len(rho_vals)))

    n_dsm = X.shape[1]
    for rho, color in zip(rho_vals, colors4):
        res = deflected_subgradient(
            X, y, LAM, w0=np.zeros(n_dsm),
            i_max=10000, beta=1.0,
            delta0=0.1*f_star, rho=rho, R=200.0, f_star=f_star,
        )
        gaps = np.maximum(res['gaps'], 1e-16)
        n_contractions = int(np.sum(np.diff(res['delta_hist']) < 0))
        label = f'$\\rho = {rho}$  ({n_contractions} contr.)'
        axes[1].semilogy(gaps, color=color, lw=1.0, label=label)
        print(f"  rho={rho}: final gap={gaps[-1]:.2e}, "
              f"contractions={n_contractions}")

    axes[1].set_xlabel('Iteration $i$')
    axes[1].set_ylabel(r'$\bar{f}^i - f^*$')
    axes[1].set_title('DSM: effect of $\\rho$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'params_dsm.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    print("\nDone.")


if __name__ == '__main__':
    run()
