"""
experiment_comparison.py
------------------------
Experiment 4d: Head-to-head comparison of IRLS (A1) vs DSM (A2).

Metrics:
  - Iterations to reach f(w) - f* <= epsilon (for several epsilon values)
  - CPU time to reach same accuracy
  - Solution quality: ||w - w_star||, sparsity pattern
  - Tests on several problem sizes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from src import irls, deflected_subgradient, make_lasso_problem
from src.lasso_utils import f_lasso

SEED = 42
LAM = 0.1
NOISE = 0.05

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
TAB_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def iters_to_reach(gaps, threshold):
    """First iteration index where gap <= threshold; or None."""
    for i, g in enumerate(gaps):
        if g <= threshold:
            return i
    return None


def time_to_reach(gaps, times, threshold):
    """CPU time at first iteration where gap <= threshold; or None."""
    for g, t in zip(gaps, times):
        if g <= threshold:
            return t
    return None


def run():
    print("=" * 60)
    print("Experiment: IRLS vs DSM Comparison")
    print("=" * 60)

    n, m = 100, 400
    X, y, _, f_star, w_star = make_lasso_problem(n=n, m=m, sparsity=0.1, noise_std=NOISE, lam=LAM, random_state=SEED)
    print(f"\nDetailed comparison: n={n}, m={m}, f*={f_star:.6f}")

    res_irls = irls(X, y, LAM, eps_thr=1e-8, eps_stop=1e-10, k_max=200, solver='cholesky', f_star=f_star)
    res_dsm = deflected_subgradient(X, y, LAM, i_max=10000, beta=1.0, delta0=0.1*f_star, rho=0.95, f_star=f_star)

    epsilons = [1e-2, 1e-3, 1e-4, 1e-6]
    print(f"\n{'epsilon':>10}  {'IRLS iters':>12}  {'IRLS time':>12}  {'DSM iters':>12}  {'DSM time':>12}")
    rows = []
    for eps in epsilons:
        i_irls = iters_to_reach(res_irls['gaps'], eps)
        t_irls = time_to_reach(res_irls['gaps'], res_irls['times'], eps)
        i_dsm  = iters_to_reach(res_dsm['gaps'],  eps)
        t_dsm  = time_to_reach(res_dsm['gaps'],  res_dsm['times'],  eps)
        print(f"{eps:>10.0e}  {str(i_irls):>12}  {str(round(t_irls,4)) if t_irls else 'N/A':>12}  {str(i_dsm):>12}  {str(round(t_dsm,4)) if t_dsm else 'N/A':>12}")
        rows.append([eps, i_irls, t_irls, i_dsm, t_dsm])

    tab_path = os.path.join(TAB_DIR, 'comparison_table.csv')
    with open(tab_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon', 'irls_iters', 'irls_time', 'dsm_iters', 'dsm_time'])
        writer.writerows(rows)
    print(f"\nSaved: {tab_path}")


    # plot: gap vs iterations for both
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # vs iterations
    ax = axes[0]
    irls_gaps = np.maximum(res_irls['gaps'], 1e-16)
    dsm_gaps = np.maximum(res_dsm['gaps'],  1e-16)
    ax.semilogy(irls_gaps, 'b-o', ms=4, lw=1.5, label='IRLS (A1)')
    ax.semilogy(dsm_gaps, 'r-', lw=1, label='DSM (A2)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f - f^*$')
    ax.set_title('Convergence vs iterations')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # vs CPU time
    ax = axes[1]
    ax.semilogy(res_irls['times'], irls_gaps, 'b-o', ms=4, lw=1.5, label='IRLS (A1)')
    ax.semilogy(res_dsm['times'],  dsm_gaps, 'r-', lw=1, label='DSM (A2)')
    ax.set_xlabel('CPU time (s)')
    ax.set_ylabel('$f - f^*$')
    ax.set_title('Convergence vs CPU time')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'IRLS vs DSM  (n={n}, m={m}, $\\lambda$={LAM})')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'comparison_irls_dsm.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


    # solution quality (sparsity)
    w_irls = res_irls['w']
    w_dsm = res_dsm['w']
    print(f"\nSolution quality:")
    print(f"  ||w_irls - w*|| = {np.linalg.norm(w_irls - w_star):.4e}")
    print(f"  ||w_dsm  - w*|| = {np.linalg.norm(w_dsm  - w_star):.4e}")
    print(f"  IRLS sparsity   = {np.mean(np.abs(w_irls) < 1e-6):.2%}")
    print(f"  DSM  sparsity   = {np.mean(np.abs(w_dsm)  < 1e-6):.2%}")
    print(f"  True sparsity   = {np.mean(np.abs(w_star) < 1e-6):.2%}")

    return res_irls, res_dsm


if __name__ == '__main__':
    run()
