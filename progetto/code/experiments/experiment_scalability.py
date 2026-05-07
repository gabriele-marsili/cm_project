"""
experiment_scalability.py
-------------------------
Experiment 4b: Scalability study.

Vary n (number of hidden neurons / features):
  n in {50, 100, 500, 1000, 5000}  with m = 5n

Measure:
  - Total CPU time for IRLS (fixed eps_stop tolerance)
  - Total CPU time for DSM (fixed iteration budget)
  - Time per iteration for each
  - Final gap achieved

Identifies the crossover point where DSM becomes competitive.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import irls, deflected_subgradient, make_lasso_problem, make_elm_problem
from src.lasso_utils import f_lasso

SEED = 42
LAM = 0.1
NOISE = 0.05
M_RATIO = 5 # m = M_RATIO * n

IRLS_KMAX = 100
IRLS_EPSSTOP = 1e-6
DSM_IMAX = 3000

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
TAB_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def run():
    print("=" * 60)
    print("Experiment: Scalability")
    print("=" * 60)

    n_values = [50, 100, 500, 1000, 3000]

    results = []
    for n in n_values:
        m = M_RATIO * n
        print(f"\nn={n:5d}, m={m:6d} ...", end=' ', flush=True)

        X, y, _, f_star, _ = make_lasso_problem(n=n, m=m, sparsity=0.1, noise_std=NOISE, lam=LAM, random_state=SEED)

        # IRLS
        t0 = time.perf_counter()
        res_i = irls(X, y, LAM, eps_thr=1e-8, eps_stop=IRLS_EPSSTOP, k_max=IRLS_KMAX, solver='cholesky', f_star=f_star)
        t_irls = time.perf_counter() - t0

        # DSM
        t0 = time.perf_counter()
        res_d = deflected_subgradient(X, y, LAM, i_max=DSM_IMAX, beta=1.0, delta0=0.1*f_star, rho=0.95, f_star=f_star)
        t_dsm = time.perf_counter() - t0

        gap_irls = res_i['gaps'][-1] if res_i['gaps'] else float('nan')
        gap_dsm = res_d['gaps'][-1] if res_d['gaps'] else float('nan')
        iter_irls = res_i['n_iter']
        iter_dsm = res_d['n_iter']

        print(f"IRLS {t_irls:.3f}s ({iter_irls} iter, gap={gap_irls:.1e})  |  DSM {t_dsm:.3f}s ({iter_dsm} iter, gap={gap_dsm:.1e})")

        results.append({
            'n': n, 'm': m,
            't_irls': t_irls, 'iter_irls': iter_irls, 'gap_irls': gap_irls,
            't_dsm':  t_dsm,  'iter_dsm':  iter_dsm,  'gap_dsm':  gap_dsm,
        })


    tab_path = os.path.join(TAB_DIR, 'scalability.csv')
    with open(tab_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {tab_path}")

    ns      = [r['n'] for r in results]
    t_irls  = [r['t_irls'] for r in results]
    t_dsm   = [r['t_dsm']  for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(ns, t_irls, 'b-o', label='IRLS')
    ax.loglog(ns, t_dsm,  'r-s', label='DSM')
    # reference O(n^3) line (cost of most linear equation solvers)
    ref = [t_irls[0] * (n / ns[0])**3 for n in ns]
    ax.loglog(ns, ref, 'b--', alpha=0.4, label='$O(n^3)$')
    ax.set_xlabel('$n$ (features / hidden neurons)')
    ax.set_ylabel('CPU time (s)')
    ax.set_title('Scalability: total time')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    t_irls_per = [r['t_irls'] / max(r['iter_irls'], 1) for r in results]
    t_dsm_per = [r['t_dsm']  / max(r['iter_dsm'],  1) for r in results]
    ax.loglog(ns, t_irls_per, 'b-o', label='IRLS (per iter)')
    ax.loglog(ns, t_dsm_per,  'r-s', label='DSM (per iter)')
    ax.set_xlabel('$n$')
    ax.set_ylabel('Time per iteration (s)')
    ax.set_title('Per-iteration cost')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'scalability.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    return results


if __name__ == '__main__':
    run()
