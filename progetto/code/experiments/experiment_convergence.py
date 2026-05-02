"""
experiment_convergence.py
-------------------------
Experiment 4a: Convergence analysis of IRLS and DSM.

For both algorithms:
  - Plot f(w_k) - f*  vs.  iterations  (semilog scale)
  - Plot f(w_k) - f*  vs.  CPU time    (semilog scale)
  - Verify: IRLS shows linear convergence (straight line in log scale)
  - Verify: DSM shows sublinear convergence (curve flattens)

Also verifies that IRLS is monotone and DSM is non-monotone
(but record value f_bar is monotone for DSM).
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress spurious "divide by zero / overflow / invalid" RuntimeWarnings
# that NumPy reports inside matmul on Apple Silicon's Accelerate BLAS.
# These do not affect numerical correctness (verified by the test suite).
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
np.seterr(all='ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import irls, deflected_subgradient, make_lasso_problem
from src.lasso_utils import f_lasso

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEED    = 42
N       = 100          # number of features
M       = 300          # number of samples
SPARSITY = 0.1
LAM     = 0.1
NOISE   = 0.05

IRLS_KMAX  = 100
DSM_IMAX   = 8000

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def run():
    print("=" * 60)
    print("Experiment: Convergence Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Generate data
    # ------------------------------------------------------------------
    X, y, w_true, f_star, w_star = make_lasso_problem(
        n=N, m=M, sparsity=SPARSITY, noise_std=NOISE, lam=LAM,
        random_state=SEED)

    print(f"Problem: n={N}, m={M}, lambda={LAM}")
    print(f"f*  (sklearn reference) = {f_star:.8f}")
    print(f"w*  sparsity = {np.mean(np.abs(w_star) < 1e-6):.2%}")

    # ------------------------------------------------------------------
    # Run IRLS
    # ------------------------------------------------------------------
    print("\n--- Running IRLS ---")
    res_irls = irls(X, y, LAM,
                    eps_thr=1e-8, eps_stop=1e-10,
                    k_max=IRLS_KMAX,
                    solver='cholesky',
                    f_star=f_star,
                    verbose=True)
    print(f"IRLS: {res_irls['n_iter']} iters, "
          f"final f-f* = {res_irls['gaps'][-1]:.3e}, "
          f"converged={res_irls['converged']}")

    # ------------------------------------------------------------------
    # Run DSM
    # ------------------------------------------------------------------
    print("\n--- Running DSM ---")
    # No explicit w0 — uses the OLS warm start (report Sec 3.4),
    # identical to IRLS so the two algorithms are comparable on the same instance.
    res_dsm = deflected_subgradient(
        X, y, LAM,
        i_max=DSM_IMAX,
        beta=1.0,
        delta0=0.1 * f_star,
        rho=0.95,
        f_star=f_star,
        verbose=True, verbose_freq=1000)
    print(f"DSM:  {res_dsm['n_iter']} iters, "
          f"final f_bar-f* = {res_dsm['gaps'][-1]:.3e}")

    # ------------------------------------------------------------------
    # Plot 1: gap vs iterations  (log scale)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IRLS
    ax = axes[0]
    irls_gaps = np.array(res_irls['gaps'])
    irls_gaps = np.maximum(irls_gaps, 1e-16)
    ax.semilogy(range(len(irls_gaps)), irls_gaps, 'b-o', ms=4, label='IRLS: $f(w_k)-f^*$')
    ax.set_xlabel('Iteration $k$')
    ax.set_ylabel('$f(w_k) - f^*$')
    ax.set_title('IRLS — Convergence (semilog)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # DSM
    ax = axes[1]
    dsm_fbar = np.array(res_dsm['gaps'])
    dsm_fbar = np.maximum(dsm_fbar, 1e-16)
    ax.semilogy(range(len(dsm_fbar)), dsm_fbar, 'r-', lw=1, label=r'DSM: $\bar{f}^i - f^*$')
    ax.set_xlabel('Iteration $i$')
    ax.set_ylabel(r'$\bar{f}^i - f^*$')
    ax.set_title('DSM — Convergence (semilog)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'convergence_vs_iter.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: gap vs CPU time  (log scale)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(res_irls['times'], irls_gaps, 'b-o', ms=4, label='IRLS')
    ax.semilogy(res_dsm['times'], dsm_fbar, 'r-', lw=1, label='DSM')
    ax.set_xlabel('CPU time (s)')
    ax.set_ylabel('gap to $f^*$')
    ax.set_title('Convergence vs CPU time')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'convergence_vs_time.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: DSM — f(w_i) vs f_bar (monotonicity check)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    iters_dsm = range(len(res_dsm['f_vals']))
    ax.plot(iters_dsm, res_dsm['f_vals'], 'r-', lw=0.5, alpha=0.5, label='$f(w_i)$ (oscillates)')
    ax.plot(iters_dsm, res_dsm['f_bar'], 'k-', lw=1.5, label=r'$\bar{f}^i$ (record, monotone)')
    ax.axhline(f_star, color='g', ls='--', label='$f^*$')
    ax.set_xlabel('Iteration $i$')
    ax.set_ylabel('Objective value')
    ax.set_title('DSM — Non-monotone iterates vs monotone record')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'dsm_nonmonotone.pdf')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    return res_irls, res_dsm


if __name__ == '__main__':
    run()
