"""Scalability of IRLS and SGPTL: analytic FLOP count vs measured wall-clock.

Two regimes are measured:

  * Synthetic, M = 5H: H and M grow together.
  * Real (california), fixed M: only H (the ELM hidden size) grows.

For each algorithm we report two quantities on the same H axis:

  * the analytic FLOP count of the operations the implementation actually runs
    (counted, not assumed, from the source; leading-order NLA constants after
    Golub & Van Loan 2013: GEMV 2mn, X^T X ~ m n^2, Cholesky n^3/3);
  * the median wall-clock over repeated runs.

The FLOP curve is the concrete instantiation of the per-iteration complexity
derived in the comparison chapter; overlaying it on wall-clock separates the
"theory -> FLOP" exponent from the "FLOP -> time" implementation gap. The
effective throughput (FLOP / wall-clock) makes that gap a measured quantity:
IRLS rises toward BLAS peak as H grows, SGPTL falls at the memory-bandwidth
cliff of its long Python iteration loop.
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

from src import irls, deflected_subgradient, make_lasso_problem
from src.elm import ELM
from src.lasso_utils import f_lasso
from src.linear_solvers import solve_spd
from _plot_style import (apply_style, style_axes,
                         COLOR_IRLS, COLOR_DSM, SIZE_DOUBLE)
apply_style()


SEED         = 42
LAMBDA       = 0.10
NOISE        = 0.05
M_RATIO      = 5
IRLS_KMAX    = 100
IRLS_EPSSTOP = 1e-8
DSM_IMAX     = 3000
N_REPEAT     = 5          # median over repeated timings smooths OS/cache jitter
COOLDOWN_S   = 3.0        # idle gap between timed problems to limit thermal drift

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


# --- analytic FLOP counts of the operations the implementations actually run ---
# Counted from src/irls.py and src/deflected_subgradient.py with leading-order
# NLA constants (Golub & Van Loan 2013): GEMV(m,n)=2mn, X^T X (m x n) ~ m n^2,
# Cholesky(n)=n^3/3, two triangular solves = 2 n^2.

def flops_irls(M: int, H: int, k: int) -> float:
    """Total FLOPs for k IRLS iterations.

    One-time: A = X^T X (~M H^2) and b = X^T y (2 M H).
    Per iteration: Cholesky (H^3/3) + two triangular solves (2 H^2) +
    one f_lasso residual (X w: 2 M H)."""
    one_time = M * H**2 + 2 * M * H
    per_iter = H**3 / 3.0 + 2 * H**2 + 2 * M * H
    return one_time + k * per_iter


def flops_sgptl(M: int, H: int, i: int) -> float:
    """Total FLOPs for i SGPTL iterations.

    Per iteration: subgradient X^T(Xw - y) (two GEMV, 4 M H) +
    one f_lasso residual (2 M H) + O(H) vector work."""
    per_iter = 6 * M * H
    return i * per_iter


def _blas_threads() -> str:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        v = os.environ.get(var)
        if v:
            return f"{var}={v}"
    return f"default (os.cpu_count={os.cpu_count()})"


def _median_time(fn, repeat=N_REPEAT):
    """Median wall-clock of `fn` over `repeat` runs (one warm-up discarded).

    Returns (median, rel_spread) where rel_spread = (max-min)/median quantifies
    run-to-run timing noise (thermal drift, OS scheduling)."""
    time.sleep(COOLDOWN_S)  # let the CPU settle before a timed batch
    fn()  # warm-up: page in arrays, prime the BLAS thread pool
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = float(np.median(samples))
    spread = (max(samples) - min(samples)) / med if med > 0 else float("nan")
    return med, spread


def _slope(H, y, frac=0.6) -> float:
    """Log-log slope fitted on the upper part of the range (overhead-free tail)."""
    H = np.asarray(H, float)
    y = np.asarray(y, float)
    k = max(2, int(np.ceil(frac * len(H))))
    mask = np.argsort(H)[-k:]
    return float(np.polyfit(np.log(H[mask]), np.log(y[mask]), 1)[0])


_w0_cache = {}


def _time_problem(X, y, f_star, f_w0):
    irls_fn = lambda: irls(
        X, y, LAMBDA, eps_thr=1e-8, eps_stop=IRLS_EPSSTOP,
        k_max=IRLS_KMAX, solver="cholesky", w0=_w0_cache["w0"], f_star=f_star)
    dsm_fn = lambda: deflected_subgradient(
        X, y, LAMBDA, w0=_w0_cache["w0"], i_max=DSM_IMAX,
        beta=1.0, delta0=0.1 * f_w0, rho=0.7, f_star=f_star)
    # Capture the actual iteration counts (early stopping makes them < the caps)
    # so the FLOP count reflects the work that was actually timed.
    k_irls = irls_fn()["n_iter"]
    k_dsm  = dsm_fn()["n_iter"]
    t_irls, sp_irls = _median_time(irls_fn)
    t_dsm, sp_dsm = _median_time(dsm_fn)
    return t_irls, t_dsm, k_irls, k_dsm, max(sp_irls, sp_dsm)


def _figure(rows, regime_label, xlabel, fname):
    """Two panels: (left) FLOP count vs wall-clock, (right) throughput."""
    Hs   = np.array([r["n"] for r in rows], float)
    M_of = np.array([r["m"] for r in rows], float)
    t_i  = np.array([r["t_irls"] for r in rows])
    t_d  = np.array([r["t_dsm"]  for r in rows])
    F_i  = np.array([r["flops_irls"] for r in rows])
    F_d  = np.array([r["flops_dsm"]  for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)

    # Left: measured wall-clock (solid) vs analytic FLOP count, the FLOP curve
    # anchored on the largest-H wall-clock point (the clean, overhead-free end)
    # so the asymptotic-slope comparison is read without small-H timing noise.
    ax = axes[0]
    ax.loglog(Hs, t_i, color=COLOR_IRLS, marker="o", markersize=7,
              linewidth=2.0, label="IRLS wall-clock")
    ax.loglog(Hs, t_d, color=COLOR_DSM, marker="s", markersize=7,
              linewidth=2.0, label="SGPTL wall-clock")
    ax.loglog(Hs, t_i[-1] * F_i / F_i[-1], color=COLOR_IRLS, linestyle="--",
              alpha=0.6, linewidth=1.3, label="IRLS FLOP count")
    ax.loglog(Hs, t_d[-1] * F_d / F_d[-1], color=COLOR_DSM, linestyle="--",
              alpha=0.6, linewidth=1.3, label="SGPTL FLOP count")
    si_t, si_F = _slope(Hs, t_i), _slope(Hs, F_i)
    sd_t, sd_F = _slope(Hs, t_d), _slope(Hs, F_d)
    ax.annotate(rf"fit $H^{{{si_t:.2f}}}$ vs FLOP $H^{{{si_F:.2f}}}$",
                xy=(Hs[-1], t_i[-1]), xytext=(-4, -16),
                textcoords="offset points", color=COLOR_IRLS, fontsize=8, ha="right")
    ax.annotate(rf"fit $H^{{{sd_t:.2f}}}$ vs FLOP $H^{{{sd_F:.2f}}}$",
                xy=(Hs[-1], t_d[-1]), xytext=(-4, 8),
                textcoords="offset points", color=COLOR_DSM, fontsize=8, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median CPU time (s)  (log scale)")
    ax.set_title(f"FLOP count vs wall-clock ({regime_label})")
    ax.legend(loc="upper left", fontsize=8)
    style_axes(ax)

    # Right: effective throughput = analytic FLOPs / measured time.
    ax = axes[1]
    ax.loglog(Hs, F_i / t_i / 1e9, color=COLOR_IRLS, marker="o", markersize=7,
              linewidth=2.0, label="IRLS")
    ax.loglog(Hs, F_d / t_d / 1e9, color=COLOR_DSM, marker="s", markersize=7,
              linewidth=2.0, label="SGPTL")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Effective throughput (GFLOP/s)  (log scale)")
    ax.set_title("FLOP / wall-clock")
    ax.legend(loc="best", fontsize=8)
    style_axes(ax)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"  IRLS  : wall-clock H^{si_t:.2f}  |  FLOP H^{si_F:.2f}")
    print(f"  SGPTL : wall-clock H^{sd_t:.2f}  |  FLOP H^{sd_F:.2f}")


def _save_csv(path, rows) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved: {path}")


def run_synthetic() -> None:
    print("=" * 60)
    print(f"Synthetic scalability (M=5H)   [{_blas_threads()}]")
    print("=" * 60)
    rows = []
    for H in [50, 100, 500, 1000, 2000]:
        M = M_RATIO * H
        print(f"H={H:5d}, M={M:6d} ...", end=" ", flush=True)
        X, y, _, f_star, _ = make_lasso_problem(
            n=H, m=M, sparsity=0.1, noise_std=NOISE, lam=LAMBDA,
            random_state=SEED)
        _w0_cache["w0"] = solve_spd(X.T @ X + 1e-12 * np.eye(H),
                                    X.T @ y, method="cholesky")
        f_w0 = float(f_lasso(X, y, _w0_cache["w0"], LAMBDA))
        t_irls, t_dsm, k_i, k_d, spread = _time_problem(X, y, f_star, f_w0)
        print(f"IRLS {t_irls:.4f}s ({k_i} it) | SGPTL {t_dsm:.4f}s ({k_d} it) "
              f"| spread {spread:.1%}")
        rows.append({"n": H, "m": M, "t_irls": t_irls, "t_dsm": t_dsm,
                     "iter_irls": k_i, "iter_dsm": k_d,
                     "flops_irls": flops_irls(M, H, k_i),
                     "flops_dsm": flops_sgptl(M, H, k_d)})
    _save_csv(os.path.join(TAB_DIR, "scalability.csv"), rows)
    _figure(rows, "synthetic, $M=5H$",
            r"$H$ (hidden size, $M=5H$)  (log scale)", "scalability.pdf")


def run_real(name="california") -> None:
    print("=" * 60)
    print(f"Real-data neuron scaling on {name} (fixed M)   [{_blas_threads()}]")
    print("=" * 60)
    from experiment_real_data import load_dataset
    X_tr_raw, _, y_tr, _ = load_dataset(name)
    M, d_in = X_tr_raw.shape
    print(f"M={M}, input dim={d_in}")
    rows = []
    for H in [50, 100, 200, 500, 1000]:
        print(f"H={H:5d} ...", end=" ", flush=True)
        elm = ELM(d=d_in, p=H, activation="sigmoid", lam=LAMBDA, random_state=SEED)
        X = elm.transform(X_tr_raw)
        _w0_cache["w0"] = solve_spd(X.T @ X + 1e-10 * np.eye(H),
                                    X.T @ y_tr, method="cholesky")
        f_w0 = float(f_lasso(X, y_tr, _w0_cache["w0"], LAMBDA))
        t_irls, t_dsm, k_i, k_d, spread = _time_problem(X, y_tr, None, f_w0)
        print(f"IRLS {t_irls:.4f}s ({k_i} it) | SGPTL {t_dsm:.4f}s ({k_d} it) "
              f"| spread {spread:.1%}")
        rows.append({"n": H, "m": M, "t_irls": t_irls, "t_dsm": t_dsm,
                     "iter_irls": k_i, "iter_dsm": k_d,
                     "flops_irls": flops_irls(M, H, k_i),
                     "flops_dsm": flops_sgptl(M, H, k_d)})
    _save_csv(os.path.join(TAB_DIR, "scalability_real.csv"), rows)
    _figure(rows, f"{name}, fixed $M={M}$",
            rf"$H$ (hidden size, fixed $M={M}$)  (log scale)",
            "scalability_real.pdf")


if __name__ == "__main__":
    run_synthetic()
    run_real("california")
