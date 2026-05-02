# P3 — Experiments

**Goal**: Re-run all experiments on the fixed code and produce final figures/tables for Chapter 5.

## Tasks

### T3.1 — Suppress macOS Accelerate spurious FPE warnings
NumPy's matmul on Apple Silicon's Accelerate BLAS reports spurious
"divide by zero / overflow / invalid" RuntimeWarnings. They are harmless
(verified by the test suite). Silenced at experiment-script entry only
(src/ stays clean for proper test diagnostics).

### T3.2 — `experiment_convergence.py`
- Removed the `w0=np.zeros(N)` override so DSM uses the OLS warm start
  (report § 3.4), making IRLS and DSM truly comparable on the same instance.
- Output: gap-vs-iter, gap-vs-time, DSM monotonicity-vs-record figures.

### T3.3 — `experiment_comparison.py`
- Reduced problem to (n=50, m=200) to give DSM realistic accuracy targets
  to reach within 30k iterations (its O(1/ε²) rate makes finer accuracy
  infeasible — itself a finding to discuss in Ch. 5).
- Added ε=1e-1 row to the table (the regime where DSM IS competitive).

### T3.4 — `experiment_params.py`
- IRLS sweep over ε_thr and λ — unchanged.
- DSM ρ sweep: switched to cold start `w_0 = 0` so the patience contraction
  mechanism actually triggers; otherwise the OLS warm start makes ρ moot.
  Now reports number of contractions per ρ for transparency.

### T3.5 — `experiment_scalability.py`
- Range `n ∈ {50, 100, 500, 1000, 2000}`. Dropped n=3000 (DSM gap meaningless,
  IRLS time prohibitive on a laptop).
- Two log-log plots: total time and per-iteration time.

### T3.6 — Verification
- All 4 scripts run without errors.
- Output figures and tables saved under `results/`.
- Numbers in CSVs are sane (no NaN, no negative gap).

## Findings recorded for Chapter 5

| Metric | Value |
|---|---|
| IRLS gap at iter 100, n=100 m=300 | 1.5e-6 |
| DSM gap at iter 8000, same problem | 3.9e-2 |
| IRLS iter to reach 1e-6, n=50 m=200 | 101 |
| DSM iter to reach 1e-6, same | did not reach |
| IRLS sparsity, λ=0.1 (true 88%) | 86% |
| DSM sparsity, same | 0% (needs post-thresholding) |
| Scalability IRLS time at n=2000 | 1.96 s |
| Scalability DSM time at n=2000 | 16.2 s (gap=1.0) |
| ε_thr below which IRLS plateaus | 1e-12 (FP limit) |
| DSM δ₀ sweet spot | 0.05·f* |
| DSM ρ effect on final gap (cold start) | none — only changes contraction count |

## Out of scope (deferred to potential P3.5)

- CVXPY cross-check: not installed; sklearn already serves as off-the-shelf reference.
- Real ELM datasets (diabetes, california housing): comando.pdf §4.5 says synthetic data is sufficient; report's focus is optimization quality not learning quality.
- Cold vs warm start for DSM head-to-head plot: would be valuable; only documented qualitatively in Ch. 5.
