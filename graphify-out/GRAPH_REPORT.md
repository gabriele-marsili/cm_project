# Graph Report - .  (2026-06-09)

## Corpus Check
- 45 files · ~811,528 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 367 nodes · 684 edges · 53 communities detected
- Extraction: 58% EXTRACTED · 42% INFERRED · 0% AMBIGUOUS · INFERRED: 286 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `ELM` - 76 edges
2. `f_lasso()` - 40 edges
3. `solve_spd()` - 29 edges
4. `deflected_subgradient()` - 28 edges
5. `irls()` - 27 edges
6. `make_lasso_problem()` - 23 edges
7. `style_axes()` - 15 edges
8. `run_one()` - 15 edges
9. `run_tests()` - 14 edges
10. `run_real()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `SGPTL sensitivity to the initial target gap delta_0: three families.  Family A:` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Solve LASSO via CVXPY interior-point (CLARABEL) to high precision.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Compute f^* as IRLS-converged value, cross-validated with CVXPY.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Family scales evaluated at the actual starting point w_0.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Long-run SGPTL rate verification on the ELM-transformed real datasets.  Runs SGP` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_sgptl_long_run.py → progetto/code/src/elm.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (38): ELM, Hidden-layer features: sigma(X_raw @ W1.T), shape (M, p)., Return yhat = transform(X_raw) @ w. Requires a prior fit()., Train/test split + standardisation fit on the train split only (no leakage)., Apply the ELM projection to both splits using the same fixed W_1., sklearn coordinate descent reference at moderate tolerance.      We use tol = 1e, IRLS-converged reference value (the same proxy used elsewhere)., Train/test split + standardisation fit on the train split only (no leakage). (+30 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (33): load_real_dataset(), Synthetic and real data generators for the experiments., Load a real regression dataset and split, scaling features on the train     spli, Algorithm A2: deflected subgradient with Polyak target level (SGPTL).  Iteration, ELM with L1-regularised output layer., Project 25 — Extreme Learning Machine with L1 regularization. Source package., Algorithm A1: IRLS for the LASSO., check_optimality() (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (28): make_elm_problem(), make_lasso_problem(), Synthetic linear LASSO problem with a planted sparse w_true.      Steps: (1) dra, Synthetic ELM-transformed LASSO problem.      Like make_lasso_problem but the de, _n_contractions(), SGPTL: hyperparameter calibration for warm-start and cold-start separately.  Goa, run(), _run_one() (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (21): deflected_subgradient(), _optimal_gamma(), Closed-form minimiser of ||gamma*g + (1-gamma)*d_prev||^2 on [gamma_min, 1]., Deflected subgradient (SGPTL) for LASSO.      Args:         w0: initial iterate., Fit output weights w by solving the LASSO on transformed inputs.          solver, _bench_real(), _bench_synthetic(), _elm() (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (17): Experiment IRLS: warm start (using OLS) vs cold start (w = 0) on SYNTHETIC DATA, run(), Parameter sweeps: IRLS (eps_thr, lambda, Cholesky vs CG) and SGPTL (delta_0, rho, run(), _safe_log(), _n_contractions(), Playground: theory-pure SGPTL with warm (OLS) vs cold (w_0=0) start.  The theory, Count how many times delta was contracted (strictly decreasing step). (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (18): _cvxpy_fstar(), _load_california_elm(), Diagnostic: when does SGPTL actually move on california ELM?  The main delta_0 s, _run_one(), sweep_for_H(), irls(), Iteratively Reweighted Least Squares for LASSO (Algorithm A1).      Solves min (, Tests for irls.py: monotonicity, convergence, KKT, solver equivalence. (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (18): build_hidden(), load_dataset(), _mse(), ols_warm_start(), IRLS and SGPTL on diabetes and california_housing under an ELM transformation., Cholesky-based (X^T X + eps I)^{-1} X^T y., Compute an independent reference f^*.      Pattern: IRLS run to convergence (hig, Train/test split + standardisation fit on the train split only (no leakage). (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (16): _blas_threads(), _figure(), flops_irls(), flops_sgptl(), _median_time(), Scalability: total wall-clock and per-iteration cost across H = 50..2000, M = 5H, Log-log slope fitted on the upper part of the range (overhead-free tail)., Two panels: (left) FLOP count vs wall-clock, (right) throughput. (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.2
Nodes (15): cholesky_solve(), conjugate_gradient(), SPD solvers used by IRLS: dense Cholesky and Jacobi-preconditioned CG., Solve Q x = b for SPD Q via Cholesky factorization (Q = L L^T)., Preconditioned CG for SPD Q x = b. Returns (x, iters_done).      precond: M^{-1}, Solve SPD system Q x = b. method in {'cholesky', 'cg'}.      return_info=True re, solve_spd(), _random_spd() (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.2
Nodes (14): build_hidden(), load_dataset(), _n_contractions(), ols_warm_start(), SGPTL (Deflected Subgradient) on diabetes and california_housing: Warm Start vs, Compute an independent reference f^* (IRLS-converged + CVXPY validation).      T, Train/test split + standardisation fit on the train split only (no leakage)., Apply the ELM projection to both splits using the same fixed W_1. (+6 more)

### Community 10 - "Community 10"
Cohesion: 0.24
Nodes (12): _count_contractions(), _f_star_cvxpy(), _instance_f_star(), _load_real_elm(), SGPTL sensitivity to the initial target gap delta_0: three families.  Family A:, Solve LASSO via CVXPY interior-point (CLARABEL) to high precision., Compute f^* as IRLS-converged value, cross-validated with CVXPY., Family scales evaluated at the actual starting point w_0. (+4 more)

### Community 11 - "Community 11"
Cohesion: 0.31
Nodes (10): build_hidden(), load_dataset(), long_run(), main(), plot(), Long-run SGPTL rate verification on the ELM-transformed real datasets.  Runs SGP, IRLS-converged reference value (the same proxy used elsewhere)., reference_fstar() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.27
Nodes (8): elm_problem(), LassoProblem, _make(), medium_problem(), Shared pytest fixtures for the CM Project 25 ML test suite.  Three reference pro, Full ELM pipeline reference problem., small_problem(), NamedTuple

### Community 13 - "Community 13"
Cohesion: 0.39
Nodes (8): build_hidden(), load_dataset(), ols_warm_start(), IRLS on diabetes and california_housing: Warm Start vs Cold Start., reference_solution(), run(), run_one(), _split_scale()

### Community 14 - "Community 14"
Cohesion: 0.28
Nodes (8): first_index_under(), first_time_under(), IRLS vs SGPTL on a moderate problem (H=50, M=200, lam=0.1): iterations and CPU t, Index of the first gap value that fell to or below ``threshold``., Wall-clock time at which the gap first fell to or below ``threshold``., run(), Precision/recall/F1 of supp(w_hat) against supp(w_true) at threshold tol.      C, support_metrics()

### Community 15 - "Community 15"
Cohesion: 0.43
Nodes (7): build_elm(), fstar_from_cache(), load_problem(), main(), Rerun SGPTL on real ELM-LASSO until record gap to f^* drops below 1e-6.  Address, run_until_crossing(), split_scale()

### Community 16 - "Community 16"
Cohesion: 0.39
Nodes (7): build_elm(), fstar_from_cache(), main(), Rerun IRLS on real ELM-LASSO until f - f^* < 1e-6, with full-setup wall-time.  A, Wall-time of (OLS warm-start + IRLS loop) until the RELATIVE gap     (f - f*)/|f, split_scale(), time_irls_full()

### Community 17 - "Community 17"
Cohesion: 0.4
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Count how many times delta was contracted (strictly decreasing step).

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Draw one panel of the SGPTL long-run figure on `ax`.      Args:         ks: iter

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): default w0 = (X^T X)^-1 X^T y.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>     for w'

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): f computed by hand on a tiny example.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Central finite-difference of f at smooth points (all w_i != 0).

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): If Xw = y, smooth gradient is exactly zero.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): At w with all w_i != 0, the subgradient is the actual gradient.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): At w=0, min-norm subgradient is grad_smooth - clip(grad_smooth, -lam, lam).

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): MM property: Q majorises f, so f(w_{k+1}) <= Q(w_{k+1}, w_k) <= f(w_k).

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Large lambda should drive most components inside [-eps_thr, eps_thr].

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Cholesky and CG must yield equivalent-quality solutions.      The two iterate se

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Preconditioned CG for SPD Q x = b. Returns (x, iters_done).      precond: M^{-1}

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Solve SPD system Q x = b. method in {'cholesky', 'cg'}.      return_info=True re

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Iteratively Reweighted Least Squares for LASSO (Algorithm A1).      Solves min (

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Hidden-layer features: sigma(X_raw @ W1.T), shape (M, p).

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Fit output weights w by solving the LASSO on transformed inputs.          solver

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Return yhat = transform(X_raw) @ w. Requires a prior fit().

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Closed-form minimiser of ||gamma*g + (1-gamma)*d_prev||^2 on [gamma_min, 1].

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Deflected subgradient (SGPTL) for LASSO.      Args:         w0: initial iterate.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Count how many times delta was contracted (strictly decreasing step).

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Solve SPD system Q x = b. method in {'cholesky', 'cg'}.      return_info=True re

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>     for w'

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Large lambda should drive most components inside [-eps_thr, eps_thr].

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Cholesky and CG must yield equivalent-quality solutions.      The two iterate se

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Precision/recall/F1 of the support of w_hat against w_true at threshold tol.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Max KKT violation. tol selects which components are treated as zero.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Closed-form minimiser of ||gamma g + (1-gamma) d_prev||^2 on [gamma_min, 1].

## Knowledge Gaps
- **99 isolated node(s):** `IRLS vs SGPTL on a moderate problem (H=50, M=200, lam=0.1): iterations and CPU t`, `Index of the first gap value that fell to or below ``threshold``.`, `Wall-clock time at which the gap first fell to or below ``threshold``.`, `Experiment IRLS: warm start (using OLS) vs cold start (w = 0) on SYNTHETIC DATA`, `Re-plot the SGPTL long-run figure from the saved CSV (no re-compute).  experimen` (+94 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (1 nodes): `dispensa_CM.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `manuale_teorico.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `guida_teorica.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `main.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `report.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `main.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Count how many times delta was contracted (strictly decreasing step).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Draw one panel of the SGPTL long-run figure on `ax`.      Args:         ks: iter`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `default w0 = (X^T X)^-1 X^T y.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>     for w'`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `f computed by hand on a tiny example.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Central finite-difference of f at smooth points (all w_i != 0).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `If Xw = y, smooth gradient is exactly zero.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `At w with all w_i != 0, the subgradient is the actual gradient.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `At w=0, min-norm subgradient is grad_smooth - clip(grad_smooth, -lam, lam).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `MM property: Q majorises f, so f(w_{k+1}) <= Q(w_{k+1}, w_k) <= f(w_k).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Large lambda should drive most components inside [-eps_thr, eps_thr].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Cholesky and CG must yield equivalent-quality solutions.      The two iterate se`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Preconditioned CG for SPD Q x = b. Returns (x, iters_done).      precond: M^{-1}`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Solve SPD system Q x = b. method in {'cholesky', 'cg'}.      return_info=True re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Iteratively Reweighted Least Squares for LASSO (Algorithm A1).      Solves min (`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Hidden-layer features: sigma(X_raw @ W1.T), shape (M, p).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Fit output weights w by solving the LASSO on transformed inputs.          solver`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Return yhat = transform(X_raw) @ w. Requires a prior fit().`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Closed-form minimiser of ||gamma*g + (1-gamma)*d_prev||^2 on [gamma_min, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Deflected subgradient (SGPTL) for LASSO.      Args:         w0: initial iterate.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Count how many times delta was contracted (strictly decreasing step).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Solve SPD system Q x = b. method in {'cholesky', 'cg'}.      return_info=True re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>     for w'`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Large lambda should drive most components inside [-eps_thr, eps_thr].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Cholesky and CG must yield equivalent-quality solutions.      The two iterate se`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Precision/recall/F1 of the support of w_hat against w_true at threshold tol.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Max KKT violation. tol selects which components are treated as zero.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Closed-form minimiser of ||gamma g + (1-gamma) d_prev||^2 on [gamma_min, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ELM` connect `Community 0` to `Community 1`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 15`, `Community 16`?**
  _High betweenness centrality (0.247) - this node is a cross-community bridge._
- **Why does `f_lasso()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 14`, `Community 15`, `Community 16`?**
  _High betweenness centrality (0.166) - this node is a cross-community bridge._
- **Why does `deflected_subgradient()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 10`, `Community 11`, `Community 14`, `Community 15`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 70 inferred relationships involving `ELM` (e.g. with `test_basic.py ------------- Quick sanity checks for all modules. Run from the co` and `SGPTL sensitivity to the initial target gap delta_0: three families.  Family A:`) actually correct?**
  _`ELM` has 70 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `f_lasso()` (e.g. with `run_tests()` and `run()`) actually correct?**
  _`f_lasso()` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `solve_spd()` (e.g. with `run()` and `run()`) actually correct?**
  _`solve_spd()` has 25 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `deflected_subgradient()` (e.g. with `run_tests()` and `run()`) actually correct?**
  _`deflected_subgradient()` has 25 INFERRED edges - model-reasoned connections that need verification._