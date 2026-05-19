# Graph Report - .  (2026-05-17)

## Corpus Check
- 37 files · ~751,809 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 232 nodes · 461 edges · 22 communities detected
- Extraction: 57% EXTRACTED · 43% INFERRED · 0% AMBIGUOUS · INFERRED: 200 edges (avg confidence: 0.77)
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

## God Nodes (most connected - your core abstractions)
1. `ELM` - 40 edges
2. `f_lasso()` - 30 edges
3. `deflected_subgradient()` - 24 edges
4. `make_lasso_problem()` - 21 edges
5. `solve_spd()` - 20 edges
6. `irls()` - 19 edges
7. `run_tests()` - 14 edges
8. `run_one()` - 12 edges
9. `style_axes()` - 11 edges
10. `run()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Smoke test: sensitivity of SGPTL to rho on small synthetic and real ELM instance` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/smoke_rho.py → progetto/code/src/elm.py
- `SGPTL sensitivity to the initial target gap delta_0: three families.  Family A:` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Solve LASSO via CVXPY interior-point (CLARABEL) to high precision.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Compute f^* as IRLS-converged value, cross-validated with CVXPY.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py
- `Family scales evaluated at the actual starting point w_0.` --uses--> `ELM`  [INFERRED]
  progetto/code/experiments/experiment_delta0_families.py → progetto/code/src/elm.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (29): make_elm_problem(), make_lasso_problem(), Synthetic and real data generators for the experiments., IRLS vs SGPTL convergence on a fixed instance (H=100, M=300, lam=0.1)., run(), _safe_log(), Empirical test of the gamma_min floor on ELM LASSO.  Hypothesis: with gamma_min, run() (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (26): ELM, build_hidden(), load_dataset(), ols_warm_start(), SGPTL (Deflected Subgradient) on diabetes and california_housing: Warm Start vs, Train/test split + standardisation fit on the train split only (no leakage)., Apply the ELM projection to both splits using the same fixed W_1., sklearn coordinate descent reference at moderate tolerance. (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (21): check_optimality(), grad_smooth(), Max KKT violation. tol selects which components are treated as zero., subgradient_f(), test_make_elm_problem_kkt_at_wstar(), At any w, subgradient_f returns g such that f(w') >= f(w) + <g, w'-w>     for w', test_dsm_subgradient_is_in_subdifferential(), Tests for lasso_utils.py. (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (7): Algorithm A2: deflected subgradient with Polyak target level (SGPTL).  Iteration, ELM with L1-regularised output layer., Project 25 — Extreme Learning Machine with L1 regularization. Source package., Algorithm A1: IRLS for the LASSO., optimality_gap(), LASSO objective f(w) = (1/2)||Xw-y||^2 + lam ||w||_1 and helpers., test_optimality_gap_zero_at_optimum()

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (14): build_hidden(), load_dataset(), _mse(), ols_warm_start(), IRLS and SGPTL on diabetes and california_housing under an ELM transformation., Cholesky-based (X^T X + eps I)^{-1} X^T y., Train/test split + standardisation fit on the train split only (no leakage)., Apply the ELM projection to both splits using the same fixed W_1. (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.26
Nodes (12): cholesky_solve(), conjugate_gradient(), SPD solvers used by IRLS: dense Cholesky and Jacobi-preconditioned CG., solve_spd(), _random_spd(), Tests for src/linear_solvers.py — Cholesky and Conjugate Gradient on SPD systems, In exact arithmetic CG converges in <= n iterations on SPD systems., test_cg_finite_termination_in_n_steps_exact() (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (12): _count_contractions(), _f_star_cvxpy(), _instance_f_star(), _load_real_elm(), SGPTL sensitivity to the initial target gap delta_0: three families.  Family A:, Solve LASSO via CVXPY interior-point (CLARABEL) to high precision., Compute f^* as IRLS-converged value, cross-validated with CVXPY., Family scales evaluated at the actual starting point w_0. (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.23
Nodes (12): irls(), Tests for irls.py: monotonicity, convergence, KKT, solver equivalence., MM property: Q majorises f, so f(w_{k+1}) <= Q(w_{k+1}, w_k) <= f(w_k)., Large lambda should drive most components inside [-eps_thr, eps_thr]., Cholesky and CG must yield equivalent-quality solutions.      The two iterate se, test_irls_converges_to_sklearn_fstar(), test_irls_kkt_residual_small(), test_irls_monotone_decrease() (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.23
Nodes (12): deflected_subgradient(), _optimal_gamma(), Closed-form minimiser of ||gamma g + (1-gamma) d_prev||^2 on [gamma_min, 1]., Tests for deflected_subgradient.py: gamma, monotonicity of f_bar, convergence., default w0 = (X^T X)^-1 X^T y., test_dsm_default_warmstart_is_ols(), test_dsm_record_converges_to_fstar(), test_dsm_record_non_increasing() (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.27
Nodes (8): elm_problem(), LassoProblem, _make(), medium_problem(), Shared pytest fixtures for the CM Project 25 ML test suite.  Three reference pro, Full ELM pipeline reference problem., small_problem(), NamedTuple

### Community 10 - "Community 10"
Cohesion: 0.28
Nodes (8): first_index_under(), first_time_under(), IRLS vs SGPTL on a moderate problem (H=50, M=200, lam=0.1): iterations and CPU t, Index of the first gap value that fell to or below ``threshold``., Wall-clock time at which the gap first fell to or below ``threshold``., run(), Precision/recall/F1 of the support of w_hat against w_true at threshold tol., support_metrics()

### Community 11 - "Community 11"
Cohesion: 0.46
Nodes (7): _bench_real(), _bench_synthetic(), _elm(), main(), _ols(), Smoke test: sensitivity of SGPTL to rho on small synthetic and real ELM instance, _split_scale()

### Community 12 - "Community 12"
Cohesion: 0.53
Nodes (5): _cvxpy_fstar(), _load_california_elm(), Diagnostic: when does SGPTL actually move on california ELM?  The main delta_0 s, _run_one(), sweep_for_H()

### Community 13 - "Community 13"
Cohesion: 0.53
Nodes (5): _n_contractions(), SGPTL: hyperparameter calibration for warm-start and cold-start separately.  Goa, run(), _run_one(), _sweep()

### Community 14 - "Community 14"
Cohesion: 0.4
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
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

## Knowledge Gaps
- **40 isolated node(s):** `IRLS vs SGPTL on a moderate problem (H=50, M=200, lam=0.1): iterations and CPU t`, `Index of the first gap value that fell to or below ``threshold``.`, `Wall-clock time at which the gap first fell to or below ``threshold``.`, `Empirical test of the gamma_min floor on ELM LASSO.  Hypothesis: with gamma_min`, `Parameter sweeps: IRLS (eps_thr, lambda, Cholesky vs CG) and SGPTL (delta_0, rho` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 15`** (1 nodes): `dispensa_CM.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `manuale_teorico.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `guida_teorica.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `main.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `report.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `main.toc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `f_lasso()` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `ELM` connect `Community 1` to `Community 3`, `Community 4`, `Community 6`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.184) - this node is a cross-community bridge._
- **Why does `deflected_subgradient()` connect `Community 8` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 10`, `Community 11`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.162) - this node is a cross-community bridge._
- **Are the 34 inferred relationships involving `ELM` (e.g. with `test_basic.py ------------- Quick sanity checks for all modules. Run from the co` and `Smoke test: sensitivity of SGPTL to rho on small synthetic and real ELM instance`) actually correct?**
  _`ELM` has 34 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `f_lasso()` (e.g. with `run_tests()` and `run()`) actually correct?**
  _`f_lasso()` has 28 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `deflected_subgradient()` (e.g. with `run_tests()` and `run()`) actually correct?**
  _`deflected_subgradient()` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `make_lasso_problem()` (e.g. with `run_tests()` and `run()`) actually correct?**
  _`make_lasso_problem()` has 20 INFERRED edges - model-reasoned connections that need verification._