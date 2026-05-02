# P2 — Test suite

**Goal**: Replace the smoke-style `test_basic.py` with a pytest suite that gives genuine confidence the algorithms compute what the report claims.

**Deliverable**: `progetto/code/tests/` directory with one file per source module, runnable via `pytest progetto/code/tests/`. Old `test_basic.py` is kept but deprecated (renamed or marked as smoke).

**Out of scope**: Performance benchmarks (P3), cross-language validation, real datasets.

## Tasks

### T2.1 — Test layout & shared fixtures
- Create `progetto/code/tests/__init__.py`, `tests/conftest.py`.
- `conftest.py`: fixtures for synthetic LASSO problem at three sizes (small/medium/elm), seeded RNG, sklearn reference solution, warning filter for sklearn convergence noise.

### T2.2 — `test_lasso_utils.py`
- `test_f_lasso_value` — handcrafted (X, y, w) where f is computable by hand.
- `test_grad_smooth_finite_diff` — central FD at 5 random smooth points (all w_i ≠ 0).
- `test_subgradient_minimum_norm` — at w with some w_i = 0, the returned subgrad must satisfy |g_i| ≤ λ AND have minimum 2-norm in the subdifferential (i.e. s_i = 0 there is optimal).
- `test_check_optimality_zero_at_optimum` — feed sklearn's solution; KKT violation must be < 1e-6.
- `test_check_optimality_positive_off_optimum` — random w; violation must be > 0.

### T2.3 — `test_linear_solvers.py`
- `test_cholesky_residual` — random SPD; residual < 1e-10.
- `test_cg_matches_cholesky` — agreement < 1e-8.
- `test_cg_iterations_bounded` — CG returns within n iterations.
- `test_solve_spd_dispatch` — both methods yield equivalent solutions.

### T2.4 — `test_irls.py`
- `test_irls_monotone_decrease` — `f_vals` non-increasing on 5 seeded problems.
- `test_irls_convergence_to_sklearn` — final f within 1e-4 of sklearn f*.
- `test_irls_kkt_at_convergence` — KKT violation < 1e-3 on active components.
- `test_irls_sparsity_recovery` — when λ is large, support of result ⊆ support of sklearn solution (within `eps_thr`).
- `test_irls_warm_start` — passing user `w0` is honoured.
- `test_irls_solver_choice` — cholesky and cg yield same w within 1e-6.

### T2.5 — `test_dsm.py`
- `test_dsm_record_non_increasing` — `f_bar` monotone non-increasing.
- `test_dsm_record_converges_to_fstar` — final record value within 1e-2 of sklearn f* on a small problem within 5000 iterations.
- `test_dsm_warm_start_is_ols` — when `w0=None`, the first iterate equals OLS solution `(X^T X)^-1 X^T y` (within 1e-10).
- `test_dsm_beta_clipping_invariant` — assert that the algorithm never uses an effective stepsize multiplier > γ_i (instrument by exposing β_i values via a debug hook OR re-derive from output).
  - Pragmatic version: write a small white-box test that calls `_optimal_gamma` and verifies the clipping in isolation.
- `test_dsm_subgradient_property` — subgradient norm bounded by `‖X‖_2 · ‖Xw-y‖_2 + λ √n` (sanity).

### T2.6 — `test_elm.py`
- `test_elm_transform_shape_and_range` — sigmoid in (0,1), tanh in (-1,1), relu ≥ 0.
- `test_elm_fit_irls_predict` — predict shape + non-trivial output.
- `test_elm_fit_dsm_predict` — same with DSM solver.
- `test_elm_sparsity_property` — `n_active ≤ p`, sparsity ∈ [0,1].

### T2.7 — `test_data_generation.py`
- `test_make_lasso_problem_shapes_and_fstar_is_minimum` — perturb `w_star` randomly; `f_lasso(X,y,w_star+δ,λ) ≥ f_star` always.
- `test_make_elm_problem_consistency` — `f_star == f_lasso(X_hid, y, w_star, λ)`.

### T2.8 — `pytest.ini` and Makefile target
- `pytest.ini` at `progetto/code/pytest.ini` with `testpaths = tests`, `filterwarnings = ignore::sklearn.exceptions.ConvergenceWarning`, `markers` for slow tests.
- Optional: `Makefile` with `test`, `test-fast` targets.

### T2.9 — Run & verify
- `pytest -v progetto/code/tests/` — all green.
- Coverage spot-check: every public function in `src/` has at least one test.
- Commit.

## Verification gate (P2 → P3)

- `pytest progetto/code/tests/` exits 0.
- No flaky tests (run twice, both pass).
- Each src module has ≥ 1 dedicated test file.
