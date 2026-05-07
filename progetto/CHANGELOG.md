# Changelog

All notable changes to the CM 646AA Project 25 ML (ELM + LASSO) implementation
and report. Dates are in `YYYY-MM-DD`.

## 2026-05-07 — Real-data validation experiment

Added an experiment that validates IRLS and SGPTL on real regression datasets,
to complement the synthetic study. The optimisation gap stays meaningful on
real data because `sklearn.linear_model.Lasso` at `tol=1e-12` provides a
high-precision $f^{*}$ reference; the support-recovery question is the only
piece that requires synthetic data and is not attempted here.

### Added

- **`progetto/code/experiments/experiment_real_data.py`** — End-to-end
  pipeline:
  1. Load `diabetes` and `california` from `sklearn.datasets`;
  2. 80/20 train/test split with feature and target standardisation fit on
     the train split only;
  3. ELM transformation $\sigma(\mathbf{X}\mathbf{W}_{1}^{\top})$ with $H=200$
     sigmoid units and a fixed random $\mathbf{W}_{1}$;
  4. Reference solution via `sklearn-Lasso` at `alpha = lam/M`,
     `tol=1e-12`, `max_iter=1e5`;
  5. OLS warm start shared by IRLS (100 iter, `eps_thr=1e-8`) and SGPTL
     (8000 iter, `delta0=0.1*f*`, `rho=0.9`);
  6. Closed-form Ridge baseline at the same regularisation strength on
     the quadratic term;
  7. Reports objective value, sparsity (count of components below `1e-6`)
     and held-out test MSE for each method.
- Outputs:
  `progetto/code/results/tables/real_data.csv` and
  `progetto/code/results/figures/real_data_convergence.pdf`.

### Changed — report

- **§5.7 Validation on real datasets** (new section, ~1.5 pages,
  Table 5.3 and Figure 5.8) — Documents the real-data results:
  - IRLS reaches the sklearn precision floor on both datasets in
    $\le 100$ iterations and returns a marginally lower $f$ than
    sklearn does at its `1e-12` tolerance ($52.949$ vs $52.950$ on
    diabetes; $2358.02$ vs $2360.83$ on California);
  - IRLS-recovered sparsity tracks sklearn within a few percentage
    points (14% vs 18% on diabetes, 4% vs 2% on California);
  - On diabetes the L1 mechanism gives a clear test-MSE gain over
    Ridge ($0.898$ vs $0.942$); on California the three methods are
    within rounding noise;
  - SGPTL converges on California ($M=16512$, $M\gg H$) but stalls on
    diabetes ($M=354$, comparable to $H=200$, $\text{cond}\sim 10^{6}$),
    consistent with the $O(\varepsilon^{-2})$ rate when the problem
    becomes underdetermined.
- **Chapter 6 Conclusions / Limitations** — Replaced the previous "we
  did not test on real datasets" caveat with a pointer to §5.7 and a
  brief summary of the regime-split result.

### Notes

- Test MSE is computed on the held-out 20% split with the standardised
  target; the units are therefore variance-of-y on the train split.
- The two datasets ship with sklearn (California is downloaded once on
  first run via `fetch_california_housing`); no external data is
  required.

## 2026-05-07 — Post-merge correctness pass

Critical algorithmic bugs introduced by the previous merge were identified and
fixed. All four experiments were re-run on the corrected code and Chapters 5
and 6 of the report were updated with the actual measurements.

### Fixed — algorithms (`progetto/code/src/`)

- **`irls.py`** — Normal-equations coefficient was `2*lam * D_k`. With
  `f(w) = ½‖Xw−y‖² + λ‖w‖₁` this makes the IRLS fixed point satisfy the KKT of
  the **2λ** problem, not the **λ** problem (factor-of-2 off in the
  regularisation), and breaks the monotone-decrease property. Replaced with
  `lam * D_k`, derived from the AM-GM majorisation
  `½(wᵢ²/|w_{k,i}| + |w_{k,i}|) ≥ |wᵢ|`. Result: IRLS now converges to the
  correct optimum and `f(w_k)` is monotone non-increasing as the theory
  requires.

- **`data_generation.py` — `make_lasso_problem`** — sklearn α was `lam/(2m)`,
  but the correct mapping between sklearn's `(1/(2M))‖Xw−y‖² + α‖w‖₁` and our
  `(1/2)‖Xw−y‖² + λ‖w‖₁` is `α = λ/M`. The buggy α made `f_star` an upper
  bound on the true minimum, masking SGPTL's actual convergence behaviour.
  Fixed to `lam / m` (matches report §5.1 specification).

- **`data_generation.py` — `make_elm_problem`** — sklearn α was
  `lam * m / 2.0` (the source code even carried a `# ??????????` comment).
  Fixed to `lam / m`, identical scaling to `make_lasso_problem`.

- **`deflected_subgradient.py` — `_optimal_gamma`** — When `d_prev = 0` the
  closed-form formula returned `γ = 0`, producing `d = 0` and stalling the
  algorithm. The fallback explicitly resets `d_{i−1} ← 0` to force a pure
  subgradient step (γ = 1), so `_optimal_gamma` now short-circuits to `1.0`
  whenever `‖d_prev‖² < 1e-30`, matching the `i = 0` convention.

- **`deflected_subgradient.py` — stepsize-restricted Polyak** — The report
  §3.2 specifies `β_i = min(β, γ_i)` so that the Polyak numerator shrinks
  when the deflection collapses. The code used a fixed `β = 1`, which let the
  current iterate diverge in non-monotone phases and produced unbounded
  Polyak steps. Implemented `beta_i = min(beta, gamma)`.

- **`deflected_subgradient.py` — iteration-count fallback** — The fallback
  described in report §5.1 (contract `δ` and reset `d_{i−1} ← 0` when the
  record value has not improved for `R_iter = max(i_max/100, 50)` consecutive
  iterations) was missing. Implemented and placed at the **top** of the main
  loop so that it counts every iteration, including those that hit the
  `numerator ≤ 0` or NaN-safety branches.

### Fixed — experiment scripts (`progetto/code/experiments/`)

- **`experiment_convergence.py`**, **`experiment_comparison.py`**,
  **`experiment_scalability.py`** — All three scripts previously left `w0`
  unset, which defaulted to `zeros` (cold start) for SGPTL. Report §5.2 calls
  for the OLS warm start `w₀ = (XᵀX)⁻¹Xᵀy` shared with IRLS. The three scripts
  now compute `w_ols` explicitly and pass it to both algorithms.

- **`experiment_scalability.py`** — `rho` was `0.9`; report §5.5 specifies
  `ρ = 0.95`. Corrected.

### Changed — report (`progetto/report/5_results/`, `6_conclusions/`)

The previous numbers in Chapter 5 were artefacts of the buggy `f_star`
reference. They have been replaced with the actual measurements collected on
the corrected code. The qualitative story (IRLS linear vs SGPTL sublinear,
IRLS dominates the project's intended `M ≫ H` regime, etc.) is unchanged.

- **§5.2 Convergence behaviour** — SGPTL final record gap on the convergence
  instance is **1.7·10⁻²** (was `4.5·10⁻⁶`); the wall-clock comparison was
  updated accordingly. The report now states that pushing the SGPTL gap
  below `10⁻⁴` is out of reach inside any reasonable budget on this
  instance.

- **§5.3.2 SGPTL: δ₀ and ρ** — Re-derived from the corrected sweeps.
  - δ₀ sweep produces a U-shape with `0.01 f*` failing to escape the warm
    start (gap 3.2), `0.05 f*` reaching `7.4·10⁻³`, `0.1 f*` reaching
    `4.6·10⁻³` (best), `0.5 f*` and `1.0 f*` deteriorating to ~`3.5·10⁻²`.
  - ρ sweep on cold start: ρ ∈ [0.5, 0.95] all converge to the same
    `1.7·10⁻²` plateau; `ρ = 0.99` is the slight outlier at `8.5·10⁻³`. The
    text now explains that under OLS warm start the iteration-count fallback
    fixes δ's effective schedule and ρ becomes effectively muted.

- **§5.4 Solution quality and sparsity** — SGPTL solution distance at the
  `10⁻⁶` tolerance is **`‖w_dsm − w*‖₂ = 4.5·10⁻²`** (was `5.9·10⁻⁴`) with
  **0%** exact sparsity (was `74%`). On the moderate instance SGPTL does not
  reach `10⁻¹` within the 30 000-iteration budget after the OLS warm start
  switch.

- **§5.5 Scalability** — Table 5.1 fully updated:

  | H    | M     | t_IRLS  | gap_IRLS  | t_SGPTL | gap_SGPTL |
  |-----:|------:|--------:|----------:|--------:|----------:|
  |   50 |   250 | 0.003 s | 5.1·10⁻⁷  | 0.028 s | 5.4·10⁻³  |
  |  100 |   500 | 0.004 s | 1.8·10⁻⁷  | 0.035 s | 4.4·10⁻²  |
  |  500 |  2500 | 0.048 s | 1.8·10⁻⁶  | 0.288 s | 9.0·10⁻³  |
  | 1000 |  5000 | 0.299 s | 7.6·10⁻⁶  | 5.269 s | 8.0·10⁻²  |
  | 2000 | 10000 | 1.701 s | 1.0·10⁻⁵  | 22.82 s | 1.4·10⁻¹  |

  IRLS still tracks the `O(MH²) = O(H³)` reference at `M = 5H`. SGPTL gap
  is two-to-four orders of magnitude looser across the whole range; the
  non-monotone gap-vs-H pattern is annotated in the text as the
  instance-to-instance variance of a sublinear method at fixed iteration
  budget.

- **§5.6 Iterations to a target accuracy** — Table 5.2 updated. Both
  algorithms hit `ε = 10⁻¹` in single-digit iterations; **SGPTL fails to
  reach `10⁻²`** within 30 000 iterations on the moderate problem (was
  `2126` iterations in the buggy run). IRLS reaches `10⁻⁶` in 101
  iterations as before.

- **Chapter 6 — Conclusions.** Updated the two paragraphs that quoted
  specific numbers: the convergence-instance gap comparison
  (`1.5·10⁻⁶` IRLS vs **`1.7·10⁻²`** SGPTL, four orders of magnitude apart,
  not three) and the iterations-to-ε contrast (IRLS reaches `10⁻²` in 3
  iterations versus SGPTL's failure to do so within the budget). Removed
  the `700×` ratio claim as the cleaner failure-to-converge framing tells
  the same story more honestly. Also softened the ρ-sweep narrative to
  reflect the U-shape we actually observed.

### Verified

- `python test_basic.py` — 17/17 pass on the corrected code.
- `latexmk -pdf -bibtex main.tex` — clean build, 45 pages.
- All four experiment scripts run end-to-end and regenerate the figures and
  tables in `progetto/code/results/`.

### Tooling notes

The post-merge `irls.py`, `deflected_subgradient.py` and `data_generation.py`
were resolved by accepting incoming changes during conflict resolution. The
incoming version was the source of the IRLS-coefficient and α-mapping bugs;
they have been corrected on top of that resolution rather than reverted, so
the merge history is preserved.
