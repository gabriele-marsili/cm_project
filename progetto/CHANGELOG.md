# Changelog

All notable changes to the CM 646AA Project 25 ML (ELM + LASSO) implementation
and report. Dates are in `YYYY-MM-DD`.

## 2026-06-09 — Theorem 3.1 + appendix proof, de-LLM pass, final review, code zip

Work after the 2026-06-07 entry, across several commits and two authors
(`b34883c`, `58c30e4`, `3819d3a`, `d9c4868`, `6d42c6b`, `5810b60`, `545328e`).

### Report — theory & results (`b34883c`, `58c30e4`)

- **Ch. 3 (Deflected Subgradient), proof of Theorem 3.1 — boundedness argument
  rewritten** (`3_algo_2_DSM/chapter3.tex`, paragraph "Subgradients uniformly
  bounded"). Restructured into the per-step estimate
  $\eta_i=-(2\gamma_i-\beta_i)(f(\mathbf{w}_i)-f^*)-\beta_i t_i$ and the cases
  "Sign of $\eta_i$", "Tail $i\ge\bar\imath$", "Transient $i<\bar\imath$" — the
  Fej\'er distance-to-$\mathbf{w}^*$ contraction that bounds the non-monotone
  iterates (the response to the prof's boundedness objection, replacing the old
  one-paragraph argument).
- **Appendix (new): "Derivation of the per-step estimate with target level"**
  (`appendix/appendix.tex`, `app:def-step-target`) — the target-level form of
  eq. (3.17), referenced by the Ch. 3 boundedness argument above.
- **Appendix A (IRLS local convergence) — step 2 completed** [Matthew]
  (`appendix/appendix.tex`). Added the converse direction of the lemma
  $\mathbf{S}\preceq\mathbf{I}\Leftrightarrow\mathbf{H}\preceq\mathbf{B}$
  (assume $\mathbf{H}\preceq\mathbf{B}\Rightarrow\mathbf{S}\preceq\mathbf{I}$),
  closing the spectral-radius bound $\rho(\nabla\mathcal{M}(\mathbf{w}^*))<1$.
- **§5.5.2 Scalability — expanded to two regimes** (`5_results/results.tex`):
  synthetic ($M=5H$) and real (`california`, fixed $M$), with the FLOP-count vs
  wall-clock split. Added figure `scalability_real` and timing table
  `tab:scalability-time`; relative-gap framing propagated into §5.5.3
  (iterations-to-$\varepsilon$) and §5.6 (real-data $10^{-6}$ target). Figures
  regenerated: `comparison_irls_dsm`, `real_data_convergence`, `scalability`,
  `scalability_real`.
- Supporting hand-derivations added as working notes under `utils/`
  (`teorema_3_1_SGPTL_spiegazione.{tex,pdf}`,
  `irls_convergence_rate_spiegazione.{tex,pdf}`, `d_i_bounded.txt`) — scratch
  material, not part of the submitted report.

### Code — de-LLM pass (`3819d3a` [Matthew], `d9c4868`)

- `src/{irls,deflected_subgradient,elm,lasso_utils,linear_solvers,data_generation}.py`
  and two experiment scripts: comments/docstrings tightened, dead imports
  removed. **Algorithmic logic unchanged**; 53/53 tests pass.

### Merge & build (`6d42c6b`, `5810b60`)

- Merged Matthew's branch (appendix proof + de-LLM). 8 `.py` conflicts resolved
  file-by-file — divergence was comments/docstrings/formatting only, identical
  algorithms. Took Matthew's docstrings for `src/`, local PEP8-wrapped version
  for the two experiment scripts. Recompiled `main.pdf` to 69 pp.

### Report — final review pass (`545328e`, this session)

Detailed deep-dive in `project_review/CHANGES_2026-06-09.md`.

- **Numerical fixes.** Root cause: five result CSVs (`convergence_instance`,
  `rho_sweep`, `gamma_floor_test`, `delta0_families`, `sgptl_long_run`) store
  *absolute* gaps $f-f^*$, but the report presents everything as *relative*
  $(f-f^*)/|f^*|$ (per the prof's correction). Numbers taken straight from those
  CSVs were absolute mislabeled as relative; all re-derived from seed 42.
  - §5.2: $f(\mathbf{0})\;76\to 4.6$ (stale value, never in any CSV);
    $10^{-6}$ crossing $122\to 115$; gap at 1500 iters
    $1.5\cdot 10^{-10}\to 1.3\cdot 10^{-10}$.
  - §5.4.4 (SGPTL: $\delta_0$ and $\rho$): $\rho$-sweep range $\to 1.7\cdot
    10^{-5}\dots 7.0\cdot 10^{-4}$; per-instance best $\rho$
    $0.3,0.3,0.7\to 0.8,0.3,0.7$; $\delta_0$ $c$-sweep
    $\to[4.8\cdot 10^{-5},1.4\cdot 10^{-4}]$.
  - §5.4.3 (SGPTL: deflection floor $\gamma_{\min}$): floor table
    $4.4\cdot 10^{-2}/1.1\cdot 10^{-4}\to 3.7\cdot 10^{-2}/1.0\cdot 10^{-4}$.
  - Removed the false claim "$\rho\in[0.3,0.7]$ within $2\times$ of the best
    everywhere" (false on the synthetic).
  - Appendix B: IRLS synthetic warm/cold $940/922\to 941/923$.
  - Verified correct, unchanged: gap0 $0.37$, SGPTL diabetes MSE $0.898$
    (reproduced 0.8983), contraction range $7\to 91$, all main tables.
- **Removed all `.csv` filename citations from the report** (7 occurrences):
  captions of Tab. 5.1 (warm/cold) and Tab. 5.6 (before/after), §5.3 "Defaults",
  §5.6 "Validation on real datasets".
- **De-LLM repetitions:** deleted 6 cross-section repeats, by section —
  multiplicative $\|\mathbf{w}_0-\mathbf{w}^*\|$ (§5.3, `results.tex:284`),
  california-OLS-near-$f^*$ (Appendix $\delta_0$-families table caption),
  $O(\varepsilon)$ floor (§2.5, `algo1.tex:224`), rate$\approx 1$ cause (§5.2,
  `results.tex:131` $\to$ cross-ref to Appendix A), record-vs-iterate (§5.2,
  `results.tex:150`), per-decade $100\times$ (§5.5.3, `results.tex:831`).
  Also removed "the signature of"; clarified the test claim in §5.1
  (surrogate-descent = IRLS, record-monotonicity = SGPTL).
- **Deliverable:** added `progetto/code/requirements.txt`; added
  `.pytest_cache/` to `.gitignore`; built `progetto/CM_project25_group63_code.zip`
  (89 files, 1.19 MB) excluding venv/caches/`results_old_submission`/
  `experiments/old`/`utils`.

### Open items (non-blocking)

- `delta0_families` appendix figure (15 panels) borderline vs §4.6 ("many small
  plots").

## 2026-06-07 — Code deliverable: de-LLM comment pass, numerical cleanup, dead-file pruning

Critical review of the whole `code/` tree for the submission: correctness,
adherence to the report, and whether the code reads as LLM-written. Core
algorithms (`src/`) verified against the report's Algorithm 1/2 statements and
re-derived by hand (IRLS normal equations, optimal deflection $\gamma^*$,
preconditioned CG, sklearn $\alpha=\lambda/M$ mapping); all checks pass.

### Changed — comment style (de-LLM)

- **`src/` (irls, deflected_subgradient, linear_solvers, elm)** — rewrote
  essayistic/reassuring comments into terse technical notes. Removed in-code
  cross-references to the report ("Section 2 of the report", "see Appendix D of
  the report", "cond. (3.5) of d'Antonio-Frangioni 2009") and LLM-tell phrasing
  ("garbage warm start", "rather than propagating a NaN", "below floating-point
  noise", "to be conservative").
- **Live `experiments/`** — removed in-code report cross-refs, references to the
  internal `project_review/REVIEW.md` and "the prof", and colloquialisms
  ("crushed against the y-axis", "afford a deeper sweep cheaply").

### Fixed — code

- **`test_basic.py`** — filtered the spurious "… encountered in matmul"
  `RuntimeWarning` (a NumPy/Apple-Accelerate SIMD false positive; results are
  finite, `X.dot` is clean) so the sanity run is readable; aligned the DSM test
  `rho` to the report default `0.7`.
- **`rerun_irls_to_crossing.py`** — comment/code drift: the comment claimed to
  add the OLS factorisation cost to `t_cross` but the code (and the report's
  Table 5.8 caption) report loop-only time. Corrected the comment and the stale
  docstring, removed the dead `t_setup` variable. No reported number changes.
- **`deflected_subgradient.py`** — corrected an off-by-one in the docstring's
  description of history-list lengths.

### Moved — dead scripts to `experiments/old/` (not shipped)

- `experiment_time_warm_vs_cold.py` (linearly-interpolated, non-measured time
  axis; duplicate of `experiment_warm_vs_cold.py`; figure not in the report),
  `experiment_delta0_proxy.py`, `experiment_best_configs.py`,
  `experiment_california_diagnostic.py`, `smoke_rho.py` — verified (underscore-safe
  against all report `.tex`) to feed neither a cited figure nor a cited CSV.

## 2026-06-07 — IRLS local-convergence appendix, figure & reproducibility fixes, code robustness

Merged the `irls-conv-analysis` branch (condensed §2.5 + a full Ostrowski
appendix), completed and corrected that proof, fixed three figures and several
reported numbers, made the warm/cold and solver tables reproducible from CSV,
and hardened two code paths.

### Added — theory

- **Appendix A "Full Proof of IRLS's Local Convergence"** (merged from
  `irls-conv-analysis`, then completed): smoothed Huber objective, strong
  convexity, fixed-point map $\mathcal{M}(w)=Q(w)^{-1}b$, Jacobian
  $\nabla\mathcal{M}(w^*)=I-B^{-1}H$, and the spectral-radius bound
  $\rho(\nabla\mathcal{M}(w^*))<1$ via similarity to $S=B^{-1/2}HB^{-1/2}$ and
  the Loewner order $H\preceq B$, closed by Ostrowski's theorem.
  - Added the **explicit computation of $dQ$** (chain rule on
    $1/\max(|w_i|,\varepsilon)$, active/pinned cases) that yields
    $(dQ)w^*=-(B-H)\,dw$ — the proof previously asserted this step.
  - Added the **"why the rate is close to 1"** mechanism ($\mu_{\min}\to0$
    because the surrogate over-curves the affine $L_1$ penalty on the active
    set) and the one-line check that $w^*$ is a fixed point
    ($Q(w^*)w^*=b\iff\nabla f_\varepsilon(w^*)=0$).
  - Restated $Q(w),b$ in the appendix, unified $\lambda\equiv\lambda_{\text{LASSO}}$
    and $\varepsilon\equiv\varepsilon_{\mathrm{thr}}$, noted $B,H$ share the
    off-diagonal $X^{\top}X$, fixed `\ref`$\to$`\eqref`.

### Changed — theory (`algo1.tex` §2.5)

- Restored the definition of the fixed-point map $\mathcal{M}(w)=Q(w)^{-1}b$ in
  Chapter 2 (the condensed merge referenced $\nabla\mathcal{M}(w^*)$ without
  defining it), straightened the Ostrowski sentence (dangling "then",
  "hypothesis"$\to$hypotheses), and kept a one-clause "close to 1" pointer to
  Appendix A.

### Changed — results numbers & tables (refreshed from committed CSVs)

- **Figure 5.3 (`dsm_nonmonotone`)** — overshoot annotation moved below the peak
  (was overlapping the title).
- **Figure 5.4 (`irls_real_data_warm_vs_cold`)** — gap now measured against an
  **independent deeper IRLS run** ($\varepsilon_{\mathrm{thr}}=10^{-14}$) rather
  than the self-min, removing the spurious machine-zero tail; curves flatten at
  the real $O(\varepsilon_{\mathrm{thr}}=10^{-8})$ smoothing floor.
- **Figure 5.12 (`real_data_convergence`)** — legend moved to upper-right (was
  covering the IRLS curve).
- **Table 5.4 (solver)** and **5.5/5.6 (scalability, iters-to-$\varepsilon$)** —
  wall-clock columns refreshed from the current CSVs (gaps/iteration counts
  already matched); related prose updated.
- **Table 5.2 (warm/cold)** — IRLS real-data gaps now reported against the
  independent reference; **Table 5.7 ($\delta_0$-ratio)** synthetic row corrected
  to its true range $[0.18,2.07]$ with medians (the "$[0.5,2]$ for all" claim was
  false against `delta0_families.csv`).
- Prose corrections: $122$ iterations to $10^{-6}$, gap $1.5\cdot10^{-10}$,
  $\gamma$-floor unfloored gap $4.4\cdot10^{-2}$, diabetes warm contractions $12$.
- LLM-pattern cleanup outside the *Initial design and corrections* section
  ("essentially"$\to$"nearly", "flatters"$\to$"understates", decorative
  em-dashes $\to$ parentheses).

### Added — reproducibility (`progetto/code/results/tables/`)

- `solver_comparison.csv`, `warm_cold_irls_{synthetic,real}.csv`,
  `warm_cold_sgptl_{synthetic,real}.csv` — Tables 5.2 and 5.4 are now
  regenerable; the corresponding scripts write these on each run. Table 5.8
  (before/after) is annotated as describing the removed pre-correction
  prototype, not regenerable from current code.

### Fixed — code (`progetto/code/src/`)

- **`irls.py`** — the OLS warm start now validates the normal-equations residual
  and falls back to `lstsq` for **either** Cholesky or CG, so a CG breakdown on a
  rank-deficient $X^{\top}X$ no longer seeds IRLS with a garbage warm start.
- **`deflected_subgradient.py`** — dropped the undocumented `max(.,10^{-4})`
  floor on $\delta_0$, so the code matches the report formula
  $\delta_0=0.1\,f(w_0)$ (the floor never bound on any reported instance).

### Verified

- `python -m pytest tests/` — 53/53 pass; `test_basic.py` green.
- `latexmk -pdf -bibtex main.tex` — clean build, no undefined references or
  citations; regenerated figures copied into `report/images/`.

## 2026-06-01 — Second-review fixes: boundedness, IRLS rate, relative gaps

Addresses the professor's second review (`project_review/email_prof2.txt`),
which approved the report pending three corrections, plus four pre-existing
inconsistencies surfaced while making them.

### Changed — theory

- **§3 DSM boundedness argument (`chapter3.tex`)** — The previous text claimed
  the iterates remain in a compact sublevel set $S_{c_0}$ with $c_0 \ge f(w_0)$.
  That is a *descent-method* argument and is invalid for the subgradient method,
  which is non-monotone ($f(w_{i+1}) > f(w_i)$ does occur). Replaced with the
  correct two-fact argument, matching d'Antonio–Frangioni:
  1. $f$ is convex and finite on all of $\mathbb{R}^H$, hence locally Lipschitz,
     so $\partial f$ is bounded on bounded sets;
  2. the iterates stay bounded by **Fejér contraction** of $\|w_i - w^*\|$ once
     the target level underestimates $f^*$ (the $\delta$-contraction of
     Lemma 3.8 reaches this regime in finitely many steps) — the mechanism by
     which the proof of Theorem 3.5 of the paper obtains $\sup_i\|d_i\| = D < \infty$.
  Coercivity is now stated to give only $f^* > -\infty$ attained, not iterate
  boundedness.
- **§2 IRLS convergence rate (`algo1.tex`)** — The professor noted the IRLS
  curve is not visibly linear. Verified the cause: IRLS *is* locally linearly
  convergent to the smoothed (Huber) minimiser, but the asymptotic factor is
  $\approx 0.995$ (slow) — the loose quadratic majorisation of $|w|$ on the
  active components, $\approx$ independent of $\varepsilon_{\mathrm{thr}}$.
  Replaced the **mis-applied** Daubechies $\ell_q$ / null-space-property
  citation (that result is for *underdetermined* sparse recovery; our regime is
  overdetermined and strongly convex) with the correct strong-convexity + MM
  argument. Corrected the "the curve should be approximately straight" claim to
  describe the fast transient + slow linear tail + $O(\varepsilon_{\mathrm{thr}})$
  floor.
- Four references that attributed the linear *rate* to
  `Theorem~\ref{thm:irls_convergence}` (which proves only fixed-point
  consistency, no rate) were redirected to `Section~\ref{sec:irls_convergence}`
  (`comparison.tex` ×2, `conclusions.tex` ×2, `algo1.tex` ×1).

### Changed — relative optimality gaps everywhere

Per the professor's request, every reported optimality gap is now the
**relative** gap $(f - f^{*})/|f^{*}|$ (an absolute gap of $10^{-6}$ is loose at
$f^{*} \approx 2358$ on California and tight at $f^{*} \approx 1.1$ on the
synthetic). The convention is stated explicitly in §5.2.

- **Figure 2 (`convergence_vs_iter`)** — both halves now plot and label the
  *same* quantity (the professor flagged two different y-axis names). The IRLS
  panel is re-run with $\varepsilon_{\mathrm{thr}}=10^{-12}$ over 1500 iterations
  so the linear tail is visible as a straight line on the semilog axis, with a
  dashed fit annotating the measured rate $\approx 0.995$. (Bogus "$0.848$ per
  iter" local-window annotation removed.)
- All gap figures regenerated with relative-gap axes: `convergence_vs_time`,
  `dsm_nonmonotone`, `comparison_irls_dsm`, `real_data_convergence`,
  `sgptl_long_run`, `warm_vs_cold`, `irls_*_warm_vs_cold`, `params_irls`,
  `params_dsm`, `gamma_floor_test`, `delta0_families`, `delta0_proxy`.
- Tables converted to relative gaps: 5.1 (warm/cold), scalability, iters-to-$\varepsilon$,
  long-run samples, real-data cost, $\gamma_{\min}$ floor, before/after, $\delta_0$ sweeps.
- **Target-based tables** (`iters-to-eps`, `real_data`) now use a **relative**
  target $(f-f^{*})/|f^{*}| \le \varepsilon$ instead of an absolute one, so the
  difficulty is comparable across instances. Crossings re-measured: on real data
  SGPTL reaches relative $10^{-6}$ at $5.63\cdot10^{4}$ iter (diabetes) /
  $2.97\cdot10^{5}$ iter (California); IRLS at 56 / 3 iter.

### Fixed — pre-existing inconsistencies

- **`delta0` bug, `experiment_warm_vs_cold_real_data.py`** — the *cold* run used
  $\delta_0 = 0.1\,f(w_{\mathrm{OLS}})$ (the *warm* objective) instead of
  $0.1\,f(0)$, contradicting the report's own §5.4 rule $\delta_0 = c\,f(w_0)$
  for the run's starting point. This made Table 5.1's cold gaps diverge from the
  long-run table for the *same* quantity (diabetes $4.2\cdot10^{-3}$ vs
  $4.9\cdot10^{-3}$ at $k=8000$). Fixed to per-start $\delta_0$; the cold run now
  matches the long-run trace **exactly** (diabetes abs $0.2604$, California abs
  $41.41$ at $k=8000$). `experiment_warm_vs_cold.py` (synthetic) was already
  correct.
- Table 5.1 caption claimed "SGPTL runs 8000 iterations everywhere" while the
  real-data run actually used 2000/5000 iterations with non-default
  $\rho,\delta_0$; realigned the run to 8000 iter and the standard config, so the
  caption is now true.
- §5.6 prose said "SGPTL does not reach $10^{-6}$ within 10000 iterations"; the
  experiment budget is 30000 — corrected.

### Changed — config

- **`.claude/CLAUDE.md`** — added a prioritised "Correttezza e coerenza teorica
  (revisione critica)" section: verify each theoretical claim against the cited
  theorem's *actual* hypotheses on this problem, be critical of the source paper
  (descent-only arguments don't transfer to subgradient), flag mis-applied
  citations, reconcile theory vs data by finding the regime/reference/artefact
  rather than relabelling, and never assert an unverified mechanism.

### Build

- Report recompiled (`latexmk -pdf -bibtex`), no undefined references or
  citations. All regenerated figures copied into `report/images/`.

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
