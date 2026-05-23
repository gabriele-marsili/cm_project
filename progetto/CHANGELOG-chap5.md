# Chapter 5 — Session Changelog

Date: 2026-05-23.
Scope: Chapter 5 "Experimental Results" of the ELM+LASSO report.

---

## §5.5.2 Scalability — repetitions with Ch. 4 removed
- The middle paragraph no longer restates the per-iteration costs
  ($O(MH^{2}+H^{3})$ for IRLS, $O(MH)$ for SGPTL); now references
  `tab:comp_cost` in Chapter 4 directly.
- The closing paragraph no longer re-derives the $O(L^{2}/\varepsilon^{2})$
  bound argument; cross-references §5.6.2 instead.
- Net: 3 paragraphs, no duplicate content; same data, no narrative loss.

## §5.6 Validation on real datasets — full rework
### Comparison criterion (fair, like-for-like)
- Switched from fixed-iteration to fixed-accuracy: each method now
  runs until $f - f^{*} \le 10^{-6}$ and the cost is reported in
  iterations and wall time.
- IRLS settings: `eps_thr=1e-12`, `k_max=2000` (was `eps_thr=1e-8`,
  `k_max=100`). This pushes IRLS to its true convergence floor and
  removes the "IRLS stopped early" artefact.
- Clarabel tolerance: `tol_gap_abs/rel = tol_feas = 1e-9` (was
  default ~1e-8). Matches the IRLS-tight reference.
- $f^{*}$ reference: IRLS with `eps_thr=1e-14`, `k_max=3000`,
  cross-validated against Clarabel-tight. The two agree to nine
  significant digits on both instances (`52.9487633708` on diabetes,
  `2358.0194491500` on california).

### Reference oracle — sklearn → Clarabel
- sklearn CD never satisfies its own duality-gap tolerance on these
  ELM-transformed problems even at `max_iter=100000`, `tol=1e-10`:
  the residual gap to $f^{*}$ is $+1.4\cdot10^{-3}$ on diabetes
  and $+2.82$ on california. Removed sklearn from Table 5.8.
- Table now uses CVXPY-Clarabel as the third-party oracle (rows:
  Clarabel / IRLS / SGPTL / Ridge). sklearn is only mentioned in
  the prose as the methodological justification for not using it.

### Table 5.8 — new layout
- Vertical layout (diabetes block above california block) instead of
  side-by-side.
- New columns: method, iter to gap $\le 10^{-6}$, wall time, sparsity
  at $10^{-3}$, test MSE.

#### Diabetes ($M=354$, $f^{*}=52.948763$)
| method        | iter           | wall time | sp.   | MSE   |
|---------------|----------------|-----------|-------|-------|
| Clarabel      | 10             | 92 ms     | 19 %  | 0.898 |
| IRLS          | 178            | 14.5 ms   | 19 %  | 0.898 |
| SGPTL cold    | 3.25·10⁵       | 6.6 s     | 19 %  | 0.898 |
| Ridge         | 1 (closed)     | < 1 ms    | —     | 0.942 |

#### California ($M=16512$, $f^{*}=2358.019449$)
| method        | iter           | wall time | sp.  | MSE   |
|---------------|----------------|-----------|------|-------|
| Clarabel      | 8              | 6.3 s     | 4 %  | 0.307 |
| IRLS          | 34             | 18.7 ms   | 4 %  | 0.307 |
| SGPTL cold    | 9.33·10⁵       | 1115 s    | 4 %  | 0.307 |
| Ridge         | 1 (closed)     | < 5 ms    | —    | 0.307 |

### Figure 5.12 — redrawn
- Old: SGPTL stopped at $i_{\max}=8000$, warm-start dashed overlay,
  IRLS at 100 iter.
- New: IRLS to convergence ($k_{\max}=2000$,
  $\varepsilon_\text{thr}=10^{-12}$); SGPTL cold long-run trace
  loaded from `long_run_cache/{name}.npz`
  ($i_{\max}=10^{7}$/$10^{6}$); theoretical envelope
  $g_{0}/\sqrt{i}$ overlaid; sklearn marker dotted for context;
  no warm-start overlay.
- The figure now shows SGPTL converging at the predicted rate
  (instead of stalling visually at 8000 iter).
- The orphan `images/real_data_convergence.pdf` (stale copy of 22 May
  that the report was actually pointing at) has been synced to the
  freshly regenerated artefact in `code/results/figures/`.

### §5.6 prose — rewritten "Reading the table"
- Removed the "overfitting at this M/H ratio" narrative anchored on
  the now-removed SGPTL-at-8000-iter MSE 0.832. The three converging
  methods (Clarabel, IRLS, SGPTL-long-run) now coincide on f,
  sparsity, MSE; the test-MSE comparison is just LASSO vs Ridge,
  which is dataset-specific.
- Removed the duplicate sentence about "OLS warm start coincides
  with f* on California" (was the third occurrence; the concept is
  already in §5.1 limitation (iii) and §5.3 warm-vs-cold).

## §5.7 Initial design and corrections — restructured
- New explicit four-part structure for each correction:
  - **Before** (the wrong choice)
  - **Why wrong** (mechanistic explanation)
  - **After** (the correct choice)
  - **Why right** (why the new choice satisfies the theorem)
- Two corrections documented this way: $\delta_{0}=c\,f^{*}$ →
  $c\,f(\mathbf{w}_{0})$, and $R\approx 894$ + iteration-count
  fallback → $R=1$ + travel-distance trigger.
- New `tab:before-after-numbers` table that compares the two
  implementations side by side on the same instance: $R$ used,
  fallback yes/no, contractions from $r_{k}>R$, contractions from
  fallback, final record gap.
- Reference to `results_old_submission/` removed (will not be
  shipped in the submission).
- LLM-style sentence "the gap values reported are now the honest
  ones rather than the artefactual ones" removed and replaced with
  a direct technical statement.

## Code
- `experiment_real_data.py`:
  - `IRLS_KMAX` 100 → 2000; IRLS `eps_thr` 1e-8 → 1e-12.
  - Reference `f^{*}` uses IRLS with `eps_thr=1e-14`, `k_max=3000`.
  - Clarabel call adds `tol_gap_abs=tol_gap_rel=tol_feas=1e-9`.
  - Returns `f_cvxpy`, `gap_cvxpy`, `spar_cvxpy_*`, `mse_cvxpy` for
    the new Clarabel table row.
  - Loads the SGPTL long-run cache when present and uses it for the
    figure trace; falls back to the in-memory $i_{\max}=8000$ run
    when the cache is absent.
- `experiment_sgptl_long_run.py`:
  - Now keeps both train and test splits.
  - Writes a `.npz` cache to `results/tables/long_run_cache/{name}.npz`
    containing the sub-sampled gap trace (~2000 points geometrically
    spaced), the final $w$, final $f$, final gap, test MSE, sparsity
    and elapsed time.
- Confirmed long-run timings: diabetes $i_{\max}=10^{7}$ in $205$ s
  (3.4 min), california $i_{\max}=10^{6}$ in $1289$ s (21.5 min).

## CLAUDE.md (project-local) — strengthened anti-LLM rules
- Forbidden patterns: "X is Y, just not Z" formulations; promotional /
  anthropomorphic phrases (*"X is the heart of Y"*, *"the picture is
  more nuanced"*, etc.); empty emphatic adjectives (*essentially*,
  *clearly*, *naturally*); forced rule-of-three; decorative em-dashes;
  figurative verbs (*amortise*, *wash out*, *kick in*); narrative
  fillers (*Importantly*, *Crucially*).
- Anti-repetition rule: grep before writing; one mechanism in one
  place; when a repetition is flagged, delete one of the two
  occurrences (do not paraphrase).
- These rules now apply to the rest of the report as well.

---

## What did NOT change in Chapter 5
- §5.1 (Setup), §5.2 (Convergence), §5.3 (Warm vs cold), §5.4
  (Hyperparameter sensitivity), §5.5.1 (Solution quality), §5.5.3
  (Iterations to target) — all left as in the previous commit, except
  for one cross-reference fix in §5.5.2.
- Algorithms (`irls.py`, `deflected_subgradient.py`, `elm.py`) — not
  modified.
- Reference $f^{*}$ values to six significant digits — unchanged
  ($52.949$, $2358.019$); the seventh and eighth digits tightened
  slightly because of the tighter `eps_thr` reference.
