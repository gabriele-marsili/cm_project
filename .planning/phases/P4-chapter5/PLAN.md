# P4 — Chapter 5: Experimental Results

**Goal**: Write Chapter 5 of the report sourcing every number from the experiments in `progetto/code/results/`.

## Constraints applied

- Every number is from a real experiment run; no fabricated values.
- Every cross-reference (`\ref{...}`) was checked against the actual labels defined in chapters 1–4 and updated when needed.
- The chapter follows the comando.pdf §4.6 structure: setup, convergence, parameter sensitivity, solution quality, scalability, head-to-head.
- Reference solver: sklearn (already used internally as `f_star`); no external CVXPY since not installed.

## Sections

1. Experimental setup (implementation, reference solver, synthetic data, hardware).
2. Convergence behaviour (Figs convergence_vs_iter, convergence_vs_time, dsm_nonmonotone).
3. Parameter sensitivity (Figs params_irls, params_dsm).
4. Solution quality and sparsity.
5. Scalability (Tab + Fig scalability).
6. Iterations to a target accuracy (Tab + Fig comparison).

## Deliverable

`progetto/report/5_results/results.tex` — full chapter, ~7 pages.
Figures copied to `progetto/report/images/`.
Compiles clean with `latexmk -pdf main.tex` (43 pages total).
