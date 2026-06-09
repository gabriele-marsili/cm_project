# ELM + LASSO — code

Extreme Learning Machine with an L1-regularised output layer. The hidden layer
is random and frozen, so training reduces to a LASSO on the hidden features:

    min_w  1/2 ||X w - y||^2 + lam ||w||_1

We solve it two ways, both written from scratch:

- **A1 — IRLS** (`src/irls.py`): reweight the L1 term into a quadratic and solve
  one SPD system per outer step.
- **A2 — DSM** (`src/deflected_subgradient.py`): deflected subgradient with a
  Polyak target level (SGPTL) on the non-smooth objective directly.

`scikit-learn` / `cvxpy` show up only as independent references for f* and as
correctness oracles, never as the solver under test.

## Layout

```
src/                 core implementation (the graded part)
tests/               pytest unit tests + correctness checks vs sklearn
test_basic.py        standalone smoke test, no pytest needed
experiments/         scripts that produce the report figures/tables
experiments/old/     earlier runs, kept for traceability
results/figures/     generated PDFs
results/tables/      generated CSVs
```

## src modules

| file | what's in it |
|------|--------------|
| `lasso_utils.py`         | objective f, smooth gradient, a subgradient, KKT residual, support metrics |
| `linear_solvers.py`      | SPD solvers for the IRLS inner step: dense Cholesky and Jacobi-preconditioned CG |
| `irls.py`                | algorithm A1 |
| `deflected_subgradient.py` | algorithm A2 (SGPTL) |
| `elm.py`                 | the `ELM` class: random hidden layer + LASSO fit on top |
| `data_generation.py`     | synthetic LASSO / ELM-LASSO generators, real-dataset loader (diabetes, california) |

Both solvers take the same `(X, y, lam)` and return a dict with the final `w`
plus per-iteration histories (`f_vals`, `gaps`, `times`, ...) used by the plots.
Pass `f_star=...` to log the optimality gap.

## Running

Everything runs from this directory (`progetto/code/`).

Tests:

```bash
python -m pytest tests/ -q     # full suite
python test_basic.py           # quick smoke test, prints PASS/FAIL
```

Experiments — each script is standalone and writes into `results/`:

```bash
python experiments/experiment_convergence.py     # IRLS vs SGPTL on one instance
python experiments/experiment_real_data.py       # both solvers on diabetes + california (ELM)
python experiments/experiment_scalability.py     # FLOP count vs measured wall-clock
```

The script names map to what they measure:

- `experiment_comparison`, `experiment_convergence` — IRLS vs SGPTL, gap vs iterations and vs time
- `experiment_real_data`, `experiment_*_warm_vs_cold*` — the two solvers on real ELM-LASSO, warm (OLS) vs cold (0) start
- `experiment_params`, `experiment_delta0_families`, `experiment_gamma_floor` — sensitivity to algorithm parameters (eps_thr, lambda, Cholesky/CG, delta_0, rho, gamma_min)
- `experiment_sgptl_long_run`, `rerun_*_to_crossing` — long SGPTL runs and time-to-reach a target gap
- files starting with `_` are helpers (shared plot style, re-plot from cache, diagnostics), not run on their own

## Dependencies

`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pytest`. Three experiments
(`experiment_real_data`, `experiment_delta0_families`,
`experiment_warm_vs_cold_real_data`) also use `cvxpy` to cross-check f* with a
second solver. Developed on Python 3.12.

Reproducibility: every generator and split takes a `random_state`, so the
figures and tables in `results/` regenerate from the seeds in the scripts.
