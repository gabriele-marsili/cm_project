# ROADMAP — CM Project 25 ML (ELM LASSO)

**Goal**: Complete the project deliverable: validated code + tests + experiments + report Chapters 5–6, all aligned with the theory in Chapters 1–4 (already approved).

**Constraints (from CLAUDE.md)**:
- Every formula/derivation must be supported by course theory (`Lessons Numerical Linear Algebra`, `Lessons Optimization`, references in report bibliography).
- No fabricated test results, no hallucinated numbers in the report.
- Use Python (existing codebase ~1755 LOC).

## Phases

| # | Phase | Goal | Status |
|---|---|---|---|
| P1 | Code audit & fix | Bring code into conformity with report definitions and algorithms (objective scaling, IRLS Q-update, DSM warm-start, DSM β_i). | TODO |
| P2 | Test suite | Replace `test_basic.py` with a proper pytest suite: KKT residual, IRLS monotonicity, gradient/subgradient finite-diff, cross-check vs sklearn/CVXPY, edge cases. | TODO |
| P3 | Experiments | Re-run all 4 experiment scripts after fixes; produce final figures and CSV tables for Chapter 5. Add CVXPY cross-check. | TODO |
| P4 | Report Chapter 5 | Write "Experimental Results" sourcing every number from `results/`. | TODO |
| P5 | Report Chapter 6 | Write "Conclusions" — synthesis of what theory predicted vs what experiments confirmed. | TODO |

## Phase dependencies

```
P1 → P2 → P3 → P4 → P5
```

P2 depends on P1 (tests must run on fixed code).
P3 depends on P1+P2 (experiments need correct code; tests are the gate).
P4 depends on P3 (numbers come from experiments).
P5 depends on P4.

## Out of scope

- Switching language (Python is fixed).
- Refactoring code architecture beyond what audit findings require.
- ML benchmarking on real datasets — the project is about optimization quality, not learning quality (cf. comando.pdf §4.6).
