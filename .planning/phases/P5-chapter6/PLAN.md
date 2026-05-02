# P5 — Chapter 6: Conclusions

**Goal**: Write a 1-page synthesis: theory predictions vs. empirical findings, recommendation, limitations.

## Sections

1. **What the experiments confirmed** — IRLS linear, monotone, sparse; SGPTL sublinear, oscillating-with-monotone-record, approximately sparse.
2. **Where bounds proved conservative** — IRLS plateau due to FP not safety threshold; SGPTL ρ has no observable effect on final gap once mechanism works; OLS warm start masks ρ behaviour.
3. **Recommendation** — IRLS for the ELM regime M≫H; SGPTL competitive only when H²≫M.
4. **Limitations and possible extensions** — synthetic data only by design; no Nesterov smoothing because spec fixes A2 = SGPTL; conditioning kept bounded by column normalisation.

## Deliverable

`progetto/report/6_conclusions/conclusions.tex` — full chapter, ~1.5 pages.
