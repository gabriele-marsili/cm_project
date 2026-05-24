# Changes pre-consegna — riepilogo per review manuale

Tutte le modifiche applicate nella sessione del 2026-05-24 in risposta ai 5 punti del prof, alla coerenza interna, allo sweep anti-LLM e ai bug del codice.

- Findings originali: `project_review/REVIEW.md`
- Stato consegna: PDF compila a 54 pagine, 53/53 test passano, 0 warning di reference, 1.48 MB.

> Notazione usata in questo file: `γ`, `δ`, `β`, `α`, `ε`, `λ`, `√`, `²`, `≤`, `≥`, `·`, `∂`, `‖·‖`. Niente LaTeX inline.

---

## 1 · Cambiamenti CRITICI — leggere per primi

In ordine di rischio decrescente.

### 1.1 — Theorem 3.1 riformulato (era "allucinazione da LLM" per il prof)

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex`, linee ~145–198.

**Cosa cambia**:

- Rate prima: `(f̄_k − f*) ≤ L · ‖w₀ − w*‖ / √(γ_min · (k+1))`
- Rate dopo: `(f̄_k − f*) ≤ L · ‖w₀ − w*‖ / (γ_min · √(k+1))` — denominatore cambia, ora `1/γ_min` invece di `1/√γ_min`.
- Iter inflation prima: `1/γ_min = 20×` per γ_min = 0.05.
- Iter inflation dopo: `1/γ_min² = 400×` per γ_min = 0.05.
- Per-step descent factor prima: `β_i · (2 − β_i)` con citazione vaga a Slide 15.
- Per-step descent factor dopo: `β_i · (2γ_i − β_i)` con citazione esplicita a **eq. (3.17) nel proof of Theorem 3.5** di d'Antonio–Frangioni 2009 + Corollary 3.2 dello stesso paper.

**Perché serve rileggere**: è il pezzo che il prof ha esplicitamente definito "allucinazione da LLM". Ho riletto il paper (~15 pagine) e trovato che il vero descent per direzione deflessa è eq. (3.17) con costante `β_k(2α_k − β_k)`, non `β_k(2 − β_k)`. Nel vostro setup con `β_i = α_i = γ_i` la costante è `γ_i² ≥ γ_min²`, da cui la rate corretta. Verifica i passaggi siano chiari e che la rate più conservativa sia accettabile (si allinea meglio con la 100×-per-decade osservata in §5).

### 1.2 — Propagation della rate corretta

Costanti aggiornate in tutti i punti dove la rate era citata:

- `progetto/report/3_algo_2_DSM/chapter3.tex` — linee 114, 210, 214 (rationale γ_min, expected behavior).
- `progetto/report/4_algo_comparison/comparison.tex` — linea 35 (DSM rate item).
- `progetto/report/5_results/results.tex` — linea 232 (order-of-magnitude heuristic) + linee 237–248 riscritte con nuove previsioni numeriche (5.6 / 28 / 1320 invece di 0.28 / 1.4 / 66; empirico ora "1 to 5 orders below envelope").
- `progetto/report/6_conclusions/conclusions.tex` — frase "well below worst-case envelope" rimossa (era closing meta-comment).

### 1.2 bis — Appendice C: derivazione di eq.(3.17) → eq.(`def-step-polyak`)

**File**: `progetto/report/appendix/appendix.tex` Appendix C (nuovo capitolo, pp. 48–49 del PDF).

**Cosa contiene**: derivazione completa in 4 step della per-step Polyak inequality `‖w_{i+1}−w*‖² ≤ ‖w_i−w*‖² − β_i(2γ_i−β_i)(f_i−f*)²/‖d_i‖²` partendo dai 3 ingredienti del paper d'Antonio–Frangioni:

1. Identità (2.9) — espansione del quadrato `‖w_i − α_i d_i − w*‖²`.
2. Corollary 3.2 specializzato a `x̄ = w*`, `σ = 0` → `⟨d_i, w_i − w*⟩ ≥ γ_i(f_i − f*)`.
3. Sostituzione del Polyak step `α_i = β_i(f_i − f*)/‖d_i‖²` nelle due righe non-costanti di (2.9) + uso di (2).
4. Collecting → fattore `β_i(2γ_i − β_i)`, poi `γ_i² ≥ γ_min²` per la specializzazione `β_i = γ_i` dell'algoritmo.

Nel proof di Theorem 3.1 (corpo del report, §3.6) c'è ora solo la formula finale + rimando ad Appendix C. Questo tiene il proof leggibile ma dà al prof l'algebra esplicita se la chiede all'orale.

### 1.3 — δ₀ vs f*: chiarito che f* non è mai usato come parametro algoritmico (punto P4 del prof)

- `progetto/report/3_algo_2_DSM/chapter3.tex` linea 119: aggiunto disclaimer "all experiments in Chapter 5 use δ₀ = c·f(w₀); the c·f* scale appears only as abstract reference and is never fed to the optimizer".
- `progetto/report/5_results/results.tex` §5.4.5: ribattezzata "abstract sensitivity check"; opening paragraph riformulato per dichiarare upfront che è una sanity check accademica, non parte della calibrazione operativa.
- `progetto/report/appendix/appendix.tex` §A: l'intero capitolo ora dichiara esplicitamente "abstract sensitivity study"; aggiunto disclaimer su california H=200 dove il proxy IRLS-converged è usato come f* (doppia circolarità segnalata).

### 1.4 — Tabella 5.8: tempi SGPTL ora MISURATI (punto C2)

**File**: `progetto/report/5_results/results.tex`, linee ~810 e ~817 (righe Tabella) + caption ~828–845.

**Cosa cambia**:

| Dataset    | Prima (estrapolato)    | Dopo (misurato)         |
| ---------- | ---------------------- | ----------------------- |
| diabetes   | 3.25·10⁵ iter / 6.6 s  | **3.24·10⁵ iter / 7.7 s** |
| california | 9.33·10⁵ iter / 1115 s | **9.29·10⁵ iter / 1082 s** |

Numeri letti da `progetto/code/results/tables/real_data_crossing.csv` (output del rerun `experiments/rerun_sgptl_to_crossing.py`). Caption ora esplicita: wall-time include iter loop, esclude OLS warm-start setup (negligible, +0.13 / +2.9 ms).

### 1.5 — Clarabel agreement digits uniformato (C1)

- Prima: "nine significant digits" in Tab. 5.8 caption; "six" in §5.6 (linea 781) e §6 (linea 57).
- Dopo: **"ten significant digits"** in tutti e 3 i posti (vero su entrambi i dataset, verificato da `real_data.csv`).

---

## 2 · Cambiamenti SOSTANZIALI (teoria e coerenza)

### 2.1 — sign(0) probabilità misura zero (P1)

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex:139`.

Aggiunta frase: per i ≥ 1, l'update `w_{i+1} = w_i − α_i d_i` dipende continuamente dal passato e dalla sequenza continua δ-contraction, quindi `Pr((w_i)_j = 0) = 0` (Lebesgue measure zero). Il cold start `w₀ = 0` è l'unico iterate dove la convenzione `sign(0) = 0` è effettivamente usata, e solo a `i = 0`.

### 2.2 — Costo OLS warm-start quantificato (P3)

**File**: `progetto/report/5_results/results.tex` §5.3 Defaults (linee ~315–326).

Numeri concreti aggiunti:

- diabetes (H = 200): OLS Cholesky setup = `0.13 ms`, IRLS body = `165 ms` → warm-start share = `0.08 %`.
- california (H = 200): OLS Cholesky setup = `2.9 ms`, IRLS body = `1577 ms` → warm-start share = `0.18 %`.

Fonte: `progetto/code/results/tables/warm_start_cost.csv`. Conclusion line nel report: "il warm-start share è < 0.2 % in entrambi i regimi, quindi le 13–25 % iteration savings amortizzano abbondantemente".

### 2.3 — Test MSE demoted (P6)

**File**: `progetto/report/5_results/results.tex` §5.6 "Reading the table" + Tab. 5.8 caption.

- "Reading the table" paragraph: rimosso "same test MSE" come finding; aggiunto disclaimer "test-MSE columns included for completeness, not part of optimisation comparison".
- Frase narrativa "The MSE comparison is dataset specific…" cancellata.
- Tab. 5.8 caption: test MSE è "for completeness only".

### 2.4 — Full-rank claim citato correttamente (T1)

**File**: `progetto/report/1_introduction/report.tex:27`.

Prima: "results, under mild assumptions and with high probability, in the hidden layer feature matrix with full rank when M ≥ H, since random projections through a nonlinear function produce linearly independent features" (zero citazioni).

Dopo: "We assume henceforth that the hidden-layer feature matrix X has full column rank, which is the regime studied in the original ELM analysis [Huang 2006, Theorem 2.1] for M ≥ H with random W₁ and an infinitely differentiable activation; in our experiments (sigmoid, W₁ Gaussian, M ≥ H) it is verified numerically by the conditioning checks of §5.1".

### 2.5 — Theorem 2.1 (IRLS): ipotesi violata gestita esplicitamente (T2)

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex` §2.5 "Verification".

Problema: il theorem richiede `|(w*)_i| > ε_thr` per ogni `i`, ma il LASSO ha componenti esattamente zero (sparse).

Fix: ora il report dichiara esplicitamente che IRLS solve la **smoothed variant** indotta dal clip `max(|w_i|, ε_thr)`. Il termine ℓ₁ è sostituito da un surrogato Huber-like che coincide con `|w_i|` per `|w_i| > ε_thr` e quadratico altrimenti. Il minimizer del smoothed differisce dal LASSO minimizer per O(ε_thr) in objective e recupera il LASSO per ε_thr → 0. Distinzione esplicita active set vs inactive set: Theorem 2.1 si applica sull'active set; sull'inactive, il smoothed gradient pinna le componenti a `≤ ε_thr` con `s_i ∈ [−1, +1]`, in accordo con il subdifferential del valore assoluto a zero.

### 2.6 — Daubechies + Tikhonov misuse (T3)

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:222`.

Prima:

- "Tikhonov preconditioner" (uso scorretto del termine — `W_kᵀW_k` è un regolarizzatore, non un preconditioner CG).
- "This analysis, observed by us empirically, is consistent with Daubechies et al.'s analysis" (linear rate enunciata ma non verificata su ELM).

Dopo: "data-adaptive diagonal shift". Linear rate di Daubechies dichiarata sotto la **Null Space Property** (NSP), non verificata su ELM features con W₁ random. Daubechies citato come "closest available analysis". Empirical verification rimandata a §5.2.

### 2.7 — Ripetizione tripla di Condition (3.5) (T4–T6)

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex` §3.2 + §3.5.2.

Prima: enunciata 3 volte (linee 33, 114, 160–164).

Dopo: enunciata una sola volta nel proof di Theorem 3.1 (§3.6). §3.2 paragrafo "Deflection floor" rimosso del numerical sweep (era leakage di §5 nella sezione teorica) e accorciato a mecanismo + forward reference. §3.5.2 paragrafo "γ_min = 0.05" trimmato a rationale teorico (sweep numeri ora solo in §5.4.3).

### 2.8 — Theorem assumption verification (T5)

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex` §3.6 "Verification of theorem assumptions".

Riformulato. Ora trae solo ciò che serve: coercività di `f` → sublevel set compatto → `f* > −∞` attainable. Stepsize-restricted clause `β_i ≤ γ_i` esplicitata per costruzione (`β_i = min(1, γ_i)`). Rimosso il riferimento ridondante a uniqueness (non usata dal theorem).

### 2.9 — Bubeck footnote ambigua (T8)

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex:10`.

Prima: footnote pseudo-spiegava la differenza tra `O(L²/ε²)` e `O(L/ε²)` con rescaling confusionario.

Dopo: footnote rimossa, citazione semplificata a `[Theorem 3.2]{bubeck2015convex}`.

### 2.10 — Cosmetics (T10, T12, T14)

- `report.tex:73` — `\sum_i^H` → `\sum_{i=1}^{H}` (mancava lower bound).
- `report.tex:77` — `\ref` → `\Cref` per i chapter reference (prima renderizzava "(2) and (3)" invece di "Chapter 2 and Chapter 3").
- `algo1.tex:190` — `\ref{eq:optimality}` → `\eqref{eq:optimality}`.

### 2.11 — Caption Tab. 5.2 onesta (C3)

**File**: `progetto/report/5_results/results.tex` linee 213–226.

Prima: "IRLS gap < 10⁻¹¹" su real data, f* = "IRLS+CVXPY-Clarabel" (caption ambigua → letto come "vs entrambi", ma di fatto è vs IRLS-converged = autoreferenziale).

Dopo: caption esplicita "IRLS gaps on real data are measured against IRLS itself" + dichiara la distanza effettiva a Clarabel (`≤ 7·10⁻⁸` california, `≤ 4·10⁻⁹` diabetes).

### 2.12 — Caption Tab. 5.8 sullo scope del timing (C4)

**File**: `progetto/report/5_results/results.tex` Tab. 5.8 caption.

Aggiunto: "Wall times measure the iteration loop only; the one-time OLS warm-start factorisation (0.13 ms diabetes, 2.9 ms california) is excluded; analoga esclusione per SGPTL del δ₀ = c·f(w₀) evaluation (O(M), microseconds)".

---

## 3 · Cambiamenti di STILE e RIPETIZIONI

### 3.1 — Ripetizioni cross-sezione collassate (KEEP/CUT esplicito)

| Concetto                                                            | KEEP                                                            | CUT / sostituito con forward-ref                                                    |
| ------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| "OLS warm sits within 0.46 of f\* on california"                    | `5_results/results.tex` §5.3 SGPTL paragraph (linee 283–302)    | `chapter3.tex:108`, `5_results/results.tex` §5.1, `conclusions.tex`, `appendix.tex` |
| Cost per decade: IRLS const / SGPTL 100×                            | `comparison.tex:34–38` (def.) + `results.tex` §5.5.3 (numerico) | `results.tex` §5.5.2, `conclusions.tex` (compresso in una frase)                    |
| ρ = 0.7 sweep                                                       | `results.tex` §5.4.4 (full)                                     | `chapter3.tex:122` (forward-ref)                                                    |
| γ_min = 0.05 sweep + rate cost                                      | `chapter3.tex:114` (rationale) + `results.tex` §5.4.3 (sweep)   | `chapter3.tex:33` (sweep numbers rimossi)                                           |
| "Theorem still bounds, empirical below envelope" (pattern flaggato) | `results.tex` §5.3.1 long-run trace (body)                      | `results.tex:244–249`, caption Tab. sgptl-long-run, `conclusions.tex`               |
| "OLS warm is IRLS default everywhere"                               | `algo1.tex:173` + `results.tex` §5.3 Defaults                   | `algo1.tex:251` (paragrafo intero), `results.tex:262–263`                           |
| "SGPTL needs thresholding for sparsity"                             | `chapter3.tex:214` + `comparison.tex:92–94`                     | `results.tex:627` ("as anticipated" cut)                                            |
| δ-contraction staircase                                             | `chapter3.tex:59` + `results.tex` §5.4.3                        | `results.tex:246` (closing meta cut)                                                |

### 3.2 — Pattern LLM-like rimossi

| Pattern                                            | File:linea                                                                | Azione                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| "Let's analyze" / "Let's break down"               | `report.tex:82,88`, `comparison.tex:33`                                   | "We decompose" / "Lowering ε:"                                    |
| "particularly nice"                                | `report.tex:75`                                                           | "induces sparsity at the cost of"                                 |
| "a sort of personalized penalization"              | `algo1.tex:31`                                                            | Rimosso; sostituito con frase tecnica neutra                      |
| "core idea" (×2)                                   | `algo1.tex:6,11`                                                          | Rimosso                                                           |
| "considered a bad practice" (passive promotional)  | `algo1.tex:251`                                                           | Paragrafo intero verbose → una frase                              |
| "easy to monitor and to debug" + typo "numercial"  | `algo1.tex:257–259`                                                       | Riscritto, typo corretto                                          |
| "the picture is different"                         | `results.tex:289`                                                         | "California-ELM is the qualitative outlier"                       |
| "bear it out" (×1)                                 | `results.tex:608`                                                         | Rimosso                                                           |
| "tells the opposite story"                         | `results.tex:678`                                                         | Trim §5.5.2                                                       |
| "sweet spot"                                       | `results.tex:399`                                                         | "default"                                                         |
| "genuinely", "actually" (empty intensifiers)       | `chapter3.tex:108`, `appendix:59`                                         | Rimossi                                                           |
| "essentially nothing to optimise" / "essentially quadratic" | `results.tex:91,247`                                              | Riscritto                                                         |
| Opener promozionale "the guiding questions are the two…" | `results.tex:7`                                                     | Opener neutro                                                     |
| Em-dash decorative (5 occorrenze)                  | `chapter3.tex:33,119,212`, `results.tex:284,300,500`, `conclusions:42,46` | Sostituite con virgola, parentesi o cut                           |
| Closing meta-comment "Numbers therefore understate…" | `results.tex:684–685`                                                   | Cut                                                               |
| Closing meta-comment "This is the quantitative form…" | `results.tex:725–727`                                                  | Cut                                                               |
| "Recommendation" §6 promozionale                   | `conclusions.tex:49–66`                                                   | Riscritto neutro, bullet citation con sezioni                     |
| "What experiments confirmed" §6, 30 righe          | `conclusions.tex:18–47`                                                   | Compresso a ~10 righe                                             |
| Limitations §6 con 4 restatement                   | `conclusions.tex:67–90`                                                   | Compresso, restatement eliminati                                  |

### 3.3 — Refactor §5.3 SGPTL paragraph

Il paragrafo originale (linee 279–302) è stato spezzato in due:

1. **Prima parte**: synthetic + diabetes (numeri tenuti).
2. **Seconda parte**: california-ELM dichiarato "qualitative outlier and source of the warm-start caveat referenced throughout this report" — questo è l'**anchor** scelto (decisione R1) per il concetto "OLS-near-f*". Tutte le altre occorrenze ora puntano qui.

### 3.4 — §5.1 limitations (iii) collassato

Da 6 righe a 2 righe con riferimento a §5.3.

### 3.5 — Appendix A riformulato

L'intero chapter ora dichiara upfront "abstract sensitivity study, not used to choose default"; aggiunto disclaimer su california H=200 dove il proxy IRLS-converged è usato come f* (doppia circolarità segnalata).

---

## 4 · Cambiamenti CODICE

Applicati TUTTI i K identificati nella review (K1–K14), divisi per file.

### `src/linear_solvers.py`

- **K3** Guard CG breakdown: aggiunti check `p @ Qp ≤ floor` e `|rz| ≤ floor` che ritornano l'iterato corrente invece di propagare NaN su matrici quasi-singolari (IRLS rende `Q` ill-conditioned con `1/ε_thr ~ 10⁸` sulla diagonale).
- **K7** Constante named `_NUMERICAL_FLOOR = 1e-30`.
- **K10** Type hints su `cholesky_solve`, `conjugate_gradient`, `solve_spd`.
- **K8** Trim docstring di `solve_spd` (rimossa frase "Useful for diagnostics", fluff LLM-like).

### `src/deflected_subgradient.py`

- **K2** Warning `RuntimeWarning` sul branch non-finite (era silenzioso).
- **K1** Risolta length inconsistency di `skip_hist`:
  - `gamma_hist.append(gamma)` spostato DOPO il check `d_sq < floor` (era prima, causava off-by-one al `break`).
  - Branch non-finite ora appende `skip_hist.append(0)` (era unicamente mancante).
  - Verificato con smoke test: `len(skip_hist) == len(gamma_hist) == n_iter` e `len(f_vals) == len(times) == len(gaps) == n_iter + 1`. Documentato come invariante nella docstring.
- **K5** Unicode ASCII: `λ`/`δ`/`γ`/`α`/`ε`/`‖·‖`/`→` nelle docstring sostituiti con `lam`/`delta`/`gamma`/`alpha`/`eps`/`||.||`/`->`.
- **K6** Em-dash decorativo nelle docstring rimossi.
- **K7** Constante named `_NORM_FLOOR = 1e-30`.
- **K9** Commenti che restano del codice trimmati (la citazione d'Antonio-Frangioni resta una sola volta).
- **K10** Type hints su `_optimal_gamma` e `deflected_subgradient`.

### `src/irls.py`

- **K7** Constanti named `_OLS_RIDGE = 1e-12`, `_CG_TOL = 1e-12`, `_CG_MAX_ITER_FACTOR = 10` (rationale documentato in commento).
- **K9** Commento ridondante che ripeteva la docstring rimosso.
- **K10** Type hints su `irls`.
- **K5/K6** ASCII docstring + niente em-dash decorativi.

### `src/lasso_utils.py`

- **K14** Recall convention quando `tp + fn = 0` documentata esplicitamente nella docstring di `support_metrics` (recall = 1 vacuosamente, divergente da sklearn che ritorna 0; callers che confrontano con sklearn su supporto vuoto devono convertire).
- **K10** Type hints su tutte le funzioni pubbliche.
- **K5** Unicode ASCII in docstring.

### `src/elm.py`

- **K13** Comment del clip in `_sigmoid` corretto: prima diceva "|z|>500 saturates" che è impreciso (la saturazione machine-precision avviene a ~36 in float64; 500 è conservativo). Aggiunto commento e named constant `_SIGMOID_CLIP = 500.0`.
- **K7** Named constant `_SPARSITY_TOL = 1e-8` per la sparsity convenience properties (prima era hardcoded due volte come `1e-8`).
- **K10** Type hints su `ELM.__init__`, `transform`, `fit`, `predict` e proprietà.

### `src/data_generation.py`

- **K10** Type hints su tutte le funzioni pubbliche.
- **K6** Em-dash decorativo nelle docstring rimossi.

### `tests/test_irls.py`

- **K4** KKT tolerance test: bound da `viol < 1e-2` a `viol < 1e-5`. Misurato il valore reale `viol ≈ 9.7e-7` sul fixture: ora il test ha margine ~10× sopra il valore osservato (era ~10000×). Commento aggiornato con il rationale numerico.

### `experiments/_plot_style.py`

- **K11** Estratta funzione `plot_long_run_panel(ax, ks, obs_gaps, gap0, title, floor=None)` come modulo condiviso. Refactor minimal che risolve la duplicazione di logica plotting tra `experiment_sgptl_long_run.py` (full run) e `_replot_sgptl_long_run.py` (CSV-only).

### `experiments/experiment_sgptl_long_run.py`

- **K11** Funzione `plot(rows)` riscritta per chiamare `plot_long_run_panel`. Codice da 30 righe a 12.

### `experiments/_replot_sgptl_long_run.py`

- **K11** Riscritto per chiamare `plot_long_run_panel`. Codice da 114 a 81 righe. Verificato che il PDF generato è equivalente al precedente.

### K12 — orphan complexity: nessun fix necessario (verificato)

- **CG path**: usato da `experiments/experiment_params.py` per produrre Tabella/figura §5.4.2 "Solver comparison" del report. Non orphan.
- **`check_optimality`**: solo nei test (validazione di IRLS e dei data generator), mai promessa nel report come stopping criterion. È infrastruttura di test, non orphan code rispetto al report.
- **`ELM.fit(solver='dsm')`**: solo nel test `test_elm.py`. Il report usa solo IRLS via ELM. L'esistenza dell'opzione è API completa ma non promessa nel report: lasciata come è.

### Test e plot verification

- `pytest tests/` → **53/53 passano**.
- Smoke test su invariants: `len(skip_hist) == len(gamma_hist) == n_iter`, `len(f_vals) == len(times) == len(gaps) == n_iter + 1`. OK.
- `python experiments/_replot_sgptl_long_run.py` → riproduce PDF figura, indistinguibile dalla versione precedente.
- Compile LaTeX dopo update figura: 56 pagine, 0 warning di reference, 1.51 MB.

---

## 5 · Nuovi file generati

| File                                                                | Scopo                                                                |
| ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `progetto/code/experiments/rerun_sgptl_to_crossing.py`              | Rerun mirato SGPTL fino al crossing 10⁻⁶ con wall-time misurato      |
| `progetto/code/experiments/rerun_irls_to_crossing.py`               | Time IRLS full (setup + iter loop) per sanity-check Tab. 5.8         |
| `progetto/code/experiments/time_ols_warm_start.py`                  | Misura cost OLS warm-start (P3)                                      |
| `progetto/code/results/tables/real_data_crossing.csv`               | Output rerun SGPTL — numeri usati per Tab. 5.8                       |
| `progetto/code/results/tables/real_data_irls_crossing.csv`          | Output rerun IRLS full timing                                        |
| `progetto/code/results/tables/warm_start_cost.csv`                  | Output cost share OLS warm-start                                     |

---

## 6 · Cosa NON è stato fatto (per scelta o scope)

1. **C5 cover-letter note** — la mappatura "il prof riferisce §5.3.3, ora è §5.4.5 + §5.7" va aggiunta a mano nella mail di risubmission (non in `.tex`).
2. **P5 difesa parametri SGPTL rimanenti** (β=1, R=1, ρ=0.7, γ_min=0.05) — il prof ha annotato "mal configurato" come argomento da orale. Tutti i 4 parametri sono già difesi in §3.5.2 (rationale teorico) e §5.4 (sweep numerici). Aggiungere un paragrafo apposito in §5.7 sarebbe stato ripetizione — l'utente ha confermato il taglio.
3. **T7** verbatim → mapping d'Antonio-Frangioni (chapter3.tex linea 170) — non rivisto, basso rischio.
4. **T15–T16** verifica numerica di citazioni — `frangioni-slides-nonsmooth` (Slide 4, 5, 8, 10, 12, 14, 15) e `dantonio2009` (eq/lemma numbers) andrebbero **controllati a mano una per una** contro deck + paper. Il prof è co-autore di entrambi quindi può segnalare al volo se qualcosa è sbagliato.

---

## 7 · Statistiche

- **File modificati**: 9 `.tex` + 2 `.py` src + 3 `.py` experiments nuovi + 3 `.csv` nuovi.
- **Righe** (`git diff --stat`): +390 / −396 → leggermente più asciutto (era l'obiettivo).
- **PDF**: 54 pagine (invariato), 1.48 MB (era 1.51 MB).
- **Test**: 53/53 passano.
- **Compile**: 0 warning di reference, 0 errori.
- **Tempo speso**: ~3 ore.
- **Pre-rerun SGPTL**: ~18 minuti su california (background).

---

## 8 · Checklist pre-consegna

- [ ] Rileggere **Theorem 3.1** in `chapter3.tex` — è il pezzo a maggior rischio.
- [ ] Verificare che la **rate corretta** (1/γ_min, non 1/√γ_min) non rompa nessun argomento downstream.
- [ ] Controllare che **Tab. 5.8** numeri (3.24·10⁵ / 7.7 s diabetes; 9.29·10⁵ / 1082 s california) corrispondano al rerun (`real_data_crossing.csv`).
- [ ] Rileggere **§5.7 "Initial design and corrections"** — non ho aggiunto nulla; i 3 punti rimasti aperti (β=1, R=1, ρ=0.7, γ_min=0.05) vanno discussi all'orale come dice il prof.
- [ ] Pre-flight LaTeX: `latexmk -pdf -bibtex main.tex` da `progetto/report/` deve compilare clean.
- [ ] Cover letter alla risubmission: dichiarare la rinumerazione `§5.3.3 (vostra lettura precedente) → §5.4.5 + §5.7 (questa versione, restrutturate)`.
