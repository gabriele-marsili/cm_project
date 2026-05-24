# Review pre-consegna — CM Project 25 ML (Group 63)

Synthesis di 4 agent paralleli (chapters 1–3, chapters 4–6, code, style sweep) +
cross-check con i 5 punti dell'email del prof.

Severità: **CRITICO** = must-fix prima della consegna · **IMPORTANTE** = correzione sostantiva ·
**STILE** = pattern LLM/ripetizioni (esplicitamente flaggati dal prof).

Workflow: applico fix per gruppi numerati. Tu approvi/scarti ogni gruppo prima che proceda.

---

## GRUPPO 0 — I 5 punti dell'email del prof (PRIORITÀ MASSIMA)

### P1 [CRITICO] §5.3.1 — claim su Pr(w_i = 0)
**Stato attuale**: chapter3.tex:139 dice "non-generic … cold start w_0=0 is the only case …".
**Mossa del prof**: "la possibilità che w_i sia 0 tende ad essere bassa quindi non mi pare critico" — chiede solo di dirlo esplicitamente.
**Fix**: aggiungere una frase: "for i≥1 l'update w_{i+1}=w_i − α_i d_i ha w_i con distribuzione continua, quindi Pr((w_i)_j=0)=0 per ogni j; il caso w_0=0 è gestito dalla convenzione sign(0)=0."

### P2 [CRITICO] Theorem 3.1 — bug rilevato, fix concreto
**Stato attuale**: chapter3.tex:179–181 asserisce per-step inequality con costante β_i(2−β_i), citando [Slide 15 Frangioni].
**Bug rilevato in d'Antonio–Frangioni 2009**: il paper, p.369 proof of Theorem 3.5 eq. (3.17), dà la per-step inequality per direzione deflessa con costante **β_k(2α_k − β_k)**, NON β_k(2 − β_k).

Derivazione (caso esatto σ_k=0, γ_k=0, x̄=x*):
- Eq. (2.9): ‖x_{k+1}−x̄‖² − ‖x_k−x̄‖² ≤ −2ν_k⟨d_k, x_k−x̄⟩ + ν_k²‖d_k‖²
- Cor. 3.2 (via Lemma 2.4 / cond. 2.13 + Lemma 3.1): ⟨d_k, x_k−x̄⟩ ≥ α_k(f_k−f*) − [f(x̄)−f*+σ̄_k]
- ν_k = β_k λ_k/‖d_k‖², λ_k = f_k−f* (caso esatto)
- Combinando: descent ≤ −β_k(2α_k − β_k)(f_k−f*)²/‖d_k‖²

Il "2" del report viene da convessità ⟨g_k, x_k−x*⟩ ≥ f_k−f* (vale solo per plain subgradient, non per deflesso d_k).

**Nel nostro algoritmo** β_i = γ_i = α_i (clip mette β_i = min(1,γ_i)=γ_i con γ_i∈[γ_min,1]), quindi:
- per-step descent factor = γ_i(2γ_i − γ_i) = **γ_i² ≥ γ_min²** (NON γ_i(2−γ_i)≥γ_min)
- telescoping: (k+1) min(f_i−f*)² ≤ L²‖w_0−w*‖²/γ_min²
- **rate corretta**: f̄_k − f* ≤ L‖w_0−w*‖ / **(γ_min · √(k+1))** (NON √(γ_min·(k+1)))
- iter to ε: k+1 ≥ L²‖w_0−w*‖²/(γ_min² · ε²); inflazione vs plain Polyak = **1/γ_min² = 400×** per γ_min=0.05 (report dice 20×, è la radice)
- order O(L²/ε²) inalterato

**Bonus**: la rate corretta (400× peggiore) si allinea meglio con l'osservazione empirica di §5.7 (SGPTL ~100× iter per decade vs IRLS), mentre il 20× del report è troppo ottimista.

**Fix concreto in chapter3.tex**:
- Theorem 3.1 statement (linea 150): cambiare denominatore da `\sqrt{\gamma_{\min}(k+1)}` a `\gamma_{\min}\sqrt{k+1}`
- Proof (linee 176–197): rimpiazzare riferimento a [Slide 15] con citazione esplicita a Corollary 3.2 + eq. (3.17) di [dantonio2009deflected]; sostituire β_i(2−β_i) con β_i(2α_i−β_i); ricalcolare costante telescopica e iter inflation
- Linea 197: "20× multiplier" → "400× multiplier" e "1/γ_min" → "1/γ_min²"
- Update propagation in §5.2.3 (results.tex:123–130) e §6 (conclusions:39–47) dove la rate è citata

### P3 [CRITICO] Costo del warm-start w_0 non discusso nel cost totale
**Stato attuale**:
- §4.3.2 (comparison.tex:76–84) lista il costo asintotico O(MH²+H³) ma non lo numerizza
- §5.3 e §5.5 non quantificano OLS-cost / total-IRLS-cost
- §5.3 Defaults raccomanda warm start "ovunque" senza weighing del costo
**Suggerimento del prof**: forse il punto sinistro di Figure 5.3 (gap-vs-CPU) lo mostra — se sì, **dirlo esplicitamente**.
**Fix**: 
- Aggiungere a §5.3 una micro-tabella o 3 righe: "Per diabetes H=200, Cholesky OLS = X ms, totale IRLS = Y ms, OLS è Z% del totale. Stesso conto su california. Per H grande il warm start è giustificato perché Z stays < N%."
- In §4.3.2 spiegare: "il costo OLS è caricato in tab. 5.8 al wall-clock totale di IRLS-warm" se non già fatto.
- In §6 menzionare il trade-off invece di tacerlo.

### P4 [CRITICO] §5.3.3 — δ_0 calibrato usando f* = "training sul test set"
**Stato attuale**:
- §3.5.2 (chapter3.tex:117–119) introduce δ_0 = c·f(w_0) (admissible)
- §5.4.5 (results.tex:570–598) confronta admissible scale vs **Family C = c·f*** (oracolo) per giustificare il default
- App. A usa la stessa Family C come riferimento; su california H=200 usa "IRLS-converged proxy" come f*, doppia circolarità
- §5.7 riconosce il problema e reframes ma §5.4.5/App.A restano contraddittorie
**Mossa del prof**: "usare il valore ottimo per determinare i parametri algoritmici … è come usare il test set per fare training in ML: semplicemente non si fa. Chiarite questo punto, quantomeno discutetelo."
**Fix**:
- §5.4.5 + App.A: aggiungere disclaimer all'inizio dichiarando esplicitamente che la sezione è un'**analisi astratta di sensitività** non usata per scegliere il default; il default operativo è δ_0 = c·f(w_0) come spiegato in §3.5.2 e usato in tutti gli esperimenti di §5.5–§5.6.
- Su california H=200 (App.A): dichiarare la doppia circolarità (proxy IRLS-converged usato come f*, poi f* usato come Family C) e marcare l'instance come "abstract only".
- In §3.5.2 (chapter3.tex:119) aggiungere "All experiments in Ch.5 use δ_0=c·f(w_0); the c·f* scale appears only as abstract reference in Fig. delta0-families."

### P5 [IMPORTANTE] §5.7 SGPTL "molto mal configurato" (prof anticipa orale)
**Stato attuale**: §5.7 spiega correzione di R e fallback + reframe δ_0. Ma non difende esplicitamente:
- β=1 (perché senza shrinkage?)
- R=1 (è 1000× il step-length tipico → ~10³ iter tra contractions; "matched" è post-hoc)
- ρ=0.7 (best su 1/3 instances; ρ=0.5 sarebbe il compromesso safer)
**Fix**: aggiungere paragrafo "These remaining values are compromises, not per-instance optima. We hold β=1 because [Polyak descent argument from chapter3:111], R=1 because [travel-distance argument], ρ=0.7 because [sweep §5.4.4 shows it's within factor-2 of best on all three instances]. A per-instance tuning would buy ≤ factor-2 in final gap (see §5.4.4)."

### P6 [IMPORTANTE — direttiva del prof] Test-MSE rimangono ma deemphasized
**Stato**: §5.6 line 836 ancora presenta MSE come *risultato* ("optimisers agree"); §6:78–81 ha già il disclaimer corretto.
**Fix**: aggiungere a §5.6 una frase: "Test MSE figures are reported for completeness but are not part of the optimisation comparison (cf. comando §4.5 — out-of-sample performance is not in scope here)."

---

## GRUPPO 1 — Coerenza interna (CRITICI)

### C1 [CRITICO] Numeri inconsistenti Clarabel↔IRLS agreement
- Table 5.8 caption: "nine significant digits"
- §5.6 line 777: "six significant digits"
- §6 line 35: "six significant digits"
- CSV reali: diabetes ~10 digits, california ~10 digits
**Fix**: scegliere UN numero ("ten" è quello vero; "six" è conservativo) e usarlo in tutti e 3 i posti.

### C2 [CRITICO] Table 5.8 wall-times SGPTL sono interpolati/extrapolati, non misurati
- `real_data.csv` ha iter_dsm=8000 (budget cap)
- I numeri 3.25·10⁵ iter / 6.6s (diabetes), 9.33·10⁵ iter / 1115s (california) vengono dal long-run trace, **interpolati** al crossing 10⁻⁶
- Su california il crossing avviene tra k=8·10⁵ e k=8·10⁶ → l'iter count è interpolato; il wall-time è estrapolato per scaling
**Fix**: aggiungere alla caption: "SGPTL iter and wall-clock at gap 10⁻⁶ are linearly interpolated from the long-run trace of §5.3.1 (the 8000-iter budget does not reach this threshold)." CLAUDE.md vieta "no allucinazioni di esperimenti" se non dichiarate.

### C3 [CRITICO] Table 5.1 caption: f* reference mismatch
- Caption dice: real-data f* = IRLS+CVXPY-Clarabel
- Ma "IRLS final gap <10⁻¹¹" sarebbe vs IRLS-converged (autoreferenziale)
- CSV: IRLS vs Clarabel ~10⁻¹⁰ su diabetes, ~6.5·10⁻⁸ su california → **non** <10⁻¹¹ su california
**Fix**: o caption diventa "vs IRLS-converged proxy" (e si dichiara la circolarità), o numero corretto è ~6.5·10⁻⁸ su california.

### C4 [IMPORTANTE] IRLS california wall-time anomalo
Table 5.8: california 34 iter / 18.7 ms su M=16512, H=200. Per-iter ~0.55ms → 1.3 TFlop/s.
Confronto con diabetes 0.08 ms/iter su M=354 → ratio osservato 6.7× ma teorico ~46×.
**Fix**: verificare se 18.7 ms include o esclude la precomputation A=XᵀX. Se include, rerun con timer separato per setup vs iter. Se no, esplicitare nella caption.

### C5 [IMPORTANTE] §5 cita §5.3.3 ma sezione si chiama §5.4.5
L'email del prof riferisce "§5.3.3 δ_0" ma nel PDF corrente è §5.4.5. Discrepanza tra il numero che il prof vede e quello attuale.
**Fix**: nel cover-letter di risubmission specificare la mappatura "§5.3.3 in your reading → §5.4.5 + §5.7 in this version, both restructured".

---

## GRUPPO 2 — Teoria (Ch. 1–3): correzioni e citazioni

### T1 [CRITICO] report.tex:27 — "with high probability, full rank when M≥H" senza citazione
Claim affidato a "random projections through a nonlinear function produce linearly independent features" — vuoto.
**Fix**: o cita Huang 2006 con ipotesi esplicite su σ e distribuzione di W_1, o rephrase "We assume X has full column rank, which holds for generic random W_1 with sigmoidal σ when M≥H."

### T2 [IMPORTANTE] algo1.tex Theorem 2.1 — ipotesi violata su LASSO
Theorem 2.1 richiede |(w*)_i| > ε_thr per **ogni** i, ma il punto di LASSO è (w*)_j=0 sparsi. La "verification" alle linee 226–229 dice "components going to zero drop out", che **non è quello che il theorem dice**: il theorem fa un'ipotesi che è violata.
**Fix**: o restate Theorem 2.1 per l'active set (provare versione active-set), o citare Daubechies et al. Theorem 5.3 che gestisce gli inactive components.

### T3 [IMPORTANTE] algo1.tex:222 — "linear convergence … Tikhonov preconditioner"
- "Linear convergence" è enunciata, non provata su ELM LASSO
- Daubechies et al. provano linear convergence sotto Null Space Property per ℓ_q, q≤1 — ipotesi da verificare
- "Tikhonov preconditioner" è terminologia sbagliata: W_k^T W_k è regolarizzatore, non preconditioner
**Fix**: "Daubechies et al. [ref] prove linear convergence under [NSP]. On ELM LASSO these hypotheses [are/are not] verified because [...]; in our experiments we observe a linear-looking decay (§5.x)." Rimuovere "Tikhonov preconditioner" → "regularising term".

### T4 [IMPORTANTE] chapter3.tex — ripetizione di Condition (3.5) di [dantonio2009] tre volte
Linee 33, 114, 160–164. Stessa condizione enunciata 3×.
**Fix**: enunciarla una volta in §3.5.2 (parameter calibration); §3.2 e §3.6 referenziano.

### T5 [IMPORTANTE] chapter3.tex:201 — verification trae in ipotesi non necessarie
Theorem `thm:dsm_convergence` richiede solo "f* > -∞ attained at some w*"; la verification cita uniqueness via Prop. `prop:irls_prop` che non serve.
**Fix**: lasciare solo "coercivity ⇒ compactness ⇒ f* attained".

### T6 [IMPORTANTE] chapter3.tex:33 — "sweep numbers" leakano in sezione teorica
"A sweep γ_min ∈ {0.05, 0.1, 0.2, 0.5} … gives record gaps in [10⁻⁷, 10⁻⁵]" dentro §3.2, prima che §5 sia esposto. Forward-ref a §5 violato.
**Fix**: cut dalla §3.2; tenere solo in §3.5.2 + §5.4.3.

### T7 [IMPORTANTE] chapter3.tex:171–172 — "verbatim" troppo forte su patience strategy
"Lines 16–21 implement that strategy verbatim under μ↔ρ" → l'incremento di r_i nella tua implementazione è ∑α_j‖d_j‖; verificare se Lemma 3.8 di [dantonio2009] usa lo stesso.
**Fix**: se identico, OK; se up-to-constant, sostituire "verbatim" con "matching Lemma 3.8 up to a positive constant".

### T8 [IMPORTANTE] chapter3.tex:204 — contraddizione constanti vs empirico
Proof conclude: floor inflates constant by 1/√γ_min (worst-case **peggiore**). Empirico (dichiarato altrove): floor migliora. Contraddizione da riconoscere.
**Fix**: una frase: "Worst-case the floor inflates the rate constant by 1/√γ_min; empirically the opposite, because the floor binds only on the few near-stationarity steps of §3.2."

### T9 [IMPORTANTE] chapter3.tex:10 — footnote Bubeck Thm 3.2 ambigua
"$O(L^2/ε^2)$ vs $O(L/ε^2)$ … same order after rescaling ε by L" — non sono lo stesso order; rescaling è corretto ma confusione. Inoltre "Theorem 3.2" di Bubeck va precisato (capitolo).
**Fix**: rephrase "Some references absorb L into ε reporting O(1/ε²); we keep L explicit." Citare capitolo/sezione di Bubeck.

### T10 [STILE/IMPORTANTE] report.tex:73 — `\sum_i^H` (missing lower bound)
Cosmetico ma visibile. Fix: `\sum_{i=1}^{H}`.

### T11 [IMPORTANTE] report.tex:60 — restatement of strict convexity 30 righe prima di Prop 1.1
Cut la frase intermedia o la Proposition (tenere la Proposition).

### T12 [IMPORTANTE] report.tex:77 — \ref usato dove serve \Cref
Renderizza "(2) and (3)" invece di "Chapter 2 and Chapter 3". Fix.

### T13 [IMPORTANTE] algo1.tex:96 — derivazione di λ_IRLS=λ_LASSO/2 cancellata
Era commentata e mai rimpiazzata. Reader vede l'equazione `eq:lambda_relation` senza giustificazione on-page.
**Fix**: aggiungere una riga "il factor 1/2 è la majorization constant da `eq:irls_quadratic`".

### T14 [STILE] algo1.tex:190 — `\ref{eq:optimality}` invece di `\eqref{eq:optimality}`
Cosmetico.

### T15 [IMPORTANTE] verifica slide numbers di `frangioni-slides-nonsmooth`
Citato con Slide 4, 5, 8, 10, 12, 14, 15. Verificare ogni numero contro il deck reale (il prof è co-autore).

### T16 [IMPORTANTE] verifica numeri di teorema/lemma di `dantonio2009deflected`
"(3.1)", "(3.5)", "Theorem 3.7", "Lemma 3.8" — verificare contro il paper pubblicato.

---

## GRUPPO 3 — Esperimenti (Ch. 4–6): coerenza e completezza

### E1 [IMPORTANTE] §6 troppo lungo, restatement di §5
Conclusions chapter ricapitola tutto §5. Trim a 5–8 righe: lista predizioni, "confirmed at §5.5.3, §5.6, §5.3", drop play-by-play di california timings.

### E2 [IMPORTANTE] §6 paragrafo "Recommendation" tono promozionale
Suona da sales pitch. Rendere neutrale + bullet citation alle sezioni di supporto.

### E3 [STILE] §5 line 7–10 — opener promotional
"the guiding questions are the two of a numerical study" + "play out". Replace con frase neutra.

### E4 [STILE] §5 lines 444–500 — "chain reaction" / closing meta-comment
Tenere il fenomeno descritto; cut "the gap drops by two more orders of magnitude" (restatement della tabella).

---

## GRUPPO 4 — Ripetizioni cross-sezione (priorità del prof)

Per ogni concetto: tenere SOLO la versione marcata KEEP, sostituire le altre con `(cf. §X)`.

### R1 — "OLS warm start sits within 0.46 of f* on california / nothing to optimise"
- ❌ ch.3:108 (cut frase, replace con forward ref a §5.3)
- ✅ KEEP ch.5:88–93 (§5.1 Limitations) OR ch.5:291–302 (§5.3 SGPTL paragraph) — scegliere una
- ❌ ch.5:291–302 OR ch.5:88–93 (cut quella non scelta)
- ❌ ch.6:75–78 (compress a "cf. §5.3")
- ❌ appendix:25–31 (replace paragrafo con one-line ref)

### R2 — "Cost per decade: IRLS constant increment, SGPTL ~100× multiplier"
- ✅ KEEP ch.4:34–38 (definizione teorica)
- ❌ ch.5:683–686 (cut)
- ✅ KEEP ch.5:722–727 (proof empirica con table)
- ❌ ch.6:44–47 (cut)

### R3 — "ρ ∈ [0.3, 0.7] within factor 2, adopt ρ=0.7"
- ❌ ch.3:122 (trim a "we adopt ρ=0.7, sensitivity sweep §5.4")
- ✅ KEEP ch.5:543–546 (full sweep)

### R4 — "γ_min=0.05 marginally best, sweep [0.05, 0.1, 0.2, 0.5]"
- ❌ ch.3:33 (cut sweep, lasciare solo "floor needed")
- ⚠️ ch.3:114 (rationale + costo 1/√γ_min, tenere; numerical sweep → §5)
- ✅ KEEP ch.5:505–511 (full sweep + decisione)

### R5 — "Theorem still bounds, empirical below envelope" (PROF-FLAGGED PATTERN)
- ❌ ch.5:240–249 (CUT — questo è il pattern esplicitamente criticato da CLAUDE.md)
- ✅ KEEP ch.5:335–340 (table-supported)
- ❌ ch.5:361–363 caption (trim restatement)
- ❌ ch.6:39–44 (cut "well below envelope")

### R6 — "OLS warm start is IRLS default everywhere"
- ✅ KEEP ch.2:173 (definizionale)
- ❌ ch.2:251 (cut intero paragrafo — restatement)
- ❌ ch.5:262–263 (cut "natural IRLS default")
- ✅ KEEP ch.5:316–317 (Defaults paragraph)

### R7 — "SGPTL needs thresholding for sparsity"
- ✅ KEEP ch.3:214 (definizionale)
- ✅ KEEP ch.4:92–94 (comparison context)
- ❌ ch.5:627 (cut "as anticipated")

### R8 — "δ-contraction staircase"
- ✅ KEEP ch.3:59 + ch.3:170 (definizione + verifica)
- ❌ ch.5:246 (cut closing meta-comment)
- ✅ KEEP ch.5:499–504 (figure context)

### R9 — "cold start is the one that genuinely optimizes"
- ❌ ch.3:108 (già coperto in R1 cut)
- ✅ KEEP ch.5:319–321 (Defaults, neutrale)

---

## GRUPPO 5 — Sweep stilistico anti-LLM (aggressivo)

### S-cut puntuali (replace o cut)
- **"the picture is different"** → cut/sostituire (ch.5:289, ch.5:853)
- **"play out"** (ch.5:10) → cut
- **"bear (it) out"** (ch.5:608) → "confirm"
- **"tells the … story"** (ch.5:673) → "reverses the ordering"
- **"sweet spot"** (ch.5:399) → "default"
- **"genuinely"** (ch.3:108) → cut
- **"silently"** (ch.3:59) → cut
- **"sort of personalized penalization"** (algo1.tex:31) → "weight |(w_k)_i|^{-1}"
- **"core idea"** (algo1.tex:6, header §2.1) → drop
- **"This reweighting mechanism is what drives IRLS to produce sparse iterates"** (algo1.tex:31) → cut (closing meta-comment)
- **"a sort of"** ovunque → cut
- **"easy to monitor and to debug"** (algo1.tex:257) → "any non-decreasing step indicates an implementation error" (più stretto)
- **"particularly nice"** (report.tex:75) → "induces sparsity at the cost of"
- **"Let's"** (report.tex:82, 88; comparison.tex:33) → "We" o cut

### S-empty intensifiers (rimuovere ogni occorrenza)
- `essentially` (ch.5:92, 247; chapter3.tex:varie)
- `actually` (ch.5:320, 502; ch.6:varie)
- `naturally` (ch.3:varie)
- `clearly`, `dramatically`, `decisively` (verificare con grep)
- `Importantly,`, `Crucially,`, `Notably,`, `It is worth noting that`

### S-em-dash decorativi (sostituire con virgola o parentesi)
- ch.3:33 "--- the iterate freezes"
- ch.3:119 "--- the latter being usable only as a reference …"
- ch.3:212 "--- the two coincide at λ=0 and vary continuously with it ---"
- ch.5:284 "--- consistent with the well-conditioned regime …"
- ch.5:300 "--- the warm-start number 0.46 therefore measures …"
- ch.5:500 "--- the gap drops by two more orders of magnitude"
- ch.6:42, ch.6:46 (analoghi)

### S-rule-of-three forzate
- ch.5:608 "accuracy/sparsity, scaling, and iterations" → bullet list invece di prosa-triadica
- comparison.tex:127 "amortizes, induces sparsity, while only having to tune" → tightenare

### S-closing meta-commentary (cut)
- comparison.tex:38 (restate dei bullet sopra)
- ch.5:484–500 trailing clause
- ch.5:684–685 "Numbers therefore understate IRLS advantage"
- ch.5:727–729 "This is the quantitative form …"
- ch.5:854–856 "follows the envelope …"
- ch.5:922–925 "qualitative ranking unaffected"

### S-other typos
- algo1.tex:258 "numercial" → "numerical"
- algo1.tex:259 "Setting it 10^-8" → "Default: 10^-8"

---

## GRUPPO 6 — Codice (src + experiments)

### K1 [IMPORTANTE] `skip_hist` length inconsistency
`src/deflected_subgradient.py:77,110,112`: append solo sui branch `num<=0` e "not skipped". Early break (`:89-90`) e non-finite (`:117-124`) non appendono → misalign con `f_vals` downstream.
**Fix**: append sentinel sul non-finite branch + accounting after early break.

### K2 [IMPORTANTE] Non-finite branch silenzioso
`deflected_subgradient.py:117-124`: quando w_new non-finite, iterate non avanza ma `f_vals.append(f_curr)` duplica → plot mostra stallo finto.
**Fix**: log warning o terminate.

### K3 [IMPORTANTE] CG breakdown non guarded
`src/linear_solvers.py:38-48`: no check `p@Qp<=0` (loss of SPD su Q ill-conditioned con 1/eps_thr~1e8), no check `rz==0`. Possibili NaN silenti.
**Fix**: aggiungere guard + fallthrough a Cholesky.

### K4 [IMPORTANTE] KKT tolerance loose nei test
`src/lasso_utils.py:51-65` + `test_irls.py:46`: viol < 1e-2 con lam=0.05 = 20% di lam. Tollerare ~ eps_thr·lam.

### K5 [STILE] Unicode in source code (λ, δ, γ, ε, →, ‖·‖² in docstrings)
`irls.py`, `deflected_subgradient.py`, `lasso_utils.py`, `data_generation.py`. Portability hazard + signature LLM.
**Fix**: ASCII in source; math nel report.

### K6 [STILE] Em-dash decorativi in docstring
`linear_solvers.py:1`, `data_generation.py:1,79-82`, `lasso_utils.py:34-35`, `irls.py:47-49`.

### K7 [STILE] Magic numbers senza giustificazione
- `irls.py:36` 1e-12 ridge → constant nominata o riferimento
- `irls.py:69` `max_iter=10*H` per CG → giustificare o droppare
- `deflected_subgradient.py:19,23` 1e-30 → documentare
- `elm.py:73,79` 1e-8 sparsity → match con `support_metrics`
- `lasso_utils.py:51,60` zero_tol=1e-6 → docstring

### K8 [STILE] Docstring più lunghi del codice
`solve_spd` (linear_solvers.py:53-66), `irls.py:34-49`. Trim.

### K9 [STILE] Commenti che restano il codice
`irls.py:64`, `deflected_subgradient.py:97-100`. Move citation to docstring.

### K10 [IMPORTANTE] Type hints mancanti su funzioni pubbliche
`irls`, `deflected_subgradient`, `f_lasso`, `solve_spd`, `ELM.__init__`. Inconsistente con uso di `NamedTuple` altrove.

### K11 [STILE] `_replot_sgptl_long_run.py` smell
Plotting duplicato vs `_plot_style.py`. Refactor.

### K12 [IMPORTANTE] Code-report coherence: orphan complexity
- CG path implementato e testato; se §5 usa solo Cholesky, è orphan
- `check_optimality` solo nei test, mai usato come stopping criterion → se report lo cita, dev'essere invocato
- `ELM.fit(solver='dsm')` → se report usa solo IRLS sull'ELM, è orphan
**Fix**: o togliere orphan, o aggiungere riga in §4.4 (code description) che dice "CG implementato per benchmark §5.4.2; check_optimality usato in tests/...".

### K13 [STILE] `_sigmoid` clip ±500 commento fuorviante
`elm.py:11-13`. Threshold pratico ~36/700; il valore 500 funziona ma il commento "stability via clipping" è impreciso.

### K14 [STILE] Recall convention quando supporto vero è vuoto
`lasso_utils.py:37`: recall=1.0 (sklearn fa 0+warning). Documentare in docstring.

---

## Esecuzione proposta

Approvi gruppi e io eseguo nell'ordine: **GRUPPO 0 → GRUPPO 1 → GRUPPO 2 → GRUPPO 3 → GRUPPO 4 → GRUPPO 5 → GRUPPO 6**.
Ogni gruppo è un commit atomico. Dopo ogni commit ricompilo PDF e verifico no-regression.

Punti dove mi serve la TUA decisione prima di muovermi:
1. **P2 Theorem 3.1**: opzione (a) riprovare con citazione corretta di d'Antonio–Frangioni 2009 vs (b) rinunciare alla rate e tenere solo convergenza qualitativa. **A** è più ambizioso ma richiede di rileggere il paper; **B** è safer per la consegna.
2. **R1**: tenere ch.5:88–93 (§5.1 Limitations) o ch.5:291–302 (§5.3 paragraph)?
3. **C1**: scegliere "ten" (vero) o "six" (conservativo) per Clarabel↔IRLS agreement.
4. **C2**: rerun SGPTL real-data fino al crossing 10⁻⁶ (costoso, possibili ore di compute) oppure aggiungere disclaimer in caption (gratis, ma il prof potrebbe notare). **B** è la mossa pragmatica.
