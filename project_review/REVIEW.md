# Review pre-consegna finale — CM Project 25 ML (Group 63)

**Data**: 2026-05-27
**Branch**: `main` @ `ca5475f`
**Stato baseline**: commits `3337573` (Matthew, cap 1→4) + `19c1021`/`f69bf47`/`56d58da`/`ca5475f` (sweep stilistico cap 3/5/6 + PDF rebuild).

Audit eseguito da 4 review pass paralleli su domini disgiunti (cap 1+2, cap 3+4, cap 5+6+appendici, code+bibliografia). Sintesi consolidata di seguito.

**Severità**:
- **CRITICO** = bug visibile in lettura, errore nel PDF, o mismatch report↔codice. Da fixare PRIMA della consegna.
- **IMPORTANTE** = correzione sostantiva (citazione mancante, ipotesi non discussa, passaggio algebrico mancante).
- **STILE** = residui di pattern LLM/colloquialismi/ripetizioni.

---

## Executive summary

| Capitolo / dominio | CRITICO | IMPORTANTE | STILE | Stato |
|---|---:|---:|---:|---|
| Cap 1 (Introduzione) | 1 | 2 | 3 | Ref rotta + 2 citazioni vaghe |
| Cap 2 (IRLS) | 1 | 2 | 2 | Asserzione non citata + 2 passaggi algebrici |
| Cap 3 (DSM) | 0 | 0 | 1 | Solo "amortizes" figurato |
| Cap 4 (Comparison) | 1 | 2 | 1 | Cross-ref rotta + rate mismatch + citation missing |
| Cap 5 (Results) | 0 | 0 | 0 | **pulito** (sweep esaustivo) |
| Cap 6 (Conclusions) | 0 | 0 | 0 | **pulito** |
| Appendici | 0 | 0 | 0 | **pulito** |
| Code (src/) | 1 | 0 | 1 | `rho` default mismatch + magic number |
| Bibliografia | 0 | 1 | 0 | 3 orphan refs + 1 key/year mismatch |
| **Totale** | **4** | **7** | **8** | **19 findings** |

**Verdetto**: il report è quasi pronto. Quattro CRITICI sono tutti fixabili in <30 min. Cap 5/6/appendici sono passati pulitamente — lo sweep precedente ha fatto il suo lavoro. Il problema principale residuo è in cap 1 e cap 4 (che il sweep non aveva toccato).

---

## CRITICI (must-fix prima della consegna)

### CR-1 — Cap 1 r124: cross-reference rotta `\ref{alg:dsm}` invece di `\ref{chap:dsm}`

**File**: `progetto/report/1_introduction/report.tex:124`

**Codice attuale**:
```latex
\textit{subgradients} (see Chapter~\ref{alg:dsm}).
```

**Problema**: `alg:dsm` è il label di `\begin{algorithm}` (chapter3.tex:69), non il label del capitolo. Il LaTeX renderà "Chapter 1" (il numero dell'algoritmo) invece di "Chapter 3". Lo stesso file usa correttamente `\ref{chap:dsm}` alla riga 74.

**Fix**: `\ref{alg:dsm}` → `\ref{chap:dsm}`. Verifica anche con `latexmk` che il render sia "Chapter 3".

---

### CR-2 — Cap 4 r41: "As expressed before" autocitazione rotta

**File**: `progetto/report/4_algo_comparison/comparison.tex:41`

**Problema**: la frase "As expressed before" rimanda al cap 2 (IRLS) ma il lettore è già nel cap di comparison; il rimando è opaco e rompe la self-containedness della sezione.

**Fix proposto**: sostituire con un riferimento esplicito al teorema:
> "As Theorem 2.1 (Chapter 2) establishes, X has full column rank..."

(o riformulazione equivalente che cita il risultato per nome anziché in modo anaforico).

---

### CR-3 — Cap 4 r35: rate SGPTL inconsistente con Theorem 3.1 (√k vs √(k+1))

**File**: `progetto/report/4_algo_comparison/comparison.tex:35` vs `progetto/report/3_algo_2_DSM/chapter3.tex:155`

**Cap 4 (r35) scrive**:
> f̄_k − f* ≲ L ‖w₀−w*‖ / (γ_min · √k)

**Cap 3 (r155, Theorem 3.1)** scrive:
> f̄_k − f* ≤ L ‖w₀−w*‖ / (γ_min · √(k+1))

**Problema**: il denominatore differisce per `+1`. Asintoticamente equivalente, ma in un report tecnico **un capitolo che riassume un teorema deve usare la stessa formula del teorema**. Il prof noterà la discrepanza.

**Fix**: allineare cap 4 a `√(k+1)` (come Theorem 3.1), oppure aggiungere esplicitamente "asintoticamente √k" se si vuole tenere la versione semplificata.

---

### CR-4 — `deflected_subgradient.py:60`: default `rho=0.95` non coerente con report

**File**: `progetto/code/src/deflected_subgradient.py:60`

**Discrepanza**:
- Modulo `deflected_subgradient.py:60`: `rho: float = 0.95` (default)
- Report §3.4.1, §5.4.4, §5.6:777: `ρ = 0.7` (default operativo)
- `experiment_real_data.py:38`: `DSM_RHO = 0.7` (override esplicito; `DSM_RHO_OLD = 0.9` come "as-submitted")

**Conseguenza**: chi chiama `deflected_subgradient(...)` senza passare `rho` esplicitamente ottiene 0.95, non 0.7. Tutti gli script di esperimento passano `rho=DSM_RHO=0.7`, quindi i risultati del report non sono inficiati. Ma:
- il prof può ispezionare il codice e trovare la discrepanza;
- chi userà il modulo in futuro (anche solo per replicare) cade nella trappola.

**Fix**: `rho: float = 0.95` → `rho: float = 0.7` in `deflected_subgradient.py:60`. Eventualmente aggiungere un commento `# matches report §3.4.1 / §5.4.4 default`. Verificare poi che `_OLD = 0.9` in experiment_real_data.py resti come riferimento storico ma non sia il default attivo.

---

## IMPORTANTI

### IM-1 — Cap 1 r27: claim full-rank ELM appoggiato a verifica numerica invece che a citazione completa

**File**: `progetto/report/1_introduction/report.tex:27`

Il claim "M ≥ H ⇒ full rank con W₁ random + σ liscio" è citato come `[Huang 2006, Thm 2.1]` ma il collegamento al regime sperimentale ELM-LASSO è fatto via "verified numerically in Section X". Il prof può chiedere "perché non lo dimostrate?" — la dimostrazione completa di Huang richiede sigmoide infinitamente derivabile, mentre noi ne usiamo solo C∞ generico.

**Proposta**: aggiungere mezza frase che dichiari esplicitamente quali ipotesi Huang richiede (random Gaussian, σ analytic non-polynomial) e che noi le rispettiamo nella scelta `sigmoid`.

### IM-2 — Cap 1 r60: "destroys the closed-form structure" è asserzione non citata

**File**: `progetto/report/1_introduction/report.tex:60`

Tecnicamente vero ma vago. Può essere lasciato com'è (è didattico nell'intro) oppure precisato in mezza riga: "the non-smoothness of ‖w‖₁ at coordinates where wᵢ=0 destroys the differentiability needed for closed-form normal equations." Marginale.

### IM-3 — Cap 2 r14: salto algebrico nella majorization IRLS

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:14`

Il passaggio da (|wᵢ| − |wₖᵢ|)² ≥ 0 alla majorization (eq. 16) salta 2-3 step algebrici (espansione del quadrato, divisione per 2|wₖᵢ|, riarrangiamento). Per il prof è banale ma per la rubrica vale uno step.

**Proposta**: aggiungere una riga di derivazione esplicita prima di (16), o spostare in appendice e referenziare.

### IM-4 — Cap 2 r223: claim "NSP non vale su ELM" non citato

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:223`

> "no sparsity-recovery guarantee is claimed for the random W₁ regime"

È un'asserzione sostanziosa (negativa) ma non ha riferimento. Bisognerebbe o citare un lavoro che lo dimostra (Donoho-Tanner phase transitions for random Gaussian + sigmoidal), oppure qualificarla: "we are not aware of a sparsity-recovery guarantee for the random W₁ regime; in our experiments support recovery succeeds on the synthetic instance (§5.5.1) but is not measured on real data."

### IM-5 — Cap 4 r33: citation missing per lower bound nonsmooth

**File**: `progetto/report/4_algo_comparison/comparison.tex:33`

> "DSM achieves the optimal sublinear O(L²/ε²) rate for nonsmooth convex functions"

Manca la citazione del lower bound. In cap 3 r212 c'è già la coppia corretta `[Slide 8 Frangioni], [Theorem 3.13 Bubeck]`. Cap 4 dovrebbe usare le stesse.

**Fix**: aggiungere `\cite[Slide 8]{frangioni-slides-nonsmooth}` (e opzionalmente Bubeck).

### IM-6 — Bibliografia: `brannlund1993generalized` chiave/anno mismatch

**File**: `progetto/report/references.bib:154-162`

Chiave dice `1993`, campo `year = {1995}`. Verificare contro il paper originale e correggere la chiave (`brannlund1995generalized`) o l'anno. Marginale ma il prof può aprire il bib.

### IM-7 — Bibliografia: 3 entries orphan (citate da nessuno)

**File**: `progetto/report/references.bib`

- `cortinovis-leastsquares` (rr 87-92) — non citato; presente forse come duplicato di `cortinovis-introleastsquares` (che è citato in report.tex:47)
- `goodfellow2016deep` (rr 95-101) — non citato
- `rahimi2007random` (rr 103-109) — non citato

**Fix**: rimuoverle (oppure mantenere ma documentare il perché). LaTeX non le include nel PDF se non `\cite`ed, ma fanno rumore.

---

## STILE (residui)

### ST-1 — Cap 1 r24: "is a machine learning model" filler

**File**: `progetto/report/1_introduction/report.tex:24`. Riformulare: "is a single-hidden-layer neural network with fixed random hidden weights and a linear output layer trained by solving a regularised least-squares problem." Marginale.

### ST-2 — Cap 1 r84: struttura "always... but if..." (falso bilanciamento)

**File**: `progetto/report/1_introduction/report.tex:84`. Riformulare causalmente: "The LS problem admits a unique solution iff X has full column rank; otherwise solutions exist but are not unique."

### ST-3 — Cap 2 r30: "pushed further toward zero" / "left almost free" antropomorfi

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:30`. Sostituire con descrizione neutra: "The weights assign larger penalties to components with |wᵢ| ≪ 1 (suppressing them) and smaller penalties to components with |wᵢ| ≫ 1."

### ST-4 — Cap 2 r173: "force every weight onto the threshold and slow the first iterations down" colloquiale

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:173`. Sostituire: "A cold start w₀=0 places all weights at εthr, delaying the first iterations." Più formale.

### ST-5 — Cap 2 r246: "However, ... typically small" falsa compensazione

**File**: `progetto/report/2_algo_1_IRLS/algo1.tex:246`. Riformulare diretto: "The per-iteration cost is O(H³) for Cholesky; total complexity is O(k·H³) with k ≈ 50–100 typical."

### ST-6 — Cap 3 r126: "amortizes over a few cheap Cholesky solves" verbo figurato

**File**: `progetto/report/3_algo_2_DSM/chapter3.tex:126`. CLAUDE.md vieta "amortise" figurato. Sostituire: "the O(MH²) precomputation cost is paid back over several cheap Cholesky solves" oppure "is offset by".

### ST-7 — Cap 4 r126 (presumibile, da verificare): stesso "amortizes" + "inducing sparsity"

**File**: `progetto/report/4_algo_comparison/comparison.tex` paragrafo costo IRLS. Riformulazione analoga al precedente. Verificare anche l'imprecisione "IRLS induces sparsity" → IRLS converge ad iterate sparse per via della majorization (non "induce").

### ST-8 — Code `data_generation.py:40,45,85`: magic numbers `1e-12`, `max_iter=100000`, `tol=1e-12` non documentati

Aggiungere nomi simbolici o commenti inline (`# tol per evitare divisione per zero su colonne nulle`, etc.). Stile non bloccante.

---

## Cap 5 / Cap 6 / Appendici — nessun finding

Il sweep `f69bf47` (19 fix cap 5) + `56d58da` (1 fix cap 6) + il lavoro su appendici di sessioni precedenti hanno reso questi capitoli puliti rispetto ai pattern LLM:

- Em-dash `---` solo dove tecnicamente necessari (tabelle/separatori, non come inciso decorativo)
- Niente "essentially/clearly/naturally/dramatically/decisively/particularly/genuinely/actually"
- Niente "amortise"/"leverage"/"unlock"/"drop-in"
- Niente "Importantly,/Crucially,/Notably,/It is worth noting that"
- Numeri anchor (0.46 california, 4.8e-5 synthetic, "ten significant digits") compaiono come anchor narrativo unico, non ripetuti verbatim cross-paragrafo
- Caption tabelle/figure non duplicano i paragrafi precedenti
- App. C/D supportano cap 3 senza ripetere statement
- App. A ha il disclaimer "abstract sensitivity" visibile

**Cross-check numerico**:
- f* reference params (k_max=3000, ε_thr=10⁻¹⁴) coerenti tra §5.6 testo, §5.6 caption, e `experiment_real_data.py:124-126`
- Clarabel ↔ IRLS agreement: "ten significant digits" coerente in §5.6 (riga 218), §5.6 (riga 768), §6 (riga 39)
- california: 340× ratio coerente tra cap 5 (riga 844) e cap 6 (riga 27)

---

## Cose già verificate e OK

Non riapro questi punti — sono stati controllati e sono a posto:

- Brace balance `{`/`}` e `\begin/\end` per tutti i .tex toccati (1040/1040 in results.tex, 599/599 in chapter3.tex)
- Compilazione `latexmk -pdf -bibtex`: 56 pp, no undefined refs, no errori semantici (solo overfull hbox cosmetici già presenti pre-sweep)
- Theorem 3.1 (cap 3): rate corretta vs d'Antonio-Frangioni 2009 eq. (3.17), iter inflation 400× con γ_min=0.05
- Appendice C/D (derivazioni Polyak + γ*) collegate correttamente al cap 3
- Test code: 53/53 passano
- `progetto/code/src/` codice principale: type hints, named constants K1-K14 già applicati

---

## Checklist esecuzione fix

Suggested order (CRITICI prima, poi IMPORTANTI, poi STILE, poi bib + code stile a chiudere):

- [ ] CR-1: fix `\ref{alg:dsm}` → `\ref{chap:dsm}` in `report.tex:124`
- [ ] CR-2: riscrivere "As expressed before" in `comparison.tex:41`
- [ ] CR-3: allineare formula rate in `comparison.tex:35` a `√(k+1)` come `chapter3.tex:155`
- [ ] CR-4: cambiare `rho: float = 0.95` → `0.7` in `deflected_subgradient.py:60`
- [ ] IM-1: precisare ipotesi Huang in `report.tex:27`
- [ ] IM-3: aggiungere step algebrici majorization in `algo1.tex:14`
- [ ] IM-4: qualificare claim NSP non valido in `algo1.tex:223`
- [ ] IM-5: aggiungere `[Slide 8 Frangioni]` in `comparison.tex:33`
- [ ] IM-6: fix `brannlund` key vs year in `references.bib:154`
- [ ] IM-7: rimuovere 3 orphan refs in `references.bib` (cortinovis-leastsquares, goodfellow2016deep, rahimi2007random)
- [ ] ST-1 → ST-8: fix di stile minori
- [ ] Recompile + verify no new warnings + git diff
- [ ] Atomic commits per gruppo, push a main

**Tempo stimato totale**: ~45-60 min per tutti i fix + rebuild PDF + commit/push.

---

## Note per l'orale (non in scope per questa review, ma da tenere a mente)

- T15/T16 della review precedente (verifica numeri Slide Frangioni + Theorem/Lemma in dantonio2009) restano pendenti — non posso verificarli senza accesso alle slide originali. Da fare manualmente con il prof in possesso del materiale.
- Cover letter risubmission: mapping "§5.3.3 (your reading) → §5.4.5 + §5.7 (current version)" come già discusso.
- Discrepanza `DSM_RHO_OLD = 0.9` (as-submitted) vs `0.7` (revised): se il prof chiede perché abbiamo cambiato, ricordarsi che la giustificazione è in §5.4.4 (sweep ρ ∈ {0.3,...,0.95} mostra ρ=0.7 within factor-2 best on all three instances).
