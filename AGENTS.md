# CM 646AA — Progetto ML-25 (ELM + LASSO)

Progetto: **ELM con regolarizzazione LASSO**, due algoritmi:
- **A1 — IRLS** (Iteratively Reweighted Least Squares) per la riformulazione liscia
- **A2 — Deflected Subgradient Method** sul problema non liscio

Docenti: Federico Poloni (numerical linear algebra) + Antonio Frangioni (optimization). Ogni mail va a entrambi.

## Fonti autoritative (in ordine di precedenza)

1. `progetto/comando.pdf` — regole d'esame, struttura del report, criteri di valutazione
2. `progetto/cm_project/` — testo del track ML-25 (`project_text.png`, `theory/`)
3. `progetto/manuale/` — riferimenti del corso
4. `Lessons Numerical Linear Algebra/` e `Lessons Optimization/` — slide ufficiali, citabili per teorema/pagina
5. Vault Obsidian (`~/Documents/ClaudeMemory/ClaudeMemory`) — collegamenti e note del corso
6. Letteratura esterna (Boyd&Vandenberghe, Nocedal&Wright, Bertsekas, ecc.) — solo se le slide non coprono il punto

**Mai** affermare un risultato teorico senza prima verificarlo in (1)–(4). Se la fonte è esterna, indicarlo esplicitamente.

## Layout del repo

- `progetto/code/src/` — implementazione (`elm.py`, `irls.py`, `deflected_subgradient.py`, `lasso_utils.py`, `linear_solvers.py`, `data_generation.py`)
- `progetto/code/experiments/` e `progetto/code/results/` — script di esperimenti e output
- `progetto/code/test_basic.py` — test
- `progetto/report/` — report LaTeX (`main.tex` + sezioni numerate `1_introduction` … `6_conclusions`, `references.bib`)

## Regole sul report (da `comando.pdf` §3–§4)

**Struttura attesa.** §4.2 setup problema/algoritmi + scelte motivate; §4.3 proprietà attese (convergenza, rate, complessità); §4.4 codice; §4.5 setup sperimentale; §4.6 plot e commenti.

**Ogni teorema/lemma/corollario citato deve riportare fonte + numero (teorema/pagina/slide).** Forma preferita: `[Slides Opt 7, Thm 3.2]` o `[Boyd&Vandenberghe 2004, p. 466]`. Mai citazioni vaghe del tipo "è noto che…".

**Verifica esplicita delle ipotesi.** Per ogni risultato di convergenza usato su IRLS o sul deflected subgradient: elencare le ipotesi (convessità, Lipschitzianità, compattezza, differenziabilità, …) e dimostrare *sul problema specifico ELM+LASSO* perché valgono o non valgono. Se non valgono, discutere il risultato "più vicino" applicabile e perché.

**Definizioni e statement di teoremi possono (e dovrebbero) essere riportati verbatim**, con citazione. Tutto il resto deve essere riformulato con parole proprie — copia anche mascherata da slides/libri/web/altri report = plagio (§3 di `comando.pdf`).

**Stay focused (§4.1).** Niente paragrafi su perché LASSO sia utile in ML, niente discussione di applicazioni pratiche, niente filler. Il valore aggiunto è la qualità matematica, non la lunghezza. Il pubblico target conosce già il corso.

**Plot (§4.6).** Scala logaritmica per residui/errori. Ogni plot ha uno scopo dichiarato (rate di convergenza, scaling CPU, gap, …). No tabelle a 7pt o 10 pagine di mini-plot.

## Regole sul codice (§4.4)

- Gli algoritmi *core* (IRLS, deflected subgradient, eventuale solver lineare custom) vanno **implementati a mano**, non chiamati come black-box di libreria.
- È lecito (e desiderato) confrontarsi con riferimenti off-the-shelf (`scikit-learn` Lasso, `cvxpy`, `scipy.optimize`, `numpy.linalg.lstsq`) — usati come **oracolo di correttezza** o per benchmark, non come implementazione del progetto.
- Sotto-step numerici (norma, prodotto matriciale, fattorizzazione QR/Chol per il sotto-problema dei minimi quadrati pesati di IRLS) possono usare `numpy`/`scipy`.
- Niente notebook monolitico: codice in moduli separati, dati in file dedicati o generati on-the-fly con seed riproducibile.
- Test su un range realistico di taglie/sparsità — non solo 10×10.

## Workflow consigliato

1. **Teoria prima del codice.** Completare §4.2–§4.3 del report (problema, algoritmi, ipotesi, convergenza attesa) **prima** di investire tempo serio in implementazione. Lo richiede esplicitamente §2 di `comando.pdf`.
2. Submission incrementali al docente (PDF via mail a entrambi); usare i feedback loop.
3. Esperimenti solo dopo che il setup teorico è stabile.

## Strumenti Codex

- **Planning/orchestrazione:** `gsd` (vedi `/gsd-help`)
- **Token efficiency:** `caveman` per sessioni lunghe; `caveman-commit` per i commit
- **Scrittura report:** `humanizer` (rimuove pattern AI dal LaTeX prima della submission — i docenti dichiarano in §4.1 di saper riconoscere testo LLM)
- **Note teoriche cross-progetto:** Obsidian
- **Compilazione LaTeX:** da `progetto/report/`, `latexmk -pdf -bibtex main.tex` (verifica anche `.bbl`/`.bcf`)

## Divieti tassativi

- **No allucinazioni di esperimenti.** Mai riportare numeri, plot, tabelle relativi a run mai eseguite. Se un risultato non è in `progetto/code/results/` o riproducibile da uno script in `experiments/`, non va nel report.
- **No referenze non verificate.** Prima di aggiungere a `references.bib`: aprire la fonte, controllare numero di teorema/pagina, controllare che il claim citato corrisponda davvero. Vale anche per le slide del corso.
- **No copia mascherata.** Riformulare sempre, anche da slide del corso, eccetto definizioni/statement di teoremi (verbatim + citazione).
- **No commit di `.aux`/`.log`/`.fls`/`.fdb_latexmk`/`.bbl`** non necessari (controllare `.gitignore`); mai rendere pubblico il repo (§3 — disservizio agli studenti futuri).
- **No tuning congiunto** di iperparametri del modello (peso del termine LASSO, # neuroni ELM) e parametri algoritmici (stepsize, tolleranze). I parametri del modello vanno fissati: tutti gli algoritmi devono risolvere lo *stesso* problema matematico (§4.5).

## Quando sei in dubbio

Se una scelta non è coperta da `comando.pdf` o dal track ML-25, **chiedi all'utente** prima di inventare una convenzione. È preferibile a una decisione che poi richiede di rifare sezioni del report.

## Imported Claude Cowork project instructions

Corso di CM
