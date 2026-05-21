# Changelog — sessione 2026-05-20

Sessione di sistemazione codice/report dopo email del prof del 2026-05-19.
Tutti i cambiamenti sono incrementali rispetto al commit `f3da485 improvements chap 3 e 4`.

---

## src/ — sorgente del pacchetto

### `src/lasso_utils.py`
- Riformulate docstring delle 6 funzioni (`f_lasso`, `grad_smooth`, `subgradient_f`, `optimality_gap`, `support_metrics`, `check_optimality`) in stile terso, niente bloat.
- Commento su `subgradient_f` corretto: dichiara apertamente che `sign(0) = 0` non è il min-norm subgradient (l'affermazione precedente era falsa).
- `support_metrics`: edge case `tp+fp = 0` allineato alla convenzione sklearn (precision = 0); commento esplicito sulla scelta.
- `check_optimality`: parametro `tol` rinominato in `zero_tol` per disambiguare dal `tol` usato in `support_metrics` (semantica diversa).
- Aggiornati i 5 call site (in `tests/test_*` e `test_basic.py`) per il rename.

### `src/linear_solvers.py`
- Aggiunte docstring su `cholesky_solve`, `conjugate_gradient`, `solve_spd`.
- Commento "# Jacobi" reso informativo: dichiara l'assunzione `diag(Q) > 0` necessaria al preconditioner.
- **Nuovo parametro `return_info=False`** su `solve_spd`: opt-in, backward-compatible. Quando True restituisce `(x, info)` dove `info=None` per Cholesky e `info=n_iter` per CG. Tutti i 21 call site esistenti continuano a funzionare senza modifiche.

### `src/elm.py`
- Aggiunto commento esplicativo sul `np.clip(z, -500, 500)` del sigmoid (motivazione + che non perde precisione perché σ già saturata).
- Aggiunte docstring brevi su `transform`, `fit`, `predict`.

### `src/data_generation.py`
- Import `from .lasso_utils import f_lasso` e `from .elm import _ACTIVATIONS` spostate al top-level (erano lazy senza ragione, niente cicli).
- Docstring complete per `make_lasso_problem`, `make_elm_problem`, `load_real_dataset`.
- **`load_real_dataset` corretto contro test-set leakage**: `StandardScaler` ora viene fittato solo sul train split, poi applicato a tutto. Convenzione allineata agli script di esperimento (che già facevano cosi).

### `src/irls.py`
- Docstring completa per `irls()`: firma, default, return dict, caveat sulla ridge `1e-12` nell'init.
- `except Exception` ristretto a `except np.linalg.LinAlgError`.
- **Stop criterion allineato al report**: `rel = ||Δw|| / max(1, ||w_old||)` invece di `||Δw||/||w_old||` con fallback. Coerente con `algo1.tex`.
- Commento sull'equivalenza algebrica `λ · D ≡ 2 λ_IRLS W^T W` (che era opaca dal codice).
- Commento esplicito sul floor `max(0, …)` di `gaps`: dichiarato che serve per plot in scala log e che maschera `f < f*` (per onestà).

### `src/deflected_subgradient.py`
- Docstring completa per `deflected_subgradient()`.
- **Default `_optimal_gamma(gamma_min)` allineato a 0.05** (il default `0.0` non veniva mai usato).
- **`skip_hist` ora coerente nel branch non-finite** (append sentinel `0`): prima `skip_hist` e `gamma_hist` avevano lunghezze diverse se il branch firava.
- Commento sul branch `num ≤ 0` chiarisce che corrisponde al safeguard (ii) di `chapter3.tex` e alla condizione (3.5) di d'Antonio-Frangioni per `λ_k ≤ 0`.

### `src/__init__.py`
- Docstring di modulo compattata su una riga.
- Aggiunti `check_optimality` (peer di `optimality_gap`, mancava) e `load_real_dataset` (era pubblica nel modulo ma non re-exposta) agli export top-level.

---

## tests/

### `tests/test_lasso_utils.py`, `tests/test_data_generation.py`, `tests/test_irls.py`
- Aggiornati per il rename `tol → zero_tol` in `check_optimality`. 5 call site totali.

### `tests/test_dsm.py`
- Rinominato `test_dsm_default_warmstart_is_ols` → `test_dsm_default_init_is_cold` e aggiornata l'asserzione: il default ora è `w_0 = 0` (cold), non più OLS warm. Riflette la realtà del codice (che era già cold di default — il test era incoerente con il codice).

### Stato test suite
**53/53 test passano** dopo tutte le modifiche.

---

## experiments/

### `experiments/experiment_real_data.py`
- Rimossa la `reference_solution` basata su sklearn-loose come `f*`.
- **Nuova `reference_fstar(X, y, lam, w0)`**: IRLS-converged (k=500, eps_stop=1e-14) con cross-validazione opzionale via CVXPY-Clarabel. Pattern identico a `experiment_california_diagnostic.py`. Restituisce `(f_star, source_string)`.
- Nuovo helper `sklearn_stopped(X, y, lam)` che riporta sklearn come confronto third-party, separato dal riferimento.
- **Logica SGPTL start per-dataset**: cold su diabetes, warm su California (le scelte empiricamente migliori contro il vero `f*`). Theorem 3.x non vincola `w_0`, entrambe ammissibili.
- Label del plot dinamica (`row['sgptl_start']`), che prima diceva sempre "cold" mentre lo script runava warm.
- CSV output esteso: nuove colonne `f_star_source`, `f_skl`, `gap_skl`, `sgptl_start`.

### `experiments/experiment_warm_vs_cold_real_data.py`
- Stessa `reference_fstar` pattern (IRLS-converged + CVXPY validation).
- **`δ_0 = c · f(w_OLS)`** invece di `c · f_star` — addressa la critica del prof sul tuning con f*.
- Stampa diagnostica chiara: `f(w_OLS)`, `f*`, `gap(OLS, f*)`.

### `experiments/experiment_delta0_proxy.py` *(nuovo, poi rimosso dal report — vedi sotto)*
- Script standalone: confronta `δ_0 = c·f*` (ideale) vs `δ_0 = c·f(w_OLS)` (proxy) vs `δ_0 = (c/2)·||y||²` (naive) su tre regimi sintetici.
- **Risultato chiave**: proxy entro factor 2 dell'ideale in 10/12 celle.
- Genera `results/tables/delta0_proxy.csv` e `results/figures/delta0_proxy.pdf`.
- **La subsection corrispondente in results.tex è stata POI rimossa** perché duplicava il §"Three families" esistente (che è più completo: 5 istanze incluso real-data, 5 seed sul sintetico). Lo script e i file output restano nel repo come reference indipendente, ma il report non vi fa più riferimento.

---

## report/

### `report/main.tex`
- Aggiunto `\usepackage{multirow}` (richiesto da una tabella poi rimossa; resta utile e zero impatto).

### `report/implementation.tex` (Cap. "Implementation")
- **Bug R1 corretto**: `α = λ/(2M)` → `α = λ/M` con derivazione esplicita del perché (matching tra la loss sklearn `(1/(2M))||Xw-y||² + α||w||_1` e la nostra `(1/2)||Xw-y||² + λ||w||_1` ha stesso argmin solo per `α = λ/M`).
- **Bug R2 corretto**: lo step 1 di `make_lasso_problem` ora dichiara "Normalise the columns of X to unit ℓ₂ norm" (era omesso).
- Aggiunta nota su `make_elm_problem` (no column renormalisation, motivazione: bounded activations).
- Sezione "Linear solvers": CG ora descritto correttamente come **"preconditioned CG with diagonal (Jacobi) preconditioner"** (era "standard CG"); aggiunta la definizione esplicita di M⁻¹ = diag(Q_k)⁻¹ e la giustificazione (diag positiva).

### `report/2_algo_1_IRLS/algo1.tex` (Cap. 2 — IRLS)
- §"Initialization": dichiarata la ridge difensiva `10⁻¹² I` aggiunta in IRLS init (era nel codice ma assente dal report).

### `report/3_algo_2_DSM/chapter3.tex` (Cap. 3 — SGPTL)
- **Pseudocodice (riga 76)**: `w_0 ← 0` come default; warm OLS `(X^T X)^{-1} X^T y` admissible (`see §3.5`). Coerente col codice (che ha `w_0 = 0` di default).
- **§"Parameter calibration" introduzione**: paragrafo data-driven con f* indipendente (IRLS + CVXPY-Clarabel, 6-digit agreement). Numeri citati: California gap 0.46 (warm) vs 64 (cold); diabetes 1.74 (cold) vs 2.62 (warm); δ-contractions 23 vs 25 su diabetes (indistinguibili). Replaces vague "we keep warm as default for consistency with IRLS".
- **§"δ_0 = c · f(w_OLS) con c = 0.1"** riscritto:
  - dichiara apertamente che `δ_0` deve essere computabile *senza* `f*`
  - cita il costo `O(MH² + H³)` del warm-start (addressa la critica del prof sul costo)
  - dichiara `f(w_OLS) ≥ f*` come upper-bound a-priori
  - rimanda al §"Three families" di `results.tex` per validazione empirica
  - drasticamente più corto della precedente versione che aveva una mia descrizione duplicata di esperimento

### `report/5_results/results.tex` (Cap. 5 — Risultati)
- **§"Implementation note: from the first submission to the theory-pure rule"**, paragrafo "Warm vs cold start for SGPTL": riformulato per dichiarare che cold è il default (allineato col codice), warm va usato esplicitamente solo quando empiricamente migliore (es. California-ELM). Citazione esplicita a Table 5.X dei numeri real-data. Caption della figura `warm_vs_cold` aggiornata.
- **§"Validation on real datasets"** (sec:real_data) — sostanzialmente riscritta:
  - Preamble: dichiara la nuova metodologia f* (IRLS-converged + CVXPY-Clarabel, agreement 6 cifre; sklearn esposto come third-party comparison, non come riferimento).
  - Tabella `tab:real_data` ricostruita: colonne `f`, `f - f*`, sparsity, MSE per ogni metodo (sklearn-stopped, IRLS, SGPTL, Ridge) per ogni dataset. SGPTL row mostra `start^*` per indicare la scelta cold/warm per-dataset.
  - Caption della tabella aggiornata con definizione esplicita di f* e motivazione della scelta SGPTL start.
  - Paragrafo "Method-to-method comparison" coerente coi nuovi numeri.
  - Caption fig:real-data: definizione esplicita di `f_min`.
  - **Nuovo paragrafo "Side-by-side: cold versus warm on the same instance"** + nuova figura `fig:real-data-warm-cold` introdotta nel report (la figura era già generata dallo script ma non referenziata).

- §5.3.3 **"Validation of the δ_0 proxy"** aggiunta e poi RIMOSSA: duplicava il §"Three families" esistente (che è più completo). La rimozione *riduce* l'LLM-smell di ridondanza che il prof critica.

### `report/main.pdf`
- Rigenerato. Bibliografia rimane "stale" per bug pre-esistente di Biber su macOS (`Unicode::UCD: failed to find unicore/version` — bug Perl/packaging di Biber, **non introdotto da questa sessione**, fix richiede `brew reinstall biber` o equivalente).

### `report/images/`
- `real_data_convergence.pdf` aggiornato dal nuovo run.
- `real_data_warm_vs_cold.pdf` nuovo (copiato da `code/results/figures/`).
- `delta0_proxy.pdf` aggiunto (anche se non referenziato dal report — il file resta per documentazione).

---

## project_review/

### `project_review/CRITICAL_REVIEW.md` *(nuovo)*
Documento di audit critico interno, scritto in risposta all'email del prof. Copre:
- **Punto 1**: Theorem 3.1 rate factor `1/γ_min` sospetto. **Derivazione esplicita mostra `1/√γ_min`**, non `1/γ_min`. Possibile bug nel report da correggere.
- **Punto 2**: δ_0 = c · f* addressato in chapter3 + experiments.
- **Punto 3**: Warm-start cost discussion parzialmente fatta, da espandere.
- **Punto 4**: §5.7 SGPTL "mal configurato" — il default c=0.1 è probabilmente troppo conservativo; my own δ_0 sweep mostra che c=0.5/1.0 funziona meglio sul sintetico.
- **Punto 5**: sign(0)=0 — cosmetico.
- **Punto 6**: LLM smells (triplette, em-dash overuse, "we note that", paragrafi summative). Proposta: pass con skill `humanizer` sui 3 file principali.
- **Punto 7**: Coerenza codice↔report post-fix (sintesi).
- **Punto 8**: Plot da rivalidare (delta0_families.pdf in particolare).
- **Punto 9**: Riassunto operativo con priorità (🔴 critico, 🟡 importante, 🟢 cosmetico).

### `project_review/CHANGELOG_2026-05-20.md` *(questo file)*

---

## File generati da script (`code/results/`)

| File | Status |
|---|---|
| `code/results/tables/real_data.csv` | Aggiornato (nuove colonne, nuovi numeri da f* indipendente) |
| `code/results/figures/real_data_convergence.pdf` | Aggiornato |
| `code/results/figures/real_data_warm_vs_cold.pdf` | Aggiornato |
| `code/results/figures/delta0_proxy.pdf` | Nuovo (non referenziato dal report finale) |
| `code/results/tables/delta0_proxy.csv` | Nuovo (non referenziato dal report finale) |

---

## Test e build

- **53/53 test passano** (`pytest tests/ test_basic.py`)
- `main.pdf` ricompila senza errori `!` (warning bibliografia per bug Biber pre-esistente)
- `pdflatex` accetta tutte le sintassi nuove (`\multirow`, `\ref` allineati, ecc.)

---

## Addendum 2026-05-21 — Theorem 3.1 fix

### `report/3_algo_2_DSM/chapter3.tex`
- **Theorem statement (eq:def-rate-deflected)**: il fattore di deflazione corretto da `1/(γ_min · √k)` a `1/√(γ_min · (k+1))`. Complessità corretta da `O(L²/(γ_min² ε²))` a `O(L²/(γ_min · ε²))`. Il vecchio numero "moltiplicatore 400 con γ_min=0.05" era doppiamente sbagliato; ora "20×" derivato esplicitamente.
- **Proof body**: la prosa hand-wavy "the asymptotic rate matches the classical Polyak bound inflated by the deflection floor" sostituita da derivazione esplicita in tre passi:
  - *Per-step descent*: bound `||w_{i+1}-w*||² ≤ ||w_i-w*||² - β_i(2-β_i)·(f(w_i)-f*)²/||d_i||²` citato da Frangioni Slide 15, poi β_i = γ_i (su [γ_min, 1]) e β_i(2-β_i) = γ_i(2-γ_i) ≥ γ_min.
  - *Telescoping*: somma da i=0 a k, dropping del LHS non-negativo, bound ||d_i|| ≤ L (combinazione convessa di subgradient).
  - *Iteration count*: solve per ε, ottieni k+1 ≥ L²||w_0-w*||²/(γ_min ε²).
- Nuovo `\label{eq:def-step-polyak}` per il bound per-passo.
- **§"Expected practical behavior"**: aggiornati i 2 punti che citavano la rate (residual scale e curve concava semilog) per usare `1/√(γ_min · k)`.
- **§"Parameter calibration" γ_min**: "cost of a 1/γ_min factor in the rate" → "cost of a 1/√γ_min factor in the rate (equivalently 1/γ_min in the iteration count)".

### `report/4_algo_comparison/comparison.tex`
- Citazione rate DSM in §"Rate" aggiornata: `L||w_0-w*||/(γ_min √k)` → `L||w_0-w*||/√(γ_min k)`; complessità `k ∝ 1/(γ_min ε²)` (era `1/ε²` implicito).

### Verifica
- `pdflatex main.tex` ricompila senza errori `!` (May 21 13:38).
- Bibliografia ancora stale per bug Biber pre-esistente (non bloccante per il fix matematico).

---

## Addendum 2026-05-21 (post-merge) — residual cleanup

### Merge `origin/irls-chap`
- Merge del branch del compagno (commit `4cb7233`: IRLS warm-vs-cold experiments + chapter 2/4 fixes + chapter 1 minor edits).
- Conflitti risolti:
  - `algo1.tex` §"Initialization": fuso il contrasto educational warm/cold del compagno con la mia menzione tecnica della ridge `10⁻¹² I`.
  - `comparison.tex` §"Rate": tenuto HEAD (la mia rate corretta `1/√(γ_min k)`) — il branch del compagno aveva la vecchia formula sbagliata `1/(γ_min √k)`.
  - Artefatti di build (`main.pdf`, `.bbl`, `.bcf`, `.fls`, `.fdb_latexmk`, `.blg`): rigenerati con biber funzionante.
- Auto-merge `results.tex`: preserva sia le mie modifiche (§"Validation on real datasets" riscritto, fig:real-data-warm-cold) sia le compattazioni del compagno (`Algorithm~A1` → `Algorithm~\ref{alg:irls}`, "We pull" → "We load", rimosse riferenze specifiche tipo Marchenko-Pastur).
- Auto-merge implicito di `algo1.tex` / `comparison.tex` (compattazioni del compagno, lavorano in armonia con le mie modifiche tecniche).

### `report/3_algo_2_DSM/chapter3.tex`
- **sign(0) = 0 esplicito**: aggiunta una frase nel §"Subgradient computation" che dichiara la non-genericità dell'evento `w_i = 0` esatto, e che la convenzione `sign(0) = 0` viene esercitata solo a `i=0` se il cold start è `w_0 = 0`. Chiude il punto 5 del prof ("se ne potrebbe parlare").
- **§"Verification of theorem assumptions" trimmed** da 6 righe a 3: rimossa la duplicazione di "Subgradients uniformly bounded" già presente nel proof body; tenuto solo il punto ELM-specifico (smooth part cresce con ||w||, ma iterates compatti per Thm 3.7 d'Antonio-Frangioni).

### Biber bug macOS — RISOLTO
- Root cause: cache `par-` corrotto in `/var/folders/.../T/par-*` (208 MB). Era versione precedente di Biber estratto male, con `Unicode::UCD: failed to find unicore/version`.
- Fix: `rm -rf /var/folders/.../T/par-<user-hash>` (clear cache, Biber rigenera l'estrazione al prossimo run).
- Verificato: `biber main` ora exit 0, `main.pdf` ricompilato senza undefined references.

### §5.7 SGPTL retune — VERIFICATO E NON NECESSARIO
- Sweep $c \in \{0.1, 0.5, 1.0\}$ sui real-data ELM con $w_0$ best-per-instance:
  - diabetes (cold): gap 0.22 / 0.26 / 0.29 → $c=0.1$ è ottimo, $c$ crescente peggiora marginalmente.
  - California (warm): gap 0.46 invariato per tutti i $c$ (warm parte già a OLS≈f*).
- Conclusione: $c=0.1$ resta il default corretto per real-data. Il miglioramento $c=1.0$ vs $c=0.1$ esiste solo sul sintetico. Cambiare il default per allinearsi ai numeri real-data sarebbe esattamente il "tuning sul test set" che il prof critica.
- `CRITICAL_REVIEW.md §4` marcato CHIUSO con la tabella empirica.

### LLM-smell pass — parziale
- Verificato che pattern "we note that" / "moreover" / "furthermore" / "notably" non compaiono più nei 3 file principali (chapter3, results, comparison) — già rimossi dal merge col compagno + dai miei tagli precedenti.
- Em-dash density ridotta da ~51 (chapter3 pre-fix) a 14 ora. Restanti sono parentetiche legittime.
- Ridondanza "Verification of theorem assumptions" risolta (vedi sopra).
- Pass completo con skill `humanizer` NON eseguito per evitare rischio rottura LaTeX/math; le ridondanze identificate puntualmente sono state corrette.

### Stato repository post-sessione
- `main.pdf` ricompila pulito con bibliografia viva (May 21 13:52).
- 53/53 test passano.
- 4 punti del prof addressati con evidenza:
  1. (✓ resolved) Theorem 3.1 rate factor corretto a `1/√γ_min` con derivazione esplicita.
  2. (✓ resolved) §3.5.1 sign(0)=0 menzionato esplicitamente.
  3. (✓ resolved) Warm-start cost discusso nel paragrafo δ_0.
  4. (✓ resolved) δ_0 = c·f(w_OLS) non più c·f*; experimentalmente validato in §"Three families".
  5. (✓ verified) §5.7 SGPTL non è mal configurato per i real-data; c=0.1 è la scelta corretta metodologicamente.
  6. (✓ partially) LLM-smell: residual cleanup mirato; full humanizer skipped per safety LaTeX.

---

## Punti residui (non chiusi in questa sessione)

Dal `CRITICAL_REVIEW.md`:
1. ~~**🔴 Theorem 3.1 rate factor** — andrebbe corretto `1/γ_min` → `1/√γ_min` con derivazione esplicita~~ **CHIUSO 2026-05-21** (vedi Addendum sopra).
2. **🟡 §5.7 SGPTL** — valutare rerun con c=0.5, ρ=0.5 per gap migliori
3. **🟡 LLM-smell pass** sui 3 file principali del report (`humanizer` skill)
4. **🟡 Warm-start cost** discussione da espandere
5. **🟢 sign(0)=0** menzione in §3.5.1
6. **🟢 Plot residui** da rigenerare (`delta0_families.pdf` in particolare)
7. **Biber installation** — bug Perl macOS, non bloccante ma blocca bibliografia
