# Codebase walkthrough — Project 25 ML (ELM LASSO)

Questa cartella contiene la documentazione interna della code base, scritta
per chi deve leggere il codice per la prima volta (te o un collega) e capire
*cosa fa, perché, e come modificarlo*. Non sostituisce il report — il report
copre la matematica e i risultati per il professore; questi documenti
spiegano l'**implementazione** così come è scritta nei file `.py`.

## Indice

| File | Cosa contiene | Quando leggerlo |
|---|---|---|
| [`01-architecture.md`](01-architecture.md) | Mappa dei file, divisione delle responsabilità, come la matematica del report vive nel codice (objective, gradient, sub-gradient, KKT, mapping sklearn α=λ/M). | Per orientarsi prima di toccare qualunque cosa. |
| [`02-algorithms.md`](02-algorithms.md) | IRLS e SGPTL spiegati riga per riga, con i riferimenti al report. La parte SGPTL include la storia del bug γ→0, il vincolo β_i = min(β, γ_i), e perché il safeguard iteration-count è necessario. | Per capire *come* gli algoritmi sono implementati e *perché* si è scelto questo o quello. |
| [`03-experiments-and-tests.md`](03-experiments-and-tests.md) | Cosa misura ogni esperimento (5 script, inclusa la validazione real-data su diabetes / California), quali figure / tabelle produce, come è organizzata la test suite, come riprodurre i risultati del report. | Per ri-eseguire o estendere gli esperimenti, e per aggiungere nuovi test. |

## Quickstart 5 minuti

```bash
cd progetto/code
python -m pytest tests/                       # 53 test, ~2.4 s
python experiments/experiment_convergence.py  # genera 3 figure in results/figures/
python experiments/experiment_comparison.py   # 1 figura + 1 tabella CSV
python experiments/experiment_params.py       # 2 figure
python experiments/experiment_scalability.py  # 1 figura + 1 tabella
python experiments/experiment_real_data.py    # 1 figura + 1 tabella su dataset reali
```

Le figure finiscono in `results/figures/*.pdf` e le tabelle in
`results/tables/*.csv`. Per ricompilare il report dopo aver rigenerato le
figure:

```bash
cp results/figures/*.pdf ../report/images/
cd ../report && latexmk -pdf main.tex
```

## Dipendenze

Python 3.12 con `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pytest`.
Nessuna libreria di ottimizzazione esterna: IRLS e SGPTL sono scritti da
zero, sklearn è usato solo come riferimento per `f^*` (vedi
[`01-architecture.md`](01-architecture.md), sezione "Reference solver").

## Convenzioni notazionali

I doc usano la stessa notazione del report:

- `H` = numero di neuroni del layer nascosto (= numero di feature dopo
  l'attivazione, = `n` nel codice di alcune funzioni).
- `M` = numero di campioni di training (= `m` nel codice).
- `X` ∈ ℝ^{M×H} = matrice delle attivazioni del hidden layer.
- `y` ∈ ℝ^M = target.
- `w` ∈ ℝ^H = pesi del layer di output, ciò che gli algoritmi cercano.
- `λ` = `λ_LASSO` = parametro di regolarizzazione.
- `f(w) = ½‖Xw − y‖² + λ‖w‖₁` = obiettivo (riga 33 di `lasso_utils.py`).
- `f*` = valore ottimo, calcolato con `sklearn.linear_model.Lasso`.

## Struttura della code base

```
progetto/code/
├── src/                          # Implementazione degli algoritmi
│   ├── __init__.py               # Re-esporta API pubblica
│   ├── lasso_utils.py            # f, gradiente, sub-gradiente, check KKT
│   ├── linear_solvers.py         # Cholesky e CG per Q w = b
│   ├── irls.py                   # Algorithm A1 (Cap. 2 del report)
│   ├── deflected_subgradient.py  # Algorithm A2 SGPTL (Cap. 3)
│   ├── elm.py                    # Modello ELM completo
│   └── data_generation.py        # Problemi sintetici + reference f*
│
├── experiments/                  # Script che producono figure + tabelle
│   ├── _plot_style.py            # Stile matplotlib condiviso
│   ├── experiment_convergence.py # Cap. 5.2 — gap vs iterazione/CPU
│   ├── experiment_params.py      # Cap. 5.3 — sensibilità eps_thr/lambda/delta_0/rho
│   ├── experiment_comparison.py  # Cap. 5.4-5.6 — quality, iters-to-eps
│   ├── experiment_scalability.py # Cap. 5.5 — scaling con H
│   └── experiment_real_data.py   # Cap. 5.7 — diabetes + California housing
│
├── tests/                        # Pytest suite (53 test)
│   ├── conftest.py               # Fixture condivise
│   ├── test_lasso_utils.py
│   ├── test_linear_solvers.py
│   ├── test_irls.py
│   ├── test_dsm.py
│   ├── test_elm.py
│   └── test_data_generation.py
│
├── results/                      # Output degli esperimenti (versionato)
│   ├── figures/   *.pdf
│   └── tables/    *.csv
│
├── utils/                        # Questa cartella
│   ├── README.md                 # Tu sei qui
│   ├── 01-architecture.md
│   ├── 02-algorithms.md
│   └── 03-experiments-and-tests.md
│
├── pytest.ini                    # Config pytest
├── test_basic.py                 # Smoke test legacy (kept for reference)
└── README.md                     # README di alto livello del codice
```

## Stato corrente (2026-05-07)

Ultimi cambiamenti rilevanti, in ordine cronologico inverso:

1. **Validation real-data** (`experiment_real_data.py`, §5.7 del report) —
   Pipeline ELM + LASSO su `diabetes` e `california_housing` di sklearn.
   IRLS matcha la precisione di sklearn-Lasso entro 100 iterazioni;
   SGPTL converge su California ($M\gg H$) ma stalla su diabetes
   ($M\sim H$, $\text{cond}\sim10^{6}$).
2. **Correzione algoritmica post-merge** (vedi `progetto/CHANGELOG.md`) —
   Quattro bug corretti in `src/`:
   - IRLS coefficient `2λD_k → λD_k` (il punto fisso ora soddisfa il KKT
     di λ, non di 2λ);
   - sklearn α: `λ/(2m) → λ/m` in `data_generation.py` (allineato a §5.1
     del report);
   - SGPTL: `_optimal_gamma` ora ritorna 1 quando `‖d_prev‖² ≈ 0`,
     `β_i = min(β, γ_i)` rispetta il vincolo del report §3.2, e il
     fallback iteration-count (`R_iter = max(i_max/100, 50)`) di §5.1
     ora è effettivamente implementato e scatta ad ogni iterazione;
   - Tutti gli script in `experiments/` ora passano l'OLS warm start a
     SGPTL (era un cold start zeros).
3. **Capitoli 5 e 6 del report riscritti** con i numeri reali dopo i
   fix. La storia qualitativa è invariata (IRLS lineare batte SGPTL
   sublineare), ma i valori specifici dei gap erano artefatti del bug
   `f_star` e sono stati sostituiti con le misure corrette.

`progetto/CHANGELOG.md` contiene il dettaglio completo dei commit di
questo ciclo.

## Per chi è questo documento

Lo stile è: *spiega abbastanza perché un lettore con background
matematico (corso CM) ma non familiare con questa code base possa, in
mezza giornata di lettura, capire dove vivere ogni pezzo, e in un'altra
mezza giornata estendere o modificare con sicurezza*. Non è un
auto-tutorial sui subgradient methods — il report copre quella parte.
