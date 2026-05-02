# Codebase walkthrough — Project 25 ML (ELM LASSO)

Questa cartella contiene la documentazione interna della code base, scritta
per chi deve leggere il codice per la prima volta (te o un collega) e capire
*cosa fa, perché, e come modificarlo*. Non sostituisce il report — il report
copre la matematica e i risultati per il professore; questi documenti
spiegano l'**implementazione** così come è scritta nei file `.py`.

## Indice

| File | Cosa contiene | Quando leggerlo |
|---|---|---|
| [`01-architecture.md`](01-architecture.md) | Mappa dei file, divisione delle responsabilità, come la matematica del report vive nel codice (objective, gradient, sub-gradient, KKT, mapping sklearn). | Per orientarsi prima di toccare qualunque cosa. |
| [`02-algorithms.md`](02-algorithms.md) | IRLS e SGPTL spiegati riga per riga, con i riferimenti al report. La parte SGPTL include la storia del bug γ→0 e perché il safeguard di pazienza è necessario. | Per capire *come* gli algoritmi sono implementati e *perché* si è scelto questo o quello. |
| [`03-experiments-and-tests.md`](03-experiments-and-tests.md) | Cosa misura ogni esperimento, quali figure / tabelle produce, come è organizzata la test suite, come riprodurre i risultati del report. | Per ri-eseguire o estendere gli esperimenti, e per aggiungere nuovi test. |

## Quickstart 5 minuti

```bash
cd progetto/code
python -m pytest tests/                       # 53 test, ~2.3 s
python experiments/experiment_convergence.py  # genera 3 figure in results/figures/
python experiments/experiment_comparison.py   # 1 figura + 1 tabella CSV
python experiments/experiment_params.py       # 2 figure
python experiments/experiment_scalability.py  # 1 figura + 1 tabella
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
│   ├── experiment_convergence.py
│   ├── experiment_comparison.py
│   ├── experiment_params.py
│   └── experiment_scalability.py
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

## Per chi è questo documento

Lo stile è: *spiega abbastanza perché un lettore con background
matematico (corso CM) ma non familiare con questa code base possa, in
mezza giornata di lettura, capire dove vivere ogni pezzo, e in un'altra
mezza giornata estendere o modificare con sicurezza*. Non è un
auto-tutorial sui subgradient methods — il report copre quella parte.
