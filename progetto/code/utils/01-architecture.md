# 01 — Architettura

Questo documento mappa la code base modulo per modulo, spiega come la
matematica del report vive nei file, e descrive le scelte di design che
hanno conseguenze sul resto del codice.

## L'obiettivo, in codice

Il report definisce in equazione (1.3):

$$
f(\mathbf{w}) \;=\; \tfrac{1}{2} \,\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_{2}^{2}
\;+\; \lambda_{\text{LASSO}} \,\|\mathbf{w}\|_{1}.
$$

Nel codice questa è una funzione di tre righe:

```python
# src/lasso_utils.py
def f_lasso(X, y, w, lam):
    residual = X @ w - y
    return float(0.5 * np.dot(residual, residual) + lam * np.sum(np.abs(w)))
```

**Il fattore 1/2 è importante.** Una versione precedente del codice
calcolava `‖Xw − y‖² + λ‖w‖₁` (senza 1/2). Conseguenze a cascata: il
gradiente liscio sarebbe `2 Xᵀ(Xw − y)`, non `Xᵀ(Xw − y)`; le normali di
IRLS diventerebbero `(XᵀX + (λ/2) Wᵀ W) w = Xᵀy` invece di
`(XᵀX + λ Wᵀ W) w = Xᵀy`; e il mapping verso sklearn dovrebbe usare
`α_sklearn = λ/(2M)` invece di `λ/M`. Tutto il codice è oggi allineato
con l'**eq. 1.3 del report**, e la test suite verifica numericamente
questa coerenza.

## Mappa dei file `src/`

### `lasso_utils.py` — primitive matematiche

Cinque funzioni pubbliche.

| Funzione | Cosa calcola | Dove serve |
|---|---|---|
| `f_lasso(X, y, w, lam)` | `f(w)` | ovunque, è il valore monitorato |
| `grad_smooth(X, y, w)` | `Xᵀ(Xw − y)` | sub-gradiente, IRLS, KKT check |
| `subgradient_f(X, y, w, lam)` | `Xᵀ(Xw − y) + λ s`, con `sᵢ = sign(wᵢ)` (e `s = 0` quando `w = 0`, sub-gradiente di norma minima — report § 3.5.1) | passo di SGPTL |
| `optimality_gap(X, y, w, lam, f_star)` | `f(w) − f*`, può essere negativo se `w` è migliore della soluzione di sklearn | grafici di convergenza |
| `check_optimality(X, y, w, lam, tol)` | violazione massima della condizione KKT (1.4): `\|Xᵀ(Xw − y)_i\| ≤ λ` per `wᵢ = 0`, `Xᵀ(Xw − y)_i + λ \,\text{sgn}(wᵢ) = 0` per `wᵢ ≠ 0` | test |

### `linear_solvers.py` — solver SPD per `Qw = b`

IRLS risolve a ogni iterazione un sistema lineare SPD:

$$
\mathbf{Q}_{k} \,\mathbf{w}_{k+1} = \mathbf{X}^{\top} \mathbf{y},
\qquad
\mathbf{Q}_{k} = \mathbf{X}^{\top}\mathbf{X} + \lambda\, \mathbf{W}_{k}^{\top} \mathbf{W}_{k}.
$$

Due implementazioni:

- **`cholesky_solve(Q, b)`** — usa `scipy.linalg.cho_factor`/`cho_solve`.
  Costo $O(H^3/3)$, esatto a meno di errore floating-point. Default per
  IRLS perché sempre stabile su questi problemi (Q è SPD ben condizionata
  finché i pesi della diagonale non esplodono — vedi sotto).
- **`conjugate_gradient(Q, b, x0, tol, max_iter)`** — implementato da zero
  seguendo il classico algoritmo CG per matrici SPD. Costo $O(p H^2)$ con
  $p$ iterazioni CG. Utile quando $H$ è grande e $\mathbf{Q}_k$ ha molti
  autovalori clusterizzati.
- **`solve_spd(Q, b, method='cholesky')`** — interfaccia unificata
  chiamata da `irls.py` e da `deflected_subgradient.py` per il warm
  start.

### `irls.py` — Algorithm A1

Implementa l'algoritmo del Cap. 2 del report. Una sola funzione pubblica,
`irls(X, y, lam, ...)`. Vedi `02-algorithms.md` per la spiegazione passo
per passo.

### `deflected_subgradient.py` — Algorithm A2 (SGPTL)

Implementa il deflected subgradient method con target level del Cap. 3.
Funzione pubblica `deflected_subgradient(...)` più una helper privata
`_optimal_gamma(g, d_prev)`. Vedi `02-algorithms.md` per i dettagli (la
parte interessante riguarda il safeguard di pazienza che abbiamo dovuto
aggiungere per uscire dal deadlock γ→0).

### `elm.py` — il modello ELM completo

Classe `ELM` che incapsula:

1. La generazione del layer nascosto fissa: `W1 ∈ ℝ^{H×d}`, gaussiana,
   non aggiornata durante il training.
2. La trasformazione `X_raw → X_hid = σ(X_raw W1ᵀ)` con
   $\sigma\in\{\text{sigmoid}, \text{tanh}, \text{relu}\}$.
3. Il fit del layer di output: chiama `irls(...)` o
   `deflected_subgradient(...)` su `(X_hid, y, λ)` e salva `w`.
4. La predict.

Esempio:

```python
from src.elm import ELM
elm = ELM(d=10, p=80, activation="sigmoid", lam=0.05, random_state=2)
elm.fit(X_train, y_train, solver="irls", k_max=200)
y_hat = elm.predict(X_test)
print("sparsity:", elm.sparsity, "n_active:", elm.n_active)
```

Negli esperimenti del report la classe `ELM` non è usata direttamente:
si lavora sul livello inferiore (`make_lasso_problem` o
`make_elm_problem` → `irls(X, y, λ, ...)`) per misurare *il gap di
ottimizzazione*, che è la quantità di interesse del corso CM. La classe
esiste perché:

- I test la usano per verificare il pipeline end-to-end.
- È utile se domani il professore chiede di mostrare una predizione su
  un dataset reale.

### `data_generation.py` — problemi sintetici

Tre funzioni:

- **`make_lasso_problem(n, m, sparsity, ..., lam, random_state)`**
  costruisce un problema LASSO sintetico: vettore `w_true` sparso,
  matrice `X` con colonne di norma 1 (per controllare il
  condizionamento), `y = X w_true + rumore` con rumore gaussiano.
  Restituisce anche `w_star` (soluzione sklearn) e
  `f_star = f(w_star)`.

- **`make_elm_problem(d, p, m, ...)`** — costruisce un'istanza ELM
  completa: input grezzi `X_raw`, hidden weights `W1`, attivazioni
  `X_hid = σ(X_raw W1ᵀ)`, ground truth `w_true`, target `y` e (di
  nuovo) `f_star`. Restituisce sia `X_raw` che `X_hid` perché alcuni
  test verificano la consistenza tra i due.

- **`load_real_dataset(name, test_size, random_state)`** — placeholder
  per dataset reali sklearn (`diabetes`, `california`). Non usato
  nelle figure del report.

## Reference solver: come `f*` viene calcolato

Il professore (`comando.pdf` § 4.6) raccomanda di confrontare con
software off-the-shelf "to check correctness". Noi non abbiamo `f*` in
forma chiusa per ELM LASSO, quindi per ogni istanza calcoliamo un
riferimento ad alta precisione con
`sklearn.linear_model.Lasso(coordinate descent, tol=1e-12)` e definiamo

$$
f^{*} \;=\; f(\mathbf{w}_{\text{sklearn}}).
$$

**Mapping di α.** Sklearn minimizza un obiettivo con scaling diverso dal
nostro:

$$
L_{\text{sklearn}}(\mathbf{w}) =
\frac{1}{2 M} \,\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_{2}^{2}
+ \alpha \,\|\mathbf{w}\|_{1}.
$$

Moltiplicando $L_{\text{sklearn}}$ per $M$ otteniamo
$\tfrac{1}{2}\,\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^{2} + M\alpha\,\|\mathbf{w}\|_{1}$.
Per coincidere col nostro $f$ serve $M\alpha = \lambda$, quindi
$\alpha = \lambda / M$. Questo mapping è applicato in due punti
(`make_lasso_problem` riga 67 e `make_elm_problem` riga 137) e
verificato dal test `test_make_lasso_problem_fstar_is_local_minimum`.

Una versione precedente del codice usava `α = λ/(2M)` (corretto per
l'obiettivo *senza* il fattore 1/2). Dopo l'allineamento all'eq. 1.3 del
report, il mapping è `α = λ/M`. Sbagliarlo significa che `f*` non è il
vero ottimo, e tutti i grafici di gap diventano falsati.

## Mappa dei file di test (`tests/`)

| File | Cosa testa | Numero di test |
|---|---|---:|
| `test_lasso_utils.py` | `f`, gradiente analitico vs differenze finite, sub-gradiente di norma minima, KKT a `w_star` | 9 |
| `test_linear_solvers.py` | Cholesky residual, CG residual, agreement Cholesky↔CG, dispatch | 9 |
| `test_irls.py` | monotonia, KKT al convergere, sparsità per λ grande, warm start, scelta solver | 11 |
| `test_dsm.py` | `_optimal_gamma` whitebox, OLS warm start, record value monotono, sub-gradient inequality | 10 |
| `test_elm.py` | shape attivazioni, fit IRLS+predict, fit DSM+predict, errori (raise) | 9 |
| `test_data_generation.py` | shape, KKT a `w_star`, consistenza ELM | 5 |

Le **fixture in `conftest.py`** definiscono tre problemi di
riferimento:

```python
small_problem   # H=10,  M=50,  λ=0.1   (per i test veloci)
medium_problem  # H=50,  M=200, λ=0.05  (per i test di convergenza)
elm_problem     # H=80,  M=300, d=10    (per il pipeline ELM end-to-end)
```

Ogni fixture restituisce `LassoProblem(X, y, lam, w_true, w_star, f_star)`.

## Mappa dei file di esperimento (`experiments/`)

Ogni script è autocontenuto: configurazione in cima, una funzione
`run()`, salvataggio di figure e CSV in `results/`. Lo stile dei plot è
centralizzato in `_plot_style.py` (funzione `apply_style()` chiamata
all'inizio di ogni script, palette esposta come costanti di modulo).

| Script | Misura | Output |
|---|---|---|
| `experiment_convergence.py` | gap vs iter, gap vs tempo, oscillazione DSM | `convergence_vs_iter.pdf`, `convergence_vs_time.pdf`, `dsm_nonmonotone.pdf` |
| `experiment_comparison.py` | iterazioni e tempo per raggiungere `ε ∈ {10⁻¹,…,10⁻⁶}` | `comparison_irls_dsm.pdf`, `comparison_table.csv` |
| `experiment_params.py` | sensibilità di IRLS (`ε_thr`, `λ`) e SGPTL (`δ₀`, `ρ`) | `params_irls.pdf`, `params_dsm.pdf` |
| `experiment_scalability.py` | tempo totale e per-iter al variare di `H` (con `M=5H`) | `scalability.pdf`, `scalability.csv` |

I dettagli di cosa fa ognuno e quali parametri scegliere sono in
`03-experiments-and-tests.md`.

## Una nota sul rumore numpy/Apple Accelerate

Su macOS Apple Silicon (M1–M4) `numpy.matmul` può emettere
`RuntimeWarning: divide by zero / overflow / invalid` anche su matrici
perfettamente sane. È un bug noto del backend BLAS Accelerate e non
influisce sui risultati (verificato dalla test suite, che gira con
warnings come errors quando serve). Negli script di esperimento li
silenziamo all'ingresso:

```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
np.seterr(all="ignore")
```

`pytest.ini` fa lo stesso. Su Linux i warning non compaiono affatto.

## Stile di codice

Niente di esotico:

- PEP-8, formattato manualmente (no formatter automatico).
- Docstring numpy-style su ogni funzione pubblica con `Parameters` /
  `Returns`.
- `numpy` solo, niente `pandas` / `polars` (le tabelle sono CSV scritti
  con `csv.writer`).
- `matplotlib` con backend `Agg` (no GUI), savefig in PDF.
- Una sola classe (`ELM` in `elm.py`); il resto sono funzioni pure.
- Nessun decorator se non per pytest (`@pytest.mark.parametrize`,
  `@pytest.fixture`).
