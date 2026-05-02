# 03 — Esperimenti e tests

Questo documento spiega cosa misurano i 4 script in `experiments/`, come
sono parametrizzati, cosa producono, e come è organizzata la test suite.
Alla fine ci sono le istruzioni per riprodurre passo-passo ogni figura
del report.

---

## I 4 esperimenti

Tutti gli script seguono la stessa struttura:

1. Import + `apply_style()` (stile matplotlib comune da
   `_plot_style.py`).
2. Costanti di configurazione in cima al file (problema, parametri
   algoritmici, percorsi di output).
3. Una funzione `run()` che genera il problema, esegue IRLS e/o SGPTL,
   produce figure e/o tabelle.
4. `if __name__ == "__main__": run()`.

### `experiment_convergence.py`

**Cosa misura.** Comportamento di convergenza dei due algoritmi sulla
*stessa* istanza con OLS warm start in comune.

**Configurazione.** $H = 100$, $M = 300$, $\lambda = 0.1$, sparsità del
`w_true` 10%, rumore $\sigma = 0.05$. IRLS con $\varepsilon_{\text{thr}} =
10^{-8}$, $\varepsilon_{\text{stop}} = 10^{-12}$, 100 iterazioni;
SGPTL con $\beta = 1$, $\delta_0 = 0.1\,f^*$, $\rho = 0.9$, 8000
iterazioni.

**Output:**

| File | Cosa mostra |
|---|---|
| `results/figures/convergence_vs_iter.pdf` | Pannello sinistro: gap IRLS vs iter su semilog (retta = rate lineare). Pannello destro: gap SGPTL con due curve sovrapposte — `f(w_i) - f*` (oscillante, sublineare) e `f_bar^i - f*` (record, scalini). |
| `results/figures/convergence_vs_time.pdf` | Stesse curve ma su asse CPU time. |
| `results/figures/dsm_nonmonotone.pdf` | Asse lineare. Mostra il fenomeno della non-monotonia: `f(w_i)` in arancione oscilla, `f_bar` in nero scende monotonamente, `f*` tratteggiato in verde. |

**Cosa cercare.** Sul pannello SGPTL del primo grafico, la curva
arancione (`f(w_i)`) deve avere oscillazioni che si smorzano nel tempo
e seguire un trend $O(1/\sqrt{i})$ visibile. Quando il record value
"salta" verso il basso, è una contrazione di $\delta$ andata a buon
fine (vedi `delta_hist` nel result dict).

### `experiment_comparison.py`

**Cosa misura.** Iterazioni e tempo CPU per raggiungere un accuracy
target $\varepsilon \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 10^{-6}\}$.

**Configurazione.** $H = 50$, $M = 200$, $\lambda = 0.1$ (problema
moderato perché SGPTL ha bisogno di tempi ragionevoli per scendere a
$\varepsilon = 10^{-4}$). IRLS con $k_{\max} = 300$, SGPTL con
$i_{\max} = 30000$.

**Output:**

| File | Cosa mostra |
|---|---|
| `results/figures/comparison_irls_dsm.pdf` | Due pannelli (vs iter, vs tempo) sovrapponendo IRLS, `SGPTL f_bar`, `SGPTL f(w_i)`. Titolo con i parametri del problema. |
| `results/tables/comparison_table.csv` | Una riga per ogni $\varepsilon$ con `irls_iters, irls_time, dsm_iters, dsm_time`. Valori vuoti se l'algoritmo non ha raggiunto $\varepsilon$ nel budget. |

**Cosa cercare.** La tabella mostra il pattern delle rate: IRLS aggiunge
$\sim 5\times$ iterazioni per ogni decade di $\varepsilon$ (lineare),
SGPTL moltiplica per $\sim 5\times$ a ogni decade ma su una base molto
più grande (sublineare). Sui valori specifici del report:
$\{3, 7, 19, 101\}$ per IRLS contro $\{3, 2126, 10591, 17837\}$ per
SGPTL.

### `experiment_params.py`

**Cosa misura.** Sensibilità ai parametri.

- IRLS: $\varepsilon_{\text{thr}} \in \{10^{-4}, 10^{-6}, 10^{-8},
  10^{-10}, 10^{-12}\}$ e $\lambda \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$.
- SGPTL: $\delta_0 / f^* \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$ e
  $\rho \in \{0.5, 0.7, 0.9, 0.95, 0.99\}$.

**Configurazione comune.** $H = 100$, $M = 400$, $\lambda = 0.1$ per i
due sweep IRLS e per il sweep $\delta_0$. Per il sweep $\rho$ si usa
*cold start* `w0=zeros`: con OLS warm start le contrazioni di $\delta$
si attivano poche volte e $\rho$ ha effetto marginale.

**Output:**

| File | Cosa mostra |
|---|---|
| `results/figures/params_irls.pdf` | Sinistra: gap vs iter per ogni $\varepsilon_{\text{thr}}$ con sparsità in legenda. Destra: gap vs iter per ogni $\lambda$ con sparsità in legenda. |
| `results/figures/params_dsm.pdf` | Sinistra: gap vs iter per ogni $\delta_0$, U-shape evidente. Destra: gap vs iter per ogni $\rho$ con conteggio contrazioni in legenda. |

**Cosa cercare.**

- $\varepsilon_{\text{thr}} = 10^{-4}$ produce un plateau (sparsità 0%).
  Da $10^{-6}$ in giù, sparsità stabile a 86% (vicino al vero 88%).
  Da $10^{-12}$ in giù, il limite è la precisione del solver lineare.
- Gap del $\lambda$-sweep è quasi indipendente da $\lambda$ (rate
  lineare poco influenzato dal condizionamento), ma sparsità monotona
  in $\lambda$.
- Sul $\delta_0$-sweep, il valore "0.01·f\*" plateaua a 0.13: target
  troppo alto, l'algoritmo non scende mai. Il minimo è a 0.05·f\*.
- Sul $\rho$-sweep (cold start), $\rho = 0.9$ è il sweet spot
  (4.9·10⁻⁶), $\rho = 0.5$ è troppo aggressivo (8.5·10⁻⁵
  in 100 contrazioni) e $\rho = 0.99$ è troppo lento
  (1.7·10⁻²: target sotto $f^*$ per troppo a lungo).

### `experiment_scalability.py`

**Cosa misura.** Tempo totale e per-iter al variare di $H \in \{50, 100,
500, 1000, 2000\}$ con $M = 5H$. Confronto con i riferimenti $O(H^3)$
(IRLS) e $O(H^2)$ (SGPTL, perché $M = 5H$ rende $O(MH) = O(H^2)$).

**Configurazione.** IRLS con 100 iterazioni Cholesky;
SGPTL con 3000 iterazioni, $\beta = 1$, $\delta_0 = 0.1\,f^*$, $\rho = 0.9$.

**Output:**

| File | Cosa mostra |
|---|---|
| `results/figures/scalability.pdf` | Sinistra: tempo totale log-log con curve $O(H^3)$ (IRLS) e $O(H^2)$ (SGPTL) come riferimento tratteggiato. Destra: tempo per iterazione log-log. |
| `results/tables/scalability.csv` | Una riga per ogni $H$ con `n, m, t_irls, iter_irls, gap_irls, t_dsm, iter_dsm, gap_dsm`. |

**Cosa cercare.** La pendenza log-log di IRLS (sinistra) deve
allinearsi con la guida $O(H^3)$. SGPTL è più piatto ($O(H^2)$) ma
ha un offset più alto: nel regime $M = 5H$ IRLS è ~5–10× più veloce
in tempo totale per ogni $H$ testato. Il **crossover** in cui SGPTL
diventa più veloce richiederebbe $M \ll H^2$ (es. $H = 5000$,
$M = 100$), regime non interessante per ELM ma teoricamente
prevedibile dalla Tabella 4.1 del report.

### Stile dei plot (`_plot_style.py`)

Tutti gli script usano lo stesso file di stile:

```python
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM, ...
apply_style()              # imposta rcParams (font serif, dpi, ecc.)
fig, ax = plt.subplots(figsize=(11.5, 4.2))
ax.semilogy(..., color=COLOR_IRLS)
style_axes(ax)             # spine senza top/right, tick orientation
fig.tight_layout()
fig.savefig("...pdf")      # 220 DPI, bbox tight
```

Costanti di colore disponibili:

- `COLOR_IRLS = "#1f4e79"` (blu scuro)
- `COLOR_DSM  = "#c0392b"` (rosso caldo)
- `COLOR_FBAR = "#1c1c1c"` (quasi-nero per il record overlay)
- `COLOR_FCUR = "#e67e22"` (arancio per `f(w_i)` overlay)
- `COLOR_REF  = "#2c7a30"` (verde per `f*`)
- `COLOR_AUX  = "#7f7f7f"` (grigio per linee di riferimento $O(\cdot)$)

E rampe sequenziali per gli sweep di parametri:

- `RAMP_BLUES`, `RAMP_REDS`, `RAMP_ORANGES`, `RAMP_PURPLES`.

---

## La test suite

53 test pytest in 6 file. Tutti girano in ~2 secondi sul mio MacBook M4.
Lanciali con:

```bash
cd progetto/code
python -m pytest tests/ -v        # verbose
python -m pytest tests/ --tb=short
python -m pytest tests/test_dsm.py::test_optimal_gamma_is_argmin_of_norm  # singolo
python -m pytest tests/ -m slow   # solo i test slow
python -m pytest tests/ -m "not slow"  # tutti tranne quelli slow
```

### Fixture condivise (`tests/conftest.py`)

```python
LassoProblem = NamedTuple(
    "LassoProblem",
    [("X", np.ndarray), ("y", np.ndarray), ("lam", float),
     ("w_true", np.ndarray), ("w_star", np.ndarray), ("f_star", float)]
)
```

| Fixture | $H$ | $M$ | $\lambda$ | sparsità | seed | uso tipico |
|---|---:|---:|---:|---:|---:|---|
| `small_problem` | 10 | 50 | 0.1 | 30% | 0 | test veloci |
| `medium_problem` | 50 | 200 | 0.05 | 20% | 1 | convergenza, KKT |
| `elm_problem` (dict) | 80 | 300 | 0.05 | 15% | 2 | pipeline ELM end-to-end |

### Cosa testano i singoli file

**`test_lasso_utils.py`** (9 test)

- `test_f_lasso_handcrafted` — esempio numerico calcolato a mano. Se
  questo test fallisce, qualcuno ha cambiato il fattore 1/2.
- `test_grad_smooth_finite_difference` — gradiente analitico vs
  differenze finite centrali su 5 punti casuali.
- `test_subgradient_at_smooth_point` — al punto liscio coincide col
  gradiente.
- `test_subgradient_minimum_norm_at_zero` — verifica che la convenzione
  `s_i = 0` per `w_i = 0` produce un sub-gradiente valido di norma minima.
- `test_check_optimality_zero_at_sklearn_solution` — la soluzione
  sklearn rispetta KKT a tolleranza < 5e-4.

**`test_linear_solvers.py`** (9 test, parametrizzati su 3 dimensioni)

- Cholesky residuo $< 10^{-9}$.
- CG agreement con Cholesky $< 10^{-7}$.
- CG terminazione finita in $\le n$ iterazioni (proprietà teorica).
- `solve_spd` con metodo sconosciuto solleva `ValueError`.

**`test_irls.py`** (11 test)

- `test_irls_monotone_decrease` — su 5 seed parametrizzati,
  $f(\mathbf{w}_{k+1}) \le f(\mathbf{w}_k)$ a meno di rumore floating-point
  ($10^{-9}$).
- `test_irls_converges_to_sklearn_fstar` — gap finale $< 10^{-3}$.
- `test_irls_kkt_residual_small` — violazione KKT $< 10^{-2}$.
- `test_irls_produces_sparse_iterate_when_lambda_large` — $\lambda = 2$
  produce sparsità $\ge 50\%$.
- `test_irls_warm_start_honoured` — partendo da `w_star` IRLS converge
  in pochi iter.
- `test_irls_solver_choice_cholesky_vs_cg` — entrambi raggiungono il
  vicinato di $f^*$, anche se la soluzione $w$ può differire (CG meno
  preciso su Q condizionata male).

**`test_dsm.py`** (10 test)

- `test_optimal_gamma_in_unit_interval` — la formula chiusa rispetta la
  proiezione su $[0, 1]$ su 20 input casuali.
- `test_optimal_gamma_when_g_equals_d` — convenzione $\gamma = 1$
  quando $\mathbf{g} = \mathbf{d}_{i-1}$.
- `test_optimal_gamma_is_argmin_of_norm` — confronto con argmin
  numerico su griglia (errore $< 2 \cdot 10^{-3}$).
- `test_dsm_default_warmstart_is_ols` — con `i_max=0`, `w_best`
  coincide con la soluzione OLS.
- `test_dsm_record_non_increasing` — $\bar{f}^i$ monotona su 3 seed.
- `test_dsm_record_converges_to_fstar` — *marker `slow`*: gap finale
  $< 5 \cdot 10^{-2}$ in 10000 iter sul `small_problem`.
- `test_dsm_subgradient_is_in_subdifferential` — disuguaglianza
  sub-gradiente $f(w') \ge f(w) + \langle g, w' - w \rangle$ in 10
  direzioni casuali.

**`test_elm.py`** (9 test) — shape e range delle attivazioni, fit
end-to-end con IRLS e SGPTL, errori (predict before fit, activation
sconosciuta, solver sconosciuto).

**`test_data_generation.py`** (5 test)

- `test_make_lasso_problem_fstar_is_local_minimum` — perturbazioni
  casuali di `w_star` non riducono `f`. Garantisce che `f_star` è
  davvero il valore ottimo.
- `test_make_elm_problem_consistency` — `X_hid == σ(X_raw W1ᵀ)`,
  `f_star == f(X_hid, y, w_star)`.
- `test_make_elm_problem_kkt_at_wstar` — KKT violation $< 10^{-2}$.

### `pytest.ini`

Filtra warning innocui (RuntimeWarning di Apple Accelerate, sklearn
ConvergenceWarning su problemi quasi-degeneri usati nei test):

```ini
[pytest]
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::FutureWarning
    ignore::RuntimeWarning
    ignore:.*ConvergenceWarning.*
markers =
    slow: marks tests as slow (deselect with -m "not slow")
addopts = -ra
```

---

## Come riprodurre tutte le figure del report

Da `progetto/code/`:

```bash
# Da zero (assumendo dipendenze installate)
python experiments/experiment_convergence.py
python experiments/experiment_comparison.py
python experiments/experiment_params.py
python experiments/experiment_scalability.py

# Copia nelle immagini del report e ricompila
cp results/figures/*.pdf ../report/images/
cd ../report && latexmk -pdf main.tex
```

Tempi tipici sul mio M4:

| Script | Tempo |
|---|---:|
| convergence | ~5 s |
| comparison  | ~2 s |
| params      | ~25 s (sweep largo) |
| scalability | ~2 min (per H=2000) |

Il totale è dominato da `scalability` per via del Cholesky $H = 2000$
(O(H³) ≈ 10⁹ flop).

## Aggiungere un esperimento nuovo

Template:

```python
"""experiment_my_thing.py — what it measures."""
import os, sys, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np; np.seterr(all="ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import irls, deflected_subgradient, make_lasso_problem
from _plot_style import apply_style, style_axes, COLOR_IRLS, COLOR_DSM
apply_style()

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def run() -> None:
    X, y, _, f_star, _ = make_lasso_problem(n=100, m=300, lam=0.1, random_state=42)
    res = irls(X, y, 0.1, f_star=f_star, ...)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy(res["gaps"], color=COLOR_IRLS)
    ax.set_xlabel("Iteration"); ax.set_ylabel("gap")
    style_axes(ax); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "my_thing.pdf"))
    plt.close(fig)

if __name__ == "__main__":
    run()
```

Dopo aver scritto lo script, aggiungi un test in `tests/` che verifichi
le proprietà che vuoi affermare nel testo del report — la *checkable
claim* viene prima della prosa.

## Aggiungere un solver nuovo (es. proximal gradient)

1. Crea `src/proximal.py` con la stessa interfaccia di `irls.py`:
   ```python
   def proximal_gradient(X, y, lam, *, k_max, ..., f_star=None) -> dict:
       return {"w": w_final, "f_vals": ..., "gaps": ..., "times": ..., "n_iter": ...}
   ```
   Tienilo *function-based*, non class-based, per coerenza con
   `irls()` e `deflected_subgradient()`.

2. Esponi la funzione in `src/__init__.py`.

3. Aggiungi il test file `tests/test_proximal.py` riusando le fixture
   `small_problem` / `medium_problem`. Prendi spunto da `test_irls.py`
   per la struttura.

4. Se vuoi confrontarlo nei plot esistenti, modifica gli script
   pertinenti (`experiment_comparison.py` è il candidato naturale).

5. Documentalo qui in `utils/` aggiungendo una sezione a
   `02-algorithms.md`.
