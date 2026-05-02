# 02 — Algoritmi: IRLS e SGPTL passo per passo

Per ciascun algoritmo: la matematica del report, come è tradotta nel
codice, e le scelte implementative non ovvie.

---

## A1 — Iteratively Reweighted Least Squares (IRLS)

### La matematica (Cap. 2 del report)

L'idea è di sostituire `f(w)` con una sequenza di surrogati quadratici
`Q(w, w_k)` che maggiorano `f` ovunque e coincidono con `f` in `w_k`:

$$
Q(\mathbf{w}, \mathbf{w}_{k}) \;=\; \tfrac{1}{2} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^{2}
\;+\; \tfrac{\lambda}{2} \mathbf{w}^{\top} \mathbf{W}_{k}^{\top} \mathbf{W}_{k} \mathbf{w}
\;+\; \tfrac{\lambda}{2} \|\mathbf{w}_{k}\|_{1},
$$

con $(\mathbf{W}_k)_{ii} = |w_{k,i}|^{-1/2}$ (fix di sicurezza con
$\varepsilon_{\text{thr}}$ qui sotto). Imponendo $\nabla_{\mathbf{w}} Q = \mathbf{0}$:

$$
\mathbf{X}^{\top}(\mathbf{X}\mathbf{w} - \mathbf{y}) \;+\; \lambda\,\mathbf{W}_{k}^{\top}\mathbf{W}_{k}\,\mathbf{w} \;=\; \mathbf{0}
\;\;\Longrightarrow\;\;
\bigl(\mathbf{X}^{\top}\mathbf{X} + \lambda\, \mathbf{W}_{k}^{\top}\mathbf{W}_{k}\bigr)\mathbf{w} \;=\; \mathbf{X}^{\top}\mathbf{y}.
$$

Con $(\mathbf{W}_k^\top \mathbf{W}_k)_{ii} = 1/|w_{k,i}|$ il sistema è SPD
e si risolve in chiusa con Cholesky. Le proprietà MM (report § 2.2)
garantiscono che $f(\mathbf{w}_{k+1}) \le f(\mathbf{w}_k)$ a ogni passo.

### Il codice riga per riga

`src/irls.py`:

```python
def irls(X, y, lam,
         eps_thr=1e-8,         # safety threshold for the diagonal weights
         eps_stop=1e-8,        # stopping criterion on relative iterate change
         k_max=200,
         solver='cholesky',    # 'cholesky' or 'cg'
         w0=None,
         f_star=None,          # if given, gap = f - f_star is recorded
         verbose=False):

    # Pre-compute (done once)
    A = X.T @ X            # O(M H^2)
    b = X.T @ y            # O(M H)

    # OLS warm start: w0 = (X^T X)^-1 X^T y, with a tiny ridge for safety
    if w0 is None:
        w = solve_spd(A + 1e-12 * np.eye(n), b, method=solver)

    # Main loop
    for k in range(k_max):
        # Step 1: diagonal weights with safety threshold (report Eq 2.5)
        diag_D = 1.0 / np.maximum(np.abs(w), eps_thr)

        # Step 2: assemble Q_k = A + lam * D_k.
        # Only the diagonal of A changes each iteration -> O(H).
        Q = A.copy()
        Q[np.arange(n), np.arange(n)] += lam * diag_D

        # Step 3: solve Q_k w_{k+1} = b   (cholesky O(H^3/3) or CG O(p H^2))
        w = solve_spd(Q, b, method=solver)

        # Step 4: stopping criterion (relative change of the iterate)
        delta_w = np.linalg.norm(w - w_old) / max(1.0, np.linalg.norm(w_old))
        if delta_w < eps_stop:
            break
```

**Tre dettagli importanti:**

1. **`A = XᵀX` è precomputato fuori dal loop.** Costa $O(MH^2)$ una sola
   volta. Ammortizzato sui ~50–100 step IRLS dei nostri esperimenti, è
   quasi gratis. Senza precomputo ogni step costerebbe $O(MH^2 + H^3)$
   → totale $O(k\,MH^2 + k\,H^3)$, e il primo addendo dominerebbe per
   $M \gg H$.

2. **Solo la diagonale di Q cambia ad ogni iterazione.** Update in-place:
   ```python
   Q = A.copy()
   Q[np.arange(n), np.arange(n)] += lam * diag_D
   ```
   La copia di $A$ costa $O(H^2)$, ma è inevitabile se vogliamo
   `cho_factor` (che modifica il buffer in-place). Se $H$ diventa
   grande conviene l'opzione CG che evita di toccare l'intera $A$.

3. **Il safety threshold $\varepsilon_{\text{thr}}$.** Quando $|w_i|$ si
   avvicina a zero, $1/|w_i|$ esplode e Q diventa numericamente
   patologica. Il report § 2.1 prescrive
   $(\mathbf{W}_k)_{ii} = (\max(|w_i|, \varepsilon_{\text{thr}}))^{-1/2}$,
   che corrisponde a `1.0 / np.maximum(np.abs(w), eps_thr)` nel codice.
   Il default $10^{-8}$ è validato dal sweep in `experiment_params.py`:
   sotto $10^{-12}$ il limite è quello floating-point del solver
   lineare, sopra $10^{-6}$ la sparsità non è più recuperata.

### Stopping criterion

L'algoritmo si ferma quando il cambiamento relativo dell'iterata scende
sotto `eps_stop` (default $10^{-8}$):

$$
\frac{\|\mathbf{w}_{k+1} - \mathbf{w}_{k}\|_{2}}{\max(1, \|\mathbf{w}_k\|_{2})} < \varepsilon_{\text{stop}}.
$$

Negli esperimenti del report si imposta `eps_stop=1e-12` per non
fermarsi prima che la curva di gap esibisca la sua forma naturale. Con
`eps_stop` realistico (es. $10^{-8}$) IRLS si ferma in 30–80 iterazioni
sulle istanze tipiche.

### Cosa restituisce

Un dict con le chiavi:

| chiave | tipo | significato |
|---|---|---|
| `w` | ndarray (H,) | soluzione finale |
| `f_vals` | list[float] | $f(\mathbf{w}_k)$ per ogni $k$ inclusa la iniziale, lunghezza $n_{\text{iter}}+1$ |
| `gaps` | list[float] | $f(\mathbf{w}_k) - f^*$ se `f_star` è dato |
| `times` | list[float] | tempo CPU cumulato per iter |
| `n_iter` | int | iterazioni effettive |
| `converged` | bool | True se `eps_stop` è stato raggiunto |

---

## A2 — Deflected Subgradient with Polyak Target Level (SGPTL)

### La matematica (Cap. 3 del report)

L'algoritmo prende un sub-gradiente $\mathbf{g}_i \in \partial f(\mathbf{w}_i)$
e costruisce una direzione *deflessa* mescolandolo con la direzione
precedente:

$$
\mathbf{d}_i \;=\; \gamma_i \,\mathbf{g}_i \;+\; (1 - \gamma_i)\, \mathbf{d}_{i-1},
\qquad \gamma_i \in [0, 1].
$$

$\gamma_i$ è scelto greedy per **minimizzare $\|\mathbf{d}_i\|^2$**, che in
chiusa dà (report eq. 3.5):

$$
\gamma^{*} \;=\; \frac{\|\mathbf{d}_{i-1}\|^{2} - \langle \mathbf{g}_i, \mathbf{d}_{i-1} \rangle}{\|\mathbf{g}_i - \mathbf{d}_{i-1}\|^{2}},
\qquad
\gamma_i = \min(1, \max(0, \gamma^{*})).
$$

Il passo è una variante del Polyak step con target level $f_{\text{ref}} - \delta$
in luogo dell'ignoto $f^*$:

$$
\beta_i = \min(\beta, \gamma_i),
\qquad
\alpha_i \;=\; \frac{\beta_i \,\bigl(f(\mathbf{w}_i) - (f_{\text{ref}} - \delta)\bigr)}{\|\mathbf{d}_i\|^{2}}.
$$

La condizione $\beta_i \le \gamma_i$ è quella che la prova del Teorema
3.1 (eq. 3.14) usa per garantire la decrescita per-step. Quando
$f(\mathbf{w}_{i+1})$ non migliora abbastanza per troppe iterazioni, si
contrae $\delta \leftarrow \rho \delta$ ("patience mechanism", report
§ 3.3).

### Il codice riga per riga

`src/deflected_subgradient.py`:

```python
def deflected_subgradient(X, y, lam,
                           w0=None,
                           i_max=5000,
                           beta=1.0,
                           delta0=None,
                           R=None,
                           rho=0.95,
                           f_star=None,
                           verbose=False, verbose_freq=500):

    # OLS warm start (same as IRLS — report Sec 3.4)
    if w0 is None:
        w = solve_spd(X.T @ X + 1e-12 * np.eye(n), X.T @ y, method='cholesky')

    # State
    delta   = delta0 or max(0.1 * f_curr, 1e-4)
    R       = R or 10.0 * np.sqrt(i_max)         # accumulated-travel patience
    R_iter  = max(int(i_max / 100), 50)          # iteration-count patience (safeguard)
    f_ref   = f_curr
    f_bar   = f_curr        # record value
    w_best  = w.copy()
    d_prev  = np.zeros(n)
    r       = 0.0           # accumulated travel without improvement
    no_imp  = 0             # consecutive iterations without sufficient decrease

    for i in range(i_max):
        # Step 1: subgradient (minimum-norm convention: s_i = sign(w_i), s_i = 0 at w_i = 0)
        g = subgradient_f(X, y, w, lam)

        # Step 2: optimal deflection gamma_i.
        # CONVENTION: when d_prev == 0 we use gamma = 1 (pure subgradient step),
        # otherwise the unconstrained minimiser of ||gamma g + (1-gamma) d_prev||^2
        # would be gamma = 0, which freezes the algorithm.
        if i == 0 or np.dot(d_prev, d_prev) < 1e-30:
            gamma = 1.0
        else:
            gamma = _optimal_gamma(g, d_prev)

        # Step 3: deflected direction
        d = gamma * g + (1.0 - gamma) * d_prev
        d_norm_sq = np.dot(d, d)
        if d_norm_sq < 1e-30:
            break       # degenerate: at a stationary point of the subdifferential

        # Step 4: stepsize-restricted Polyak step (report Alg 2 lines 11-12)
        beta_i = min(beta, gamma)
        target = f_ref - delta
        # Clip a negative numerator to zero (alpha = 0 is a no-op step,
        # avoids deadlock when beta_i = gamma_i is very small).
        numerator = max(0.0, beta_i * (f_curr - target))
        alpha = numerator / d_norm_sq

        # Step 5: update
        w_new = w - alpha * d
        if not np.all(np.isfinite(w_new)):
            w_new = w.copy()             # safety: a non-finite step becomes a no-op

        f_new = f_lasso(X, y, w_new, lam)

        # Step 6: record value (always tracks the best f seen)
        if f_new < f_bar:
            f_bar  = f_new
            w_best = w_new.copy()

        # Step 7: target-level logic (report Alg 2 lines 14-21) plus safeguard
        if f_new <= f_ref - delta / 2.0:
            f_ref  = f_bar      # significant improvement
            r      = 0.0
            no_imp = 0
            d_prev = d
        elif r > R or no_imp > R_iter:
            delta *= rho        # contract delta (report) OR stagnation safeguard
            r      = 0.0
            no_imp = 0
            d_prev = np.zeros(n)   # reset memory to escape gamma -> 0 deadlock
        else:
            r      += alpha * np.sqrt(d_norm_sq)
            no_imp += 1
            d_prev = d

        w = w_new
        f_curr = f_new

    return {'w': w_best, 'f_vals': ..., 'f_bar': ..., 'gaps': ..., ...}
```

### La parte importante: il bug γ→0 e perché c'è il safeguard

La prima implementazione fedele al pseudocodice del report mostrava un
sintomo strano: su istanze con OLS warm start, il record value
$\bar{f}^i$ si congelava dopo ~50 iterazioni e non migliorava mai più.
Tracciando l'algoritmo passo per passo:

1. L'OLS warm start posiziona $\mathbf{w}_0$ in una regione vicino a un
   punto di KKT (perché OLS minimizza la parte liscia $\tfrac12\|Xw-y\|^2$
   e il vettore così ottenuto è già "quasi sparso" sui componenti dove
   $|X^T(Xw-y)_i| \le \lambda$).
2. Lì il sub-gradiente di norma minima $\mathbf{g}$ diventa quasi
   parallelo a $\mathbf{d}_{i-1}$ (entrambi vivono nello stesso
   sottospazio "lento" del problema).
3. La $\gamma$ ottima, calcolata per minimizzare $\|\mathbf{d}_i\|^2$,
   tende a 0: se $\mathbf{g}$ è collineare a $\mathbf{d}_{i-1}$, la
   combinazione che minimizza la norma è quella che pesa di più
   $\mathbf{d}_{i-1}$ (memoria), non $\mathbf{g}$ (innovazione).
4. La condizione $\beta_i = \min(\beta, \gamma_i)$ del report (necessaria
   per la prova di Teorema 3.1) moltiplica il numeratore Polyak per
   $\gamma_i$, quindi $\alpha_i \to 0$.
5. L'iterata smette di muoversi.
6. Il counter di pazienza $r = \sum_i \alpha_i \|\mathbf{d}_i\|$ smette
   di crescere.
7. La condizione $r > R$ del report non si verifica mai → $\delta$ non
   si contrae → il bug è permanente, l'algoritmo è in deadlock.

**Il safeguard.** Tre modifiche minimali, tutte coerenti con la
convergenza del report:

- **Numeratore clippato a zero** invece di "skip iteration con `continue`".
  Un passo $\alpha = 0$ è innocuo (la new iterate è uguale alla
  precedente) e permette al meccanismo di pazienza di contare
  l'iterazione come non-improvement, invece di bypassarlo.

- **Pazienza basata su iter count** $R_{\text{iter}} = \max(i_{\max}/100, 50)$
  *in OR* con $r > R$. Cattura il caso patologico in cui $r$ non cresce.

- **Reset di $\mathbf{d}_{i-1} \leftarrow \mathbf{0}$** alla contrazione di
  $\delta$, *insieme* alla convenzione $\gamma = 1$ quando
  $\mathbf{d}_{i-1}$ è il vettore zero. Questo forza un passo
  sub-gradiente puro nell'iterazione successiva, sufficiente a uscire
  dal punto stagnante.

**Perché il reset non rompe la prova di convergenza.** Il Teorema 3.1
(Equazione 3.14) della convergenza è una disuguaglianza per-step:

$$
\|\mathbf{w}_{i+1} - \mathbf{w}^{*}\|^{2} \;\le\; \|\mathbf{w}_{i} - \mathbf{w}^{*}\|^{2} - \frac{(f(\mathbf{w}_{i}) - f^{*})^{2}}{\|\mathbf{d}_{i}\|^{2}}.
$$

Vale per ogni $i$ indipendentemente, *non richiede* che la memoria
$\mathbf{d}_{i-1}$ sia stata accumulata su tutta la storia. Inserire un
passo sub-gradiente puro ($\mathbf{d}_i = \mathbf{g}_i$) in mezzo a una
sequenza deflessa è ancora coperto dalla prova: quel singolo passo
soddisfa banalmente la disuguaglianza (è quella standard del Polyak
sub-gradiente).

### Il helper `_optimal_gamma`

Implementa la formula chiusa di Eq. 3.5:

```python
def _optimal_gamma(g, d_prev):
    diff = g - d_prev
    denom = np.dot(diff, diff)
    if denom < 1e-30:
        return 1.0                    # g == d_prev (report convention)
    gamma_star = (np.dot(d_prev, d_prev) - np.dot(g, d_prev)) / denom
    return float(np.clip(gamma_star, 0.0, 1.0))
```

Testato a parte (`test_dsm.py::test_optimal_gamma_is_argmin_of_norm`)
contro un argmin numerico su griglia: l'errore è < 2e-3 in modulo.

### Cosa restituisce SGPTL

| chiave | tipo | significato |
|---|---|---|
| `w` | ndarray (H,) | $\mathbf{w}_{\text{best}}$, l'iterata che realizza $\bar{f}$ — non $\mathbf{w}_{i_{\max}}$! |
| `f_vals` | list[float] | $f(\mathbf{w}_i)$ per ogni $i$ (oscillante) |
| `f_bar` | list[float] | $\bar{f}^i$ (record, monotono non crescente) |
| `gaps` | list[float] | $\bar{f}^i - f^*$ se `f_star` è dato |
| `times` | list[float] | tempo CPU cumulato |
| `delta_hist` | list[float] | evoluzione di $\delta$ (utile per diagnostica) |
| `n_iter` | int | iterazioni effettive |

### Choosing parameters: cheat sheet

Su istanze tipo "ELM-LASSO con $H \le 2000$, $M = 5H$, $\lambda \in [0.05, 0.2]$":

| Parametro | Default raccomandato | Range che ha senso esplorare |
|---|---:|---|
| `beta` | `1.0` | sempre `1` (vedi report § 3.2 — più alti vengono clippati) |
| `delta0` | `0.1 * f(w0)` | `[0.05, 0.2] * f_star` (sweet spot da `experiment_params.py`) |
| `rho` | `0.9` | `[0.85, 0.95]` (vedi sweep ρ) |
| `R` | `10 * sqrt(i_max)` | non è critico, il safeguard basato su `R_iter` lo copre |
| `i_max` | `8000` | proporzionale all'accuracy target: $O(\varepsilon^{-2})$ |

Negli esperimenti del report il convergence_vs_iter usa esattamente
questi default.

---

## Confronto operativo IRLS vs SGPTL (riassunto)

| Aspetto | IRLS | SGPTL |
|---|---|---|
| Costo per-iter (dopo precompute) | $O(H^3/3)$ Cholesky | $O(MH)$ |
| Costo precompute | $O(MH^2)$ | nullo |
| Tasso di convergenza | lineare (geometrico) | $O(1/\sqrt{i})$ |
| Monotonia $f$ | sì (MM) | no, solo $\bar{f}$ è monotona |
| Sparsità | esatta ($w_i = 0$ a tolleranza $\varepsilon_{\text{thr}}$) | approssimata, richiede thresholding |
| Parametri da tunare | `eps_thr` | `delta0`, `rho`, `R`, (`beta`) |
| Stopping criterion affidabile | sì (`eps_stop` su $\|w_{k+1} - w_k\|$) | no, si usa budget di iterazioni |
| Quando preferirlo | `M \gg H` (regime tipico ELM) | `H^2 \gg M`, o memoria limitata |

Tutti questi numeri sono verificati empiricamente nel report, Cap. 5.
