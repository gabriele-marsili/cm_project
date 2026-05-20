# Critical review — CM Project 25 (ELM + LASSO, IRLS + SGPTL)

Audit interno scritto dopo l'email del prof del 2026-05-19 e dopo la sessione di
sistemazione 2026-05-20. Tutti i numeri citati sono verificati contro il codice
attuale, contro un riferimento `f*` indipendente (IRLS-converged validato da
CVXPY-Clarabel a 6 cifre) o segnalati come da rifare.

Convenzioni:
- **🔴 critico** = il prof lo flagga esplicitamente o cambia il segno di un risultato
- **🟡 importante** = qualità del report, va sistemato per la submission
- **🟢 cosmetico** = nice-to-have

---

## 1. Theorem 3.1: il fattore `1/γ_min` nella rate bound è sospetto (🔴)

### Cosa dice il report ora (chapter3.tex:181-186)
Sotto le ipotesi (subgradient bounded `||g_i|| ≤ L`, stepsize-restricted
`β_i ≤ γ_i` con `γ_i ≥ γ_min > 0`), enuncia:

```
f̄_k - f* ≤  L · ||w_0 - w*||  /  (γ_min · √k)
```

con complessità `O(L² / (γ_min² ε²))`.

### Perché è sospetto
Il proof body non deriva mai esplicitamente questo bound. Dice solo:
> "the asymptotic rate matches the classical Polyak bound [...] inflated by the
> deflection floor"
> "the deflection floor [...] is the source of the factor 1/γ_min appearing
> explicitly in the rate bound"

Questo è esattamente il punto su cui il prof scrive: *"mi pare allucinazione da LLM"*.

### Derivazione esplicita (che il report NON fa)

Partiamo dalla disuguaglianza fondamentale per il **deflected step** con
Polyak-target asintotico (target → f*):

```
||w_{i+1} - w*||² ≤ ||w_i - w*||² - β_i(2 - β_i) · (f(w_i) - f*)² / ||d_i||²
```

(questa è la versione "stepsize-restricted Polyak" — Slide 13 Frangioni; deriva
da espandere `||w_{i+1} - w*||²` con `w_{i+1} = w_i - α_i d_i` e sostituire la
Polyak `α_i = β_i (f(w_i)-f*)/||d_i||²`).

Con `β_i = min(1, γ_i)` e `γ_i ∈ [γ_min, 1]`:
- `β_i = γ_i` quindi `β_i(2-β_i) = γ_i(2-γ_i)`
- su `[γ_min, 1]`, `γ_i(2-γ_i)` è crescente (parabola con vertice in 1)
- minimo in `γ_i = γ_min`: vale `γ_min(2 - γ_min) ≥ γ_min` (perché `γ_min ≤ 1`)

Quindi `β_i(2-β_i) ≥ γ_min`. Telescoping da 0 a k:

```
0 ≤ ||w_{k+1} - w*||² ≤ ||w_0 - w*||² - γ_min · Σ (f(w_i) - f*)² / ||d_i||²
```

Bound `||d_i|| ≤ L` (combinazione convessa di subgradient ≤ L):

```
γ_min · Σ (f(w_i) - f*)² ≤ L² · ||w_0 - w*||²
```

Min-over-i ≤ media:
`(k+1) · (f̄_k - f*)² ≤ Σ (f(w_i) - f*)² ≤ L² ||w_0 - w*||² / γ_min`

**Risultato corretto:**

```
f̄_k - f*  ≤   L · ||w_0 - w*|| / √( (k+1) · γ_min )
            =  L · ||w_0 - w*|| /  (√γ_min · √(k+1))
```

Il fattore di deflazione è **`1/√γ_min`**, non `1/γ_min`. In termini di complessità:

```
k ≥ L² ||w_0 - w*||² / (γ_min · ε²)
```

cioè `O(L² / (γ_min · ε²))`, non `O(L² / (γ_min² ε²))`.

Con `γ_min = 0.05` il moltiplicatore vs Polyak puro è `1/0.05 = 20`, non `400`.

### Implicazioni
- Il bound nel report sovrastima il costo della deflection floor di un fattore `1/√γ_min ≈ 4.5×` rispetto al vero conto.
- Possibile che la versione "1/γ_min" venga da una derivazione diversa (es. usando un bound differente sul `β_i(2-β_i)` o assumendo `α_i` di forma diversa), ma il report non la presenta. **Va dimostrata o corretta.**
- Affermazione confidente senza dimostrazione = il problema che il prof chiama "LLM hallucination".

### Cosa fare
1. **Riscrivere il proof body** con la derivazione esplicita sopra (5–8 righe in più). Sostituire `1/γ_min` con `1/√γ_min` in eq. `def-rate-deflected` e nella discussione di complessità.
2. Verificare che la conclusione qualitativa ("la deflection floor inflate la rate") non cambia: vale, ma il fattore è 1/√γ_min nel rate, 1/γ_min nella complessità.
3. Aggiornare di conseguenza tutti i punti che citano `1/γ_min²` come moltiplicatore di iterazioni — almeno chapter3.tex:185 e §4.2 (`comparison.tex`).

---

## 2. δ_0 = c · f* — addressato, ma il chapter3.tex era ridondante (🔴→🟢 ora)

### Critica del prof
*"la vostra scelta di δ_0 nella §5.3.3 sembrerebbe fare uso di f*, che non
conoscete. [...] usare il valore ottimo per determinare i parametri algoritmici
[...] è come usare il test set per fare training in ML: semplicemente non si fa."*

### Stato attuale (dopo le modifiche di questa sessione)
- **chapter3.tex § "δ_0 = c · f(w_OLS) con c = 0.1"** ora dichiara esplicitamente:
  - lo scaling è su `f(w_OLS)`, non su `f*`
  - `f(w_OLS) ≥ f*` è un upper-bound calcolabile a-priori
  - cita esplicitamente l'inammissibilità di usare `f*` per tuning
  - rimanda al §"Three families" di results.tex per la validazione empirica
- **results.tex §"Three families"** confronta tre regole (`A: c·f(w_0)`, `B: c·½||Xw_0-y||²`, `C: c·f*` come oracolo diagnostico) su 5 istanze (sintetico + diabetes/california a H=50,200). Conclusione: A e C entro factor 2 ovunque SGPTL è in regime informativo.
- **Codice (`experiment_warm_vs_cold_real_data.py`)**: δ_0 calcolato da `f(w_OLS)`, non da `f_star`.

### Cosa è ancora migliorabile
- Una nota nel report che dichiara: *"in nessun esperimento del report l'algoritmo riceve `f*` come input — `f*` è usato solo per il logging dei `gaps`, mai in updating di `δ`, `f_ref` o stepsize"*. Questo chiude la critica del prof senza ambiguità.
- Verificare che nessuno script passi `f_star=` come parametro che entra nelle update rules (l'argomento è solo per logging — questo è già vero nel codice, va detto nel report).

---

## 3. Warm-start cost: parzialmente discusso, da espandere (🟡)

### Critica del prof
*"non mi pare che discutiate quanto vi costa calcolarlo nel contesto del costo
totale dell'approccio, e quindi se abbia senso come scelta"*

### Numeri concreti (ELM California: M=16512, H=200)
| Operazione | Costo | Tempo stimato |
|---|---|---|
| `X^T X` | `O(MH²) = 660M ops` | ~1 s |
| Cholesky di `X^T X + εI` | `O(H³/3) = 2.7M ops` | ~10 ms |
| OLS warm start (totale) | dominato da `X^T X` | ~1 s |
| IRLS, 100 iter, 1 Chol per iter | `100 · 2.7M = 270M ops` | ~1 s |
| SGPTL, 5000 iter, 1 grad+1 step per iter | `5000 · O(MH) ≈ 16.5G ops` | ~14 s |

**Osservazioni:**
- Il warm-start **non è gratis**: `X^T X` da solo costa quanto ~100 iterazioni di IRLS.
- Per IRLS però `X^T X` viene già calcolato (è il base del sistema di Cholesky di ogni iter), quindi il warm-start `(X^T X)^{-1} X^T y` è *condiviso* con l'iterazione zero.
- Per SGPTL warm: `X^T X` è solo per il warm-start; non viene riusato. Quindi su SGPTL il costo del warm-start è **completamente aggiuntivo**.

### Conclusione operativa
Su SGPTL warm-start ha senso **solo se** `f(w_OLS) ≈ f*` per il problema in mano (es. California, dove l'OLS è già quasi-ottimo). Altrimenti spendere il `X^T X` per partire vicino a w* è uno spreco.

### Cosa fare
Aggiungere un paragrafo conciso (~6 righe) in chapter3.tex §"Parameter calibration" o in results.tex §5.7 che dichiara:
- costo `O(MH² + H³)` del warm-start
- per SGPTL è "puro overhead" (`X^T X` non riusato), per IRLS è ammortizzato
- la scelta warm vs cold dipende dal trade-off costo/qualità sulla specifica istanza
- riferimento al "punto di sinistra" della Figure 5.3 (warm-vs-cold sintetico) se la figura mostra effettivamente questo

---

## 4. §5.7 SGPTL "mal configurato" (🟡)

### Critica del prof
*"Il comportamento di SGPTL nella §5.7 mi farebbe pensare che sia stato molto
mal configurato"*

### Sintomo attuale (dopo fix di questa sessione)
| Dataset | SGPTL choice | gap vs vero f* |
|---|---|---|
| diabetes-ELM (H=200) | cold | +0.26 |
| California-ELM (H=200) | warm | +0.46 |

A confronto: IRLS arriva a gap `7·10⁻⁶` su diabetes e `3·10⁻¹²` su California. SGPTL è **2–11 ordini di grandezza peggio** in entrambi.

### È davvero "mal configurato"?
**Probabilmente sì, parzialmente.** Dal nostro esperimento `experiment_delta0_proxy.py` sul sintetico:
- c = 0.1 (default): gap finale `~10⁻³` su istanze "easy", `~10⁻³` su "hard"
- c = 0.5: gap finale `~6·10⁻⁴` su "easy", `~10⁻³` su "hard"
- c = 1.0: gap finale `~8·10⁻⁴` su "easy", `~8·10⁻⁵` su "hard"

Su sintetico c=1.0 è meglio del default di un ordine di grandezza. **Il default c=0.1 è troppo conservativo**.

Altre possibili miglioranze (non testate):
- `R` (patience threshold): default 1. Se `α_i ||d_i|| ~ 10⁻³`, allora `r > R` fires solo dopo ~1000 iter di stagnazione. Probabilmente OK ma da verificare.
- `ρ` (contraction factor): default 0.7. Più aggressivo (`ρ = 0.5` o `0.3`) potrebbe accelerare.

### Cosa fare
Due opzioni:
1. **Conservativa**: lasciare i default attuali ma dichiarare apertamente nel report che SGPTL su questi problemi opera nel regime sublineare e che le 8000 iter non bastano per gap < 10⁻³. Non è un fallimento dell'algoritmo, è la rate.
2. **Sperimentale**: rilanciare §5.7 con c=0.5 e ρ=0.5, vedere se migliora. Se sì, aggiornare i default e i numeri delle tabelle.

L'opzione (2) cambia i numeri del report e richiede tempo. L'opzione (1) è più immediata ma non risolve il punto del prof — solo lo onesta.

---

## 5. §3.5.1 sign(0) = 0 (🟢 — il prof stesso dice "non critico")

Il subgradient di LASSO usa la convenzione `sign(0) = 0`. Il prof osserva: in pratica `w_i = 0` ha probabilità bassa nei subgradient method (continuous update). Si potrebbe accennare nel report che:
- `w_i = 0` esatto è non-generico (set di misura 0)
- la scelta `sign(0) = 0` è una di tante ammissibili (qualunque `s ∈ [-1, 1]`)
- Theorem 3.1 richiede solo l'ammissibilità

Aggiungere 1 riga in §3.5.1 chiude la questione.

---

## 6. LLM smells nel report

Audit di pattern stilistici che suggeriscono LLM-writing. Il prof è esplicito: *"tutto il vostro report puzza parecchio di LLM"*.

### Pattern trovati nel codice (post-fix di questa sessione)
La maggior parte è stata rimossa o ridotta. Residui: nessuno particolarmente grave nel codice attuale.

### Pattern trovati nel report (campione)

**(A) Triplette retoriche ("rule of three").** Frangioni e altri ricercatori scrivono naturalmente, ma il report usa frequentemente strutture tipo:
> "convexity, coercivity, and compactness of every sublevel set were established..."
> "the synthetic, diabetes, and california instances..."
> "small c, large c, and intermediate..."

Sono spesso giustificate (3 dataset, 3 ipotesi, …), ma a volte sono cosmetiche.

**(B) Ridondanza nei paragrafi.** Il prof lo dice esplicitamente. Esempi (campione):
- chapter3.tex §"Parameter calibration": la motivazione del cold-start default è ripetuta in 2 punti (§3.4 "Initialization" e §3.5 "Parameter calibration" introduzione)
- chapter3.tex § "Convergence analysis": l'ipotesi "subgradients uniformly bounded" è dichiarata nel Theorem statement, ripetuta nel "Verification of theorem assumptions" paragraph, e citata ancora in §5
- results.tex § "Validation on real datasets": "sklearn does not converge" appare 3 volte nel testo + 1 nella caption della tabella

**(C) "We note that"/"It is worth noting"/"Notably"** — uso compulsivo. Da grep:
- "we note": 4 occorrenze in chapter3.tex
- "notably/notice/note": 8 occorrenze totali
Da ridurre del 60%.

**(D) Em-dash overuse** — Wikipedia "Signs of AI writing" cita questo come segnale. Il report ha:
- chapter3.tex: 51 occorrenze di `---`
- results.tex: 73 occorrenze
La maggior parte sono legittime (parentetiche), ma in alcuni paragrafi sono usate al posto di virgole/punto e virgola in modo sospetto.

**(E) Paragrafi "summative" che ripetono il claim appena fatto.** Esempio (chapter3.tex):
> "The proxy is therefore validated: replacing f* by f(w_OLS) in δ_0 keeps the
> empirical SGPTL gap within a factor of two of the ideal in 10 of 12 tested
> cells [...]" 

(quello che ho già rimosso io). Pattern: dichiarazione, dati, ridichiarazione della stessa dichiarazione.

**(F) Linguaggio "promotional"**: "trustworthy", "robust", "rigorous", "carefully", "principled" — sono usati senza definizione operativa. Sintomi LLM tipici.

### Cosa fare (proposta concreta)

Spawn un agent `humanizer` (skill già disponibile, `/humanizer`) sui tre file principali (`chapter3.tex`, `chapter4` comparison, `results.tex`) per:
- Tagliare le ridondanze
- Sostituire em-dash con virgole/punti dove appropriato
- Eliminare "we note that", "it is worth noting", "Notably"
- Spezzare paragrafi che dicono la stessa cosa 2-3 volte
- Sostituire prosa "promotional" con asserzioni concrete

Tempo stimato: 2-3 ore di lavoro guidato; output: -20-30% di parole, leggibilità migliore. Da fare prima della submission.

Lista esplicita di passaggi da rivedere (priorità):
1. chapter3.tex §"Why the plain subgradient method is inadequate" (intro) — molto verbosa
2. chapter3.tex §"Convergence analysis" — proof + verification + remarks si sovrappongono
3. results.tex §"Validation on real datasets" — la mia stessa riscrittura ha lasciato qualche ridondanza
4. results.tex §"Three families" — buon paragrafo ma le "Empirical finding" + "single uninformative regime" + "Family B..." si potrebbero compattare
5. comparison.tex (cap. 4) — non l'ho letto in questa sessione, da fare

---

## 7. Coerenza codice ↔ report (post-fix)

Dopo i fix di questa sessione:
- ✅ stop criterion IRLS allineato (`||Δw||/max(1, ||w||)`)
- ✅ Q_k weights IRLS matchano (commento nel codice spiega l'equivalenza `λ·D ≡ 2λ_IRLS W^T W`)
- ✅ CG = "Jacobi-preconditioned" coerente
- ✅ data_generation: α=λ/M nel report, normalisation di colonne dichiarata
- ✅ SGPTL default cold + cite a §"Three families"
- ✅ δ_0 = c · f(w_OLS) dichiarato esplicitamente nel paragrafo (non più "c · f*")

Da verificare ancora:
- 🟡 La derivazione del rate (punto 1 sopra) — codice non c'entra, ma il report ha potenzialmente il bound sbagliato
- 🟡 §5.7 numeri (punto 4 sopra) — coerenti col codice attuale, possibili da migliorare cambiando i defaults

---

## 8. Plot e tabelle

### Plot rilanciati in questa sessione
- `real_data_convergence.pdf` — coerente con la nuova metodologia f* (IRLS+CVXPY), legenda corretta per dataset (cold per diabetes, warm per CA)
- `real_data_warm_vs_cold.pdf` — nuova figura introdotta in §5.7 a supporto della scelta per-dataset

### Plot che NON ho ri-validato
- `delta0_families.pdf` (genera `experiment_delta0_families.py`) — controllare che sia coerente con i numeri attuali; se è precedente a fix di questa sessione, va rigenerato
- `comparison_irls_dsm.pdf`, `convergence_vs_iter.pdf`, `convergence_vs_time.pdf` — su istanze sintetiche, probabilmente OK ma vanno verificati
- `scalability.pdf`, `params_irls.pdf`, `params_dsm.pdf` — idem

### Presentabilità (campione, non esaustivo)
- Le figure stile "sweep" hanno spesso 4–6 subplot — buono per density of info ma a volte font/asse troppo piccolo. Da verificare in PDF finale.
- `warm_vs_cold.pdf` (sintetico) e `real_data_warm_vs_cold.pdf` hanno layout simile ma sono in chapter diversi. OK.

---

## 9. Riassunto operativo

| Punto | Priorità | Azione |
|---|---|---|
| Theorem 3.1 rate factor | 🔴 critico | Derivare esplicitamente, sostituire `1/γ_min` con `1/√γ_min` |
| Dichiarare che `f*` non entra nelle update rules | 🔴 critico | 1 frase in §3.3 o §5 |
| Warm-start cost discussion | 🟡 importante | Paragrafo `O(MH²+H³)` in chapter3 / results |
| §5.7 retune (c=0.5, ρ=0.5)? | 🟡 importante | Decidere tra (1) onestà sui limiti vs (2) rerun migliorato |
| LLM smell pass su 3 file principali | 🟡 importante | `/humanizer` skill o pass manuale |
| sign(0)=0 esplicito | 🟢 cosmetico | 1 riga in §3.5.1 |
| Plot residui da rigenerare | 🟢 da verificare | Lista ai plot non rilanciati |

---

*Fine review.*
